import tensorflow as tf
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from conversion_tf import GEMMDecisionTreeImpl
from hummingbird.ml import convert

def generate_model(bs, nf, nt, md):
    # variables 
    BATCH_SIZE = bs
    N_FEATURES = nf
    N_TREES = nt
    MAX_DEPTH = md
    
    # init random forest
    forest = RandomForestClassifier(n_estimators=N_TREES, max_depth=MAX_DEPTH)

    # representative dataset for tflite conversion and int8 quantization
    def representative_dataset():
        for _ in range(100):
          data = np.random.uniform(low=0., high=10., size=(BATCH_SIZE, N_FEATURES))
          yield [data.astype(np.float32)]
    
    tf.config.run_functions_eagerly(True)

    X, y = make_classification(n_samples=1300, n_features=N_FEATURES,
                               n_informative=N_FEATURES, n_redundant=0,
                               random_state=0, shuffle=True,
                               n_classes=4)

    x_train, y_train = X[:1000], y[:1000]
    x_test, y_test = X[1000:], y[1000:]

    forest.fit(x_train, y_train)

    # generate input data
    X_f_b = np.random.randint(low=0, high=10, size=(BATCH_SIZE, N_FEATURES)).astype(np.int8)
    X_f_b_f = X_f_b.astype(np.float32)



    conv_model = convert(forest, 'torch', extra_config={"tree_implementation":"gemm"})

    # initialize model
    model_gemm = GEMMDecisionTreeImpl(forest)

    # get model prediction
    y_gemm = model_gemm(X_f_b_f)

    concrete_func = model_gemm.__call__.get_concrete_function()

    # convert model to tflite
    converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func], model_gemm)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8  
    converter.inference_output_type = tf.int8

    tflite_model_gemm = converter.convert()

    # interprete tflite model
    interpreter = tf.lite.Interpreter(model_content=tflite_model_gemm)
    interpreter.allocate_tensors()

    output = interpreter.get_output_details()[0]  
    input = interpreter.get_input_details()[0]  

    interpreter.set_tensor(input['index'], X_f_b)
    interpreter.invoke()
    y_lite_gemm = interpreter.get_tensor(output['index'])
    y_pred_lite_gemm = np.argmax(y_lite_gemm, axis=1)

    # save tflite model
    with open(f'./saved_models/random_forest/gemm/float32/final_eval/model_{N_TREES}_{MAX_DEPTH}_{N_FEATURES}_{BATCH_SIZE}.tflite', 'wb') as f:
        f.write(tflite_model_gemm)

def main():
    bsizes = [1, 64, 256, 512]
    nfeat  = [2, 8, 16]
    mdepth = [1, 4, 16]
    n_trees = [1, 32, 128]

    for max_depth in mdepth:
        for n_tree in n_trees:
            generate_model(256, 16, n_tree, max_depth)


if __name__ == '__main__':
    main()