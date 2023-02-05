from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
import tensorflow as tf
from conversion_tf import GEMMDecisionTreeImpl
import numpy as np

def representative_dataset():
    for _ in range(100):
      data = np.random.uniform(low=0., high=8., size=(300,8))
      yield [data.astype(np.float32)]
 

forest = RandomForestClassifier(n_estimators=105)
X, y = make_classification(n_samples=1300, n_features=8,
                           n_informative=4, n_redundant=1,
                           random_state=0, shuffle=True,
                           n_classes=4)

x_train, y_train = X[:1000], y[:1000]
x_test, y_test = X[1000:], y[1000:]

forest.fit(x_train, y_train)


model = GEMMDecisionTreeImpl(forest)
X = tf.constant([1, 2, 3, 4, 5, 6, 7, 8], shape=[1, 8], dtype=tf.int8)
X_float = tf.constant([1., 2., 3., 4., 5., 6., 7., 8.], shape=[1, 8])

#y_pred, y = model(X_float)

concrete_func = model.__call__.get_concrete_function()

converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func], model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8  
converter.inference_output_type = tf.int8  
tflite_model = converter.convert()

interpreter = tf.lite.Interpreter(model_content=tflite_model)
interpreter.allocate_tensors()

output = interpreter.get_output_details()[0]  
input = interpreter.get_input_details()[0]  


#interpreter.set_tensor(input['index'], X)
#interpreter.invoke()
#y_lite = interpreter.get_tensor(output['index'])
#
#y_pred_lite = np.argmax(y_lite, axis=1)

#print(f'Prediction     : {y_pred.numpy()[0]} \t Class Confidence: {y.numpy()[0]}')
#print(f'Prediction lite: {y_pred_lite[0]} \t Class Confidence: {y_lite[0]}')

preds, _ = model(x_test)
diff = preds - y_test
correct = diff[diff == 0].shape[0]
total = diff.shape[0]

interpreter.set_tensor(input['index'], x_test.astype(np.int8))
interpreter.invoke()
test_pred_quant = interpreter.get_tensor(output['index'])

preds_quant = np.argmax(test_pred_quant, axis=1)

diff_quant = preds_quant - y_test
correct_quant = diff_quant[diff_quant == 0].shape[0]
total = diff_quant.shape[0]

print(f'Accuracy pre  quantization: {correct / total * 100 : .2f}%')
print(f'Accuracy post quantization: {correct_quant / total * 100 : .2f}%')

