from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
import tensorflow as tf
from conversion_tf import GEMMDecisionTreeImpl, TreeTraversalDecisionTreeImpl
import numpy as np
from hummingbird.ml import convert
import torch

###################################### DEFINITIONS #################################################################

def representative_dataset():
    for _ in range(100):
      data = np.random.uniform(low=0., high=8., size=(1,8))
      yield [data.astype(np.float32)]
 
tf.config.run_functions_eagerly(True)

forest = RandomForestClassifier(n_estimators=105)
X, y = make_classification(n_samples=1300, n_features=8,
                           n_informative=4, n_redundant=1,
                           random_state=0, shuffle=True,
                           n_classes=4)

x_train, y_train = X[:1000], y[:1000]
x_test, y_test = X[1000:], y[1000:]

forest.fit(x_train, y_train)

X = tf.constant([1, 2, 3, 4, 5, 6, 7, 8], shape=[1, 8], dtype=tf.int8)
X_float = tf.constant([1., 2., 3., 4., 5., 6., 7., 8.], shape=[1, 8])


##################################### WORKING TFLITE FOR TT ############################################################

conv_model = convert(forest, 'torch', extra_config={"tree_implementation":"tree_trav"})

y_mod_pred_tt, y_mod_tt = conv_model.model._operators[0].forward((torch.tensor([[1., 2., 3., 4., 5., 6., 7., 8.]])))

model_tt = TreeTraversalDecisionTreeImpl(forest)

y_pred_tt, y_tt = model_tt(X_float)

concrete_func = model_tt.__call__.get_concrete_function()

converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func], model_tt)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8  
converter.inference_output_type = tf.int8  
tflite_model_tt = converter.convert()

interpreter = tf.lite.Interpreter(model_content=tflite_model_tt)
interpreter.allocate_tensors()

output = interpreter.get_output_details()[0]  
input = interpreter.get_input_details()[0]  

interpreter.set_tensor(input['index'], X)
interpreter.invoke()
y_lite_tt = interpreter.get_tensor(output['index'])

y_pred_lite_tt = np.argmax(y_lite_tt, axis=1)


###################################### WORKING TFLITE FOR GEMM ##########################################################

model_gemm = GEMMDecisionTreeImpl(forest)

conv_model = convert(forest, 'torch', extra_config={"tree_implementation":"gemm"})

y_mod_pred_gemm, y_mod_gemm = conv_model.model._operators[0].forward((torch.tensor([[1., 2., 3., 4., 5., 6., 7., 8.]])))

y_pred_gemm, y_gemm = model_gemm(X_float)

concrete_func = model_gemm.__call__.get_concrete_function()

converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func], model_gemm)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8  
converter.inference_output_type = tf.int8  
tflite_model_gemm = converter.convert()

interpreter = tf.lite.Interpreter(model_content=tflite_model_gemm)
interpreter.allocate_tensors()

output = interpreter.get_output_details()[0]  
input = interpreter.get_input_details()[0]  


interpreter.set_tensor(input['index'], X)
interpreter.invoke()
y_lite_gemm = interpreter.get_tensor(output['index'])

y_pred_lite_gemm = np.argmax(y_lite_gemm, axis=1)

##################### ACCURACY TEST WITH BATCH SIZE 300 COMMENTED WHILE BATCH SIZE ABOVE IS 1 ###########################
#diff = preds - y_test
#correct = diff[diff == 0].shape[0]
#total = diff.shape[0]
#
#interpreter.set_tensor(input['index'], x_test.astype(np.int8))
#interpreter.invoke()
#test_pred_quant = interpreter.get_tensor(output['index'])
#
#preds_quant = np.argmax(test_pred_quant, axis=1)
#
#diff_quant = preds_quant - y_test
#correct_quant = diff_quant[diff_quant == 0].shape[0]
#total = diff_quant.shape[0]
#
#print(f'Accuracy pre  quantization: {correct / total * 100 : .2f}%')
#print(f'Accuracy post quantization: {correct_quant / total * 100 : .2f}%')


############################################ RESULTS ###############################################
print(f'############################################ GEMM #############################################################\n')
print(f'Prediction original    : {y_mod_pred_tt.detach().numpy()[0]} \t Class Confidence: {y_mod_tt.detach().numpy()[0]}')
print(f'Prediction tf module   : {y_pred_tt.numpy()[0]} \t Class Confidence: {y_tt.numpy()[0]}')
print(f'Prediction tflite quant: {y_pred_lite_tt[0]} \t Class Confidence: {y_lite_tt[0]}')
print(f'\n############################################ TT ###############################################################\n')
print(f'Prediction original    : {y_mod_pred_gemm.detach().numpy()[0]} \t Class Confidence: {y_mod_gemm.detach().numpy()[0]}')
print(f'Prediction tf module   : {y_pred_gemm.numpy()[0]} \t Class Confidence: {y_gemm.numpy()[0]}')
print(f'Prediction tflite quant: {y_pred_lite_gemm[0]} \t Class Confidence: {y_lite_gemm[0]}')
