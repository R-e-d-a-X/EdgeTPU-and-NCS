#from pycoral.utils import edgetpu 
import time
import tensorflow as tf
import numpy as np

# hardcoded paths
MODEL_PATH = './saved_models/random_forest/gemm/float32/one_tree_depth_1/model11.tflite'
COUNT = 101

def main():
    np.random.seed(0)

    # input for batch size = 1
    x = tf.constant([1,2,3,4,5,6,7,8], shape=[1,8], dtype=tf.int8)

    # prepare file for results
    #f = open('results.csv', 'w')
    #f.write("TimeMS\n")


    # create interpreter for tflite-model
    # if meant to run on EdgeTPU change tf.lite.Interpreter(MODEL) to edgetpu.make_interpreter(MODEL)
    interpreter = tf.lite.Interpreter(model_path=MODEL_PATH) 
    #interpreter = edgetpu.make_interpreter(MODEL_PATH)
    interpreter.allocate_tensors()
    
    output = interpreter.get_output_details()[0]  
    input = interpreter.get_input_details()[0] 


    # set model input und run inference 
    interpreter.set_tensor(input['index'], x)
    for _ in range(COUNT):
        start = time.perf_counter()
        interpreter.invoke()
        inf_time_s = time.perf_counter() - start

        inf_time_ms = inf_time_s * 1000
        #f.write(f'{inf_time_ms}\n')

        output = interpreter.get_output_details()[0]  

        y_lite_gemm = interpreter.get_tensor(output['index'])
        y_pred_lite_gemm = np.argmax(y_lite_gemm, axis=1) 

        # index label with class.id and get classification-confidence with class.score
        print(f'Prediction: {y_pred_lite_gemm} \t Confidence: {y_lite_gemm} \t Time: {inf_time_ms:.2f}ms')

    #f.close()

if __name__ == '__main__':
    main()