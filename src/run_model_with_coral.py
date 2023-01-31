from PIL import Image
import tensorflow as tf # If run on EdgeTPU comment this out and add commented line
#from pycoral.utils import edgetpu 
from pycoral.utils.dataset import read_label_file
from pycoral.adapters import common
from pycoral.adapters import classify
import time

# hardcoded paths
MODEL_PATH = '/home/tobi/Bachelorarbeit/EdgeTPU-and-NCS/saved_models/efficientnet/effnet-edgetpu-S/efficientnet-edgetpu-S_quant.tflite'
INPUT_PATH = '/home/tobi/Bachelorarbeit/EdgeTPU-and-NCS/test_data/bird.jpg'
LABEL_PATH = '/home/tobi/Bachelorarbeit/EdgeTPU-and-NCS/test_data/imagenet_labels.txt'
COUNT = 101

def main():

    # read inputimage and class labels
    img = Image.open(INPUT_PATH)
    labels = read_label_file(LABEL_PATH)

    # prepare file for results
    f = open('results.csv', 'w')
    f.write("TimeMS\n")


    # create interpreter for tflite-model
    # if meant to run on EdgeTPU change tf.lite.Interpreter(MODEL) with edgetpu.make_interpreter(MODEL)
    interpreter = tf.lite.Interpreter(model_path=MODEL_PATH) 
    interpreter.allocate_tensors()
    
    # get inputsize and resize input image accordingly
    size = common.input_size(interpreter=interpreter)
    img = img.resize(size, Image.ANTIALIAS)
    #img.show()
    
    # set model input und run inference 
    common.set_input(interpreter, img)
    for _ in range(COUNT):
        start = time.perf_counter()
        interpreter.invoke()
        inf_time_s = time.perf_counter() - start

        inf_time_ms = inf_time_s * 1000

        # get top_k classification-classes (in this case 2) 
        classes = classify.get_classes(interpreter, top_k=1)
        f.write(f'{inf_time_ms}\n')

        # index label with class.id and get classification-confidence with class.score
        for c in classes:
            print(f'Prediction: {labels[c.id]} \t Confidence: {c.score * 100 :.2f}% \t Time: {inf_time_ms:.2f}ms')

    f.close()

if __name__ == '__main__':
    main()