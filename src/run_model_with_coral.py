from PIL import Image
import tensorflow as tf
from pycoral.utils.dataset import read_label_file
from pycoral.utils import edgetpu 
from pycoral.adapters import common
from pycoral.adapters import classify

# hardcoded paths
MODEL_PATH = '/home/tobi/Bachelorarbeit/EdgeTPU-and-NCS/test_data/efficientnet-edgetpu-L_quant.tflite'
INPUT_PATH = '/home/tobi/Bachelorarbeit/EdgeTPU-and-NCS/test_data/parrot.jpg'
LABEL_PATH = '/home/tobi/Bachelorarbeit/EdgeTPU-and-NCS/test_data/imagenet_labels.txt'

def main():

    # read inputimage and class labels
    img = Image.open(INPUT_PATH)
    labels = read_label_file(LABEL_PATH)

    # create interpreter for tflite-model
    # if meant to run on EdgeTPU change tf.lite.Interpreter(MODEL) with edgetpu.make_interpreter(MODEL)
    interpreter = tf.lite.Interpreter(model_path=MODEL_PATH) 
    interpreter.allocate_tensors()
    
    # get inputsize and resize input image accordingly
    size = common.input_size(interpreter=interpreter)
    img = img.resize(size, Image.ANTIALIAS)
    img.show()
    
    # set model input und run inference 
    common.set_input(interpreter, img)
    interpreter.invoke()

    # get top_k classification-classes (in this case 1) 
    classes = classify.get_classes(interpreter, top_k=1)

    # index label with class.id and get classification-confidence with class.score
    for c in classes:
        print(f'Prediction: {labels[c.id]} \t Confidence: {c.score * 100 :.2f}%')


if __name__ == '__main__':
    main()