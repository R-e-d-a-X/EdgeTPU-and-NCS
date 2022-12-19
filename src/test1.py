import numpy as np
from PIL import Image
import tensorflow as tf


def main():
    #labels = read_label_file('/home/tobi/Bachelorarbeit/EdgeTPU-and-NCS/test_data/imagenet_labels.txt')
    img = Image.open('/home/tobi/Bachelorarbeit/EdgeTPU-and-NCS/test_data/parrot.jpg')
    img = img.resize((300, 300))
    img.show()
    input_data = tf.reshape(tf.convert_to_tensor(img), (1, 300, 300, 3))
    print(input_data.shape)

    interpreter = tf.lite.Interpreter(model_path='/home/tobi/Bachelorarbeit/EdgeTPU-and-NCS/test_data/efficientnet-edgetpu-L_quant.tflite')
    interpreter.allocate_tensors()
    output = interpreter.get_output_details()[0]
    input = interpreter.get_input_details()[0]
    interpreter.set_tensor(input['index'], input_data)
    interpreter.invoke()
    result = interpreter.get_tensor(output['index'])
    print(f'Shape: {result.shape}')
    print(f'Tensor: {result} \t 1: {np.argmax(result)}')




    



if __name__ == '__main__':
    main()