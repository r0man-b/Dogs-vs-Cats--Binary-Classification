import numpy as np
import pandas as pd
import keras
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import img_to_array, load_img
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image

def main():
    model = load_model('model.h5')
    print()
    while True:
        print("Welcome to the Cat or Dog image distinguisher, please enter the name of your image file:")
        usr_input = input()
        print()
        test_image = image.load_img(usr_input,
                            target_size = (128, 128))
        test_image = image.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis = 0)
        result = model.predict(test_image)
        if result[0][0] == 1:
            prediction = 'I think that\'s a dog!'
        else:
            prediction = 'I think that\'s a cat!'
        print(prediction)
        print()
        print("Continue? Y/N")
        usr_input = input()
        if usr_input == 'N':
            return

if __name__ == '__main__':
    main()
