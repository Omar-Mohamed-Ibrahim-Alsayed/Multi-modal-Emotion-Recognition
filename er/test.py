import tensorflow as tf
import numpy as np
from skimage.io import imread
from skimage.transform import resize



class TestModels:
    def __init__(self, h5_address:str, GPU=True):
        self.exps = ['Neutral', 'Happy', 'Sad', 'Surprise', 'Fear', 'Disgust', 'Anger']
        self.model = self.load_model(h5_address=h5_address)

    def load_model(self, h5_address: str):
        "load weight file and create the model once"
        model = tf.keras.models.load_model(h5_address, custom_objects={'tf': tf})
        return model

    def recognize_fer(self, img_path:str):
        "create and image from the path and recognize expression"
        img = imread(img_path)
        # resize img to 1*224*224*3
        img = resize(img, (224, 224,3))
        img = np.expand_dims(img, axis=0)
        #
        prediction = self.model.predict_on_batch([img])
        exp = np.array(prediction[0])
        '''in case you need the embeddings'''
        # embeddings = prediction[1:]
        print(exp)
        print(self.exps[np.argmax(exp)])

    def recognize_fer2(self, img):
        "create and image from the path and recognize expression"
        img = resize(img, (224, 224,3))
        img = np.expand_dims(img, axis=0)
        #
        prediction = self.model.predict_on_batch([img])
        exp = np.array(prediction[0])
        '''in case you need the embeddings'''
        # embeddings = prediction[1:]
        return self.exps[np.argmax(exp)],exp

