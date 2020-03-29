import base64
import json
import os
import numpy as np
import tensorflow as tf

from PIL import Image
from io import BytesIO
from flask import Flask, request, jsonify

from architecture import LeNet, preprocess, make_predictions


app = Flask(__name__)


def load_model():
    model_path = 'weights.h5'
    model = LeNet()
    model.load_weights(model_path)
    return model


def read_image(files):
    """
    files: bytes
    """
    image = Image.open(files)
    image = np.array(image)
    return image


@app.route('/hello/', methods=['GET'])
def hello():
    return "Hello World"

@app.route('/predict/', methods=['POST'])
def predict():
    model = load_model()

    # image -> bytes-> decode -> numpyarray -> tenor -> resize -> model fit
    files = request.files['images']
    
    #numpy array
    image = read_image(files)

    # tensor, resize, standardize
    image = preprocess(image)
    
    predicted = model(image)

    value = make_predictions(predicted)

    print(value)
    #finding class with highest probability
     
    maxm = tf.argmax(value,axis=1)
    print(maxm)
    
    response = {
        'predicted_number': np.array_str(maxm.numpy())
    }
    return json.dumps(response)


if __name__ == "__main__":
    app.run()