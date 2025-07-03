import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
import numpy as np

model = tf.keras.applications.MobileNetV2(weights='imagenet')

def load_model_and_predict(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = preprocess_input(np.expand_dims(x, axis=0))
    preds = model.predict(x)
    label = decode_predictions(preds, top=1)[0][0][1]
    return label
