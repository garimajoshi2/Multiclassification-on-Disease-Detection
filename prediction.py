import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import tensorflow as tf
from tensorflow.keras.preprocessing import image as keras_image

class DiseasePredictor:
    def __init__(self, model_name, image_path):
        # Load the custom layer definition
        class SelfAttention(tf.keras.layers.Layer):
            def __init__(self, **kwargs):
                super(SelfAttention, self).__init__(**kwargs)

            def build(self, input_shape):
                self.w_q = self.add_weight(name='q_kernel',    
                                           shape=(input_shape[-1], input_shape[-1]),
                                           initializer='uniform',
                                           trainable=True)
                self.w_k = self.add_weight(name='k_kernel',
                                           shape=(input_shape[-1], input_shape[-1]),
                                           initializer='uniform',
                                           trainable=True)
                self.w_v = self.add_weight(name='v_kernel',
                                           shape=(input_shape[-1], input_shape[-1]),
                                           initializer='uniform',
                                           trainable=True)
                super(SelfAttention, self).build(input_shape)

            def call(self, x):    
                q = tf.matmul(x, self.w_q)
                k = tf.matmul(x, self.w_k)
                v = tf.matmul(x, self.w_v)
                attn_scores = tf.matmul(q, k, transpose_b=True)
                attn_scores = tf.nn.softmax(attn_scores, axis=-1)
                output = tf.matmul(attn_scores, v)
                return output

        # Register the custom layer
        tf.keras.utils.get_custom_objects().update({'SelfAttention': SelfAttention})

        # Define class labels dictionary
        self.class_labels_dict = {
        
    0: 'Apple___Apple_scab',
    1: 'Apple___Black_rot',
    2: 'Apple___Cedar_apple_rust',
    3: 'Apple___healthy',
    4: 'Blueberry___healthy',
    5: 'Cherry_(including_sour)___Powdery_mildew',
    6: 'Cherry_(including_sour)___healthy',
    7: 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
    8: 'Corn_(maize)___Common_rust_',
    9: 'Corn_(maize)___Northern_Leaf_Blight',
    10: 'Corn_(maize)___healthy',
    11: 'Grape___Black_rot',
    12: 'Grape___Esca_(Black_Measles)',
    13: 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
    14: 'Grape___healthy',
    15: 'Orange___Haunglongbing_(Citrus_greening)',
    16: 'Peach___Bacterial_spot',
    17: 'Peach___healthy',
    18: 'Pepper,_bell___Bacterial_spot',
    19: 'Pepper,_bell___healthy',
    20: 'Potato___Early_blight',
    21: 'Potato___Late_blight',
    22: 'Potato___healthy',
    23: 'Raspberry___healthy',
    24: 'Soybean___healthy',
    25: 'Squash___Powdery_mildew',
    26: 'Strawberry___Leaf_scorch',
    27: 'Strawberry___healthy',
    28: 'Tomato___Bacterial_spot',
    29: 'Tomato___Early_blight',
    30: 'Tomato___Late_blight',
    31: 'Tomato___Leaf_Mold',
    32: 'Tomato___Septoria_leaf_spot',
    33: 'Tomato___Spider_mites Two-spotted_spider_mite',
    34: 'Tomato___Target_Spot',
    35: 'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
    36: 'Tomato___Tomato_mosaic_virus',
    37: 'Tomato___healthy'
            # Class labels dictionary here...
        }

        # Load the model
        self.loaded_model = tf.keras.models.load_model(model_name)
        self.image_path = image_path

    def predict_disease(self):
        # Load and preprocess the image
        new_image = keras_image.load_img(self.image_path, target_size=(224, 224))
        new_image_array = keras_image.img_to_array(new_image)
        new_image_array = np.expand_dims(new_image_array, axis=0)
        new_image_array = new_image_array / 255.0  # Normalize the image

        # Get the predicted class probabilities
        predicted_probabilities = self.loaded_model.predict(new_image_array)[0]
        predicted_class_index = np.argmax(predicted_probabilities)

        # Check if the predicted_class_index is present in the dictionary
        if predicted_class_index in self.class_labels_dict:
            predicted_class_label = self.class_labels_dict[predicted_class_index]
        else:
            predicted_class_label = 'Unknown Class'

        return predicted_class_label

    def display_prediction(self):
        # Display the image and prediction
        new_image = Image.open(self.image_path)
        plt.imshow(new_image)
        plt.title(f"Predicted Class: {self.predict_disease()}")
        plt.axis('off')
        plt.show()



