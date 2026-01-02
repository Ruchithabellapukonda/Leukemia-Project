import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from flask import Flask, request, jsonify, render_template
import tensorflow as tf
import numpy as np
import cv2
import base64
import gradcam
from tensorflow.keras.layers import InputLayer

# --- PATCH 1: Fix batch_shape ---
class PatchedInputLayer(InputLayer):
    def __init__(self, **kwargs):
        if 'batch_shape' in kwargs:
            kwargs['batch_input_shape'] = kwargs.pop('batch_shape')
        super().__init__(**kwargs)

# --- PATCH 2: Fix DTypePolicy (UPDATED) ---
# We are adding more details to the fake ID card so Keras is happy.
class DTypePolicy:
    def __init__(self, name="float32", **kwargs):
        self.name = name
        self.compute_dtype = name   # <--- THIS WAS MISSING
        self.variable_dtype = name  # Adding this to be extra safe
        self._name = name

    def get_config(self):
        return {"name": self.name}

    @classmethod
    def from_config(cls, config):
        return cls(**config)

app = Flask(__name__)

MODEL_PATH = 'leukemia_alexnet_model.h5'
print("Loading Model...")

try:
    # Register our smart patches
    custom_objects = {
        'InputLayer': PatchedInputLayer,
        'DTypePolicy': DTypePolicy
    }
    
    with tf.keras.utils.custom_object_scope(custom_objects):
        model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    print("Model Loaded Successfully!")
except Exception as e:
    print(f"CRITICAL ERROR LOADING MODEL: {e}")
    model = None

# Wake up model
if model:
    try:
        dummy = np.zeros((1, 224, 224, 3))
        _ = model(dummy)
        
        target_layer = None
        for layer in reversed(model.layers):
            if 'conv' in layer.name.lower():
                target_layer = layer.name
                break
    except Exception as e:
        print(f"Error waking up model: {e}")

CLASS_NAMES = ['Benign', 'Early', 'Pre', 'Pro']

def prepare_image(path):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))
    img = img / 255.0
    return np.expand_dims(img, axis=0)

def image_to_base64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode('utf-8')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if not model:
        return jsonify({'error': 'Model failed to load on server start'}), 500

    if 'file' not in request.files:
        return jsonify({'error': 'No file'}), 400
    
    file = request.files['file']
    file.save("temp.jpg")

    try:
        img = prepare_image("temp.jpg")
        preds = model.predict(img)
        idx = np.argmax(preds)
        result = CLASS_NAMES[idx]
        conf = float(np.max(preds)) * 100

        hm_b64 = None
        if target_layer:
            try:
                hm = gradcam.make_gradcam_heatmap(img, model, target_layer)
                hm_path = gradcam.save_and_display_gradcam("temp.jpg", hm)
                hm_b64 = image_to_base64(hm_path)
            except:
                pass 

        return jsonify({'diagnosis': result, 'confidence': f"{conf:.2f}", 'heatmap': hm_b64})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)