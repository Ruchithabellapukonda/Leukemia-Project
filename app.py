import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
import cv2
import base64
import gradcam

app = Flask(__name__)

# Load Model
MODEL_PATH = 'leukemia_alexnet_model.h5'
print("Loading Model...")
model = tf.keras.models.load_model(MODEL_PATH)

# Wake up model
dummy = np.zeros((1, 224, 224, 3))
_ = model(dummy)

# Find Layer
target_layer = None
for layer in reversed(model.layers):
    if 'conv' in layer.name.lower():
        target_layer = layer.name
        break

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

@app.route('/predict', methods=['POST'])
def predict():
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

        # Heatmap
        hm = gradcam.make_gradcam_heatmap(img, model, target_layer)
        hm_path = gradcam.save_and_display_gradcam("temp.jpg", hm)
        hm_b64 = image_to_base64(hm_path)

        return jsonify({'diagnosis': result, 'confidence': f"{conf:.2f}", 'heatmap': hm_b64})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)