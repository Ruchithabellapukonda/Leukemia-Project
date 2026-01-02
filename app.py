import os
import numpy as np
import tensorflow as tf
from flask import Flask, request, render_template, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import InputLayer
import cv2
import gradcam  # Import our heatmap helper

# --- 1. PATCHES FOR COMPATIBILITY (The Time Travel Fixes) ---

# FIX A: Handle "batch_shape" vs "batch_input_shape" mismatch
class PatchedInputLayer(InputLayer):
    def __init__(self, **kwargs):
        if 'batch_shape' in kwargs:
            kwargs['batch_input_shape'] = kwargs.pop('batch_shape')
        super().__init__(**kwargs)

    def get_config(self):
        config = super().get_config()
        return config

# FIX B: Handle "DTypePolicy" mismatch (The error you just saw)
class DTypePolicy:
    def __init__(self, *args, **kwargs):
        # We just ignore whatever policy settings the model had
        pass
    
    def get_config(self):
        return {}

    @classmethod
    def from_config(cls, config):
        return cls(**config)

# --- 2. SETUP FLASK & LOAD MODEL ---
app = Flask(__name__)

# Config
MODEL_PATH = 'leukemia_alexnet_model.h5'
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

print("Loading Model... Please wait...")
try:
    # Load model with BOTH custom patches
    model = load_model(MODEL_PATH, custom_objects={
        'InputLayer': PatchedInputLayer,
        'DTypePolicy': DTypePolicy  # <--- This fixes the new error
    })
    print("✅ Model Loaded Successfully!")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    # We exit because if the model doesn't load, the app is useless
    exit(1)

# The classes your model knows (Must match training order)
CLASSES = ['Benign', 'Early', 'Pre', 'Pro']

# --- 3. ROUTES ---

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'})

    if file:
        # Save file temporarily
        filename = "temp_upload.jpg"
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(file_path)

        # --- IMAGE PREPROCESSING (THE LOGIC FIX) ---
        try:
            # 1. Read Image
            img = cv2.imread(file_path)
            
            # 2. Fix Color: OpenCV loads as BGR, but Keras needs RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # 3. Resize: Force it to 224x224 (AlexNet standard)
            img = cv2.resize(img, (224, 224))
            
            # 4. Normalize: Convert 0-255 (integers) to 0-1 (decimals)
            img_array = img / 255.0
            
            # 5. Add Batch Dimension: Change shape from (224,224,3) to (1,224,224,3)
            img_array = np.expand_dims(img_array, axis=0)

            # --- PREDICTION ---
            predictions = model.predict(img_array)
            
            # DEBUG: Print raw numbers to terminal
            print(f"🔍 RAW PROBABILITIES: {predictions}")

            # Get the highest score
            class_index = np.argmax(predictions[0])
            predicted_label = CLASSES[class_index]
            confidence = float(np.max(predictions[0])) * 100

            # --- GRAD-CAM HEATMAP ---
            try:
                # Find the last convolutional layer automatically
                target_layer = None 
                for layer in reversed(model.layers):
                    if 'conv' in layer.name:
                        target_layer = layer.name
                        break
                
                print(f"🔥 Generating Heatmap using layer: {target_layer}")
                
                heatmap = gradcam.make_gradcam_heatmap(img_array, model, target_layer)
                
                # Save heatmap result
                heatmap_filename = "heatmap_result.jpg"
                heatmap_path = os.path.join(UPLOAD_FOLDER, heatmap_filename)
                
                # Superimpose on original image
                gradcam.save_and_display_gradcam(file_path, heatmap, alpha=0.4)
                
                # Move the result to static folder
                if os.path.exists("heatmap_result.jpg"):
                    os.rename("heatmap_result.jpg", heatmap_path)
                    
            except Exception as hm_error:
                print(f"⚠️ Heatmap Failed: {hm_error}")
                heatmap_filename = None 

            # Return JSON result
            return jsonify({
                'prediction': predicted_label,
                'confidence': f"{confidence:.2f}",
                'heatmap_url': f"/{UPLOAD_FOLDER}/{heatmap_filename}" if heatmap_filename else None
            })

        except Exception as e:
            print(f"❌ Prediction Error: {e}")
            return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)