from flask import Flask, request, jsonify
import numpy as np
import tensorflow as tf
from PIL import Image
import faiss
import os
import traceback

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load model
model = tf.keras.applications.MobileNetV2(weights='imagenet', include_top=False, pooling='avg')

try:
    product_ids = np.load('product_ids.npy')
    features_db = np.load('features_db.npy')

    # Ensure features and IDs match
    assert features_db.shape[0] == product_ids.shape[0], "Mismatch in features and product IDs"

    # Create FAISS index
    index = faiss.IndexFlatL2(features_db.shape[1])
    index.add(features_db)
except Exception as e:
    print("🔥 Error loading model/data:", e)
    traceback.print_exc()

def extract_features(image_path):
    try:
        img = Image.open(image_path).resize((224, 224)).convert('RGB')
        x = tf.keras.preprocessing.image.img_to_array(img)
        x = tf.keras.applications.mobilenet_v2.preprocess_input(np.expand_dims(x, axis=0))
        features = model.predict(x)[0]
        return features
    except Exception as e:
        print("❌ Error in feature extraction:", e)
        traceback.print_exc()
        return None

@app.route('/search', methods=['POST'])
def search():
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image uploaded'}), 400
        file = request.files['image']
        filepath = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(filepath)

        query_vector = extract_features(filepath)
        if query_vector is None:
            return jsonify({'error': 'Feature extraction failed'}), 500

        query_vector = query_vector.reshape(1, -1)
        _, indices = index.search(query_vector, 5)
        matched_ids = [str(product_ids[i]) for i in indices[0]]
        return jsonify(matched_ids)
    except Exception as e:
        print("❌ Error in /search route:", e)
        traceback.print_exc()
        return jsonify({'error': 'Internal Server Error'}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
