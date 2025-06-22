from flask import Flask, request, jsonify
import numpy as np
from PIL import Image
import faiss
import os
import traceback
import torch
import torchvision.transforms as transforms
from torchvision.models import mobilenet_v2

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load PyTorch MobileNetV2 model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = mobilenet_v2(pretrained=True)
model.classifier = torch.nn.Identity()  # Remove classification head
model.eval()
model.to(device)

# Image preprocessor
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Load features
try:
    product_ids = np.load('product_ids.npy')
    features_db = np.load('features_db.npy').astype(np.float32)

    assert features_db.shape[0] == product_ids.shape[0], "Mismatch in features and product IDs"

    index = faiss.IndexFlatL2(features_db.shape[1])
    index.add(features_db)
except Exception as e:
    print("🔥 Error loading model/data:", e)
    traceback.print_exc()

def extract_features(image_path):
    try:
        img = Image.open(image_path).convert('RGB')
        img_tensor = transform(img).unsqueeze(0).to(device)

        with torch.no_grad():
            features = model(img_tensor).cpu().numpy()[0]
        return features.astype(np.float32)
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
