import mysql.connector
import requests
from PIL import Image
import io
import numpy as np
import tensorflow as tf
import os

# Load AI model
model = tf.keras.applications.MobileNetV2(weights='imagenet', include_top=False, pooling='avg')

def extract_features_from_url(url):
    try:
        response = requests.get(url, timeout=5)
        img = Image.open(io.BytesIO(response.content)).resize((224, 224)).convert('RGB')
        x = tf.keras.preprocessing.image.img_to_array(img)
        x = tf.keras.applications.mobilenet_v2.preprocess_input(np.expand_dims(x, axis=0))
        features = model.predict(x)
        return features[0]
    except Exception as e:
        print(f"Failed to process {url}: {e}")
        return None

# Connect to MySQL database
conn = mysql.connector.connect(
    host=os.environ.get('DB_HOST'),
    user=os.environ.get('DB_USER'),
    password=os.environ.get('DB_PASS'),
    database=os.environ.get('DB_NAME')
)
cursor = conn.cursor()
cursor.execute("SELECT product_id, product_image, product_image2, product_image3 FROM product_items")
rows = cursor.fetchall()

base_url = "https://dabramart.in/admin_panel/"
product_ids = []
features = []

for row in rows:
    pid = row[0]
    for image_field in row[1:]:
        if image_field:
            full_url = base_url + image_field
            feat = extract_features_from_url(full_url)
            if feat is not None:
                product_ids.append(pid)
                features.append(feat)

np.save("product_ids.npy", np.array(product_ids))
np.save("features_db.npy", np.array(features))
print("Feature extraction complete.")
