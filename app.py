from flask import Flask, render_template, request
from keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import os
import warnings
import math
# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings("ignore")

app = Flask(__name__)

# Folder to save uploaded images
UPLOAD_FOLDER = 'static/uploads/'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load models (including EfficientNetB0)
models = {
    "DenseNet201": load_model("models/DenseNet201_banana_classification_model80.h5", compile=False),
    "MobileNetV2": load_model("models/MobileNet_banana_classification_model80.h5", compile=False),
    "ResNet50": load_model("models/ResNet50_banana_classification_model80.h5", compile=False),
    "InceptionV3": load_model("models/InceptionV3_banana_classification_model80.h5", compile=False),
    "EfficientNetB0": load_model("models/EfficientNet_banana_classification_model80.h5", compile=False)
}

# Class labels
classes = ['Panama Disease','cordana','healthy','pestalotiopsis','sigatoka','xamthomonas']

# Image preprocessing
def prepare_image(image_path, model_name):
    target_size = (250, 250)
    img = load_img(image_path, target_size=target_size)
    img = img_to_array(img)

    if model_name == "DenseNet201":
        from tensorflow.keras.applications.densenet import preprocess_input
    elif model_name == "MobileNetV2":
        from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
    elif model_name == "ResNet50":
        from tensorflow.keras.applications.resnet50 import preprocess_input
    elif model_name == "InceptionV3":
        from tensorflow.keras.applications.inception_v3 import preprocess_input
    elif model_name == "EfficientNetB0":
        from tensorflow.keras.applications.efficientnet import preprocess_input
    else:
        preprocess_input = lambda x: x / 255.0

    img = preprocess_input(img)
    img = np.expand_dims(img, axis=0)
    return img



@app.route('/', methods=['GET', 'POST'])
def index():
    predictions = {}
    image_url = None

    if request.method == 'POST':
        image = request.files.get('image')
        if image:
            filename = image.filename
            image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            image.save(image_path)
            image_url = image_path

            for model_name, model in models.items():
                processed_img = prepare_image(image_path, model_name)
                pred = model.predict(processed_img, verbose=0)
                confidence = float(np.max(pred))
                if confidence < 0.40:
                    predictions[model_name] = {
                        "label": "Invalid Image",
                        "confidence": round(confidence * 100, 2)
                    }
                else:
                    predicted_label = classes[np.argmax(pred)]
                    predictions[model_name] = {
                    "label": predicted_label,
                    "confidence": round(confidence * 100, 2)
                }

    return render_template("index.html", predictions=predictions, image_url=image_url)

if __name__ == '__main__':
    app.run(debug=True)
