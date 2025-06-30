from flask import Flask, render_template, request, redirect, url_for, flash
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os
import pickle

app = Flask(__name__)
app.secret_key = 'your_secret_key'

# Load the model
model = load_model('./plant_disease_model_kaggle_one.keras')


with open('class_indices.pkl', 'rb') as f:
    class_indices = pickle.load(f)
class_labels = {v: k for k, v in class_indices.items()}  

UPLOAD_FOLDER = 'static/uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


disease_info = {
    "Pepper__bell___Bacterial_spot": {
        "plant": "Bell Pepper",
        "disease": "Bacterial Spot",
        "description": "Caused by Xanthomonas bacteria, leading to lesions on leaves and fruits.",
        "symptoms": ["Water-soaked dark spots on leaves", "Yellowing of leaves", "Scabby, rough spots on fruit"],
        "solutions": ["Use copper-based sprays", "Avoid overhead watering", "Use disease-free seeds"]
    },
    "Pepper__bell___healthy": {
        "plant": "Bell Pepper",
        "disease": "No Disease",
        "description": "Your plant is healthy! Keep maintaining good care.",
        "symptoms": ["No Symptoms"],
        "solutions": ["Continue proper watering and sunlight"]
    },
    "Potato___Early_blight": {
        "plant": "Potato",
        "disease": "Early Blight",
        "description": "Fungal disease caused by Alternaria solani, affecting older leaves first.",
        "symptoms": ["Brown spots with concentric rings", "Yellowing around lesions", "Premature leaf drop"],
        "solutions": ["Remove infected leaves", "Apply fungicides (Chlorothalonil, Copper)", "Crop rotation"]
    },
    "Potato___Late_blight": {
        "plant": "Potato",
        "disease": "Late Blight",
        "description": "Caused by Phytophthora infestans, spreads rapidly in wet conditions.",
        "symptoms": ["Large dark blotches on leaves", "White mold under leaves", "Stem rot, wilting"],
        "solutions": ["Apply fungicides", "Improve air circulation", "Avoid excessive moisture"]
    },
     "Potato___healthy": {
        "plant": "Potato",
        "disease": "No Disease",
        "description": "Your plant is healthy! Keep maintaining good care.",
        "symptoms": ["No Symptoms"],
        "solutions": ["Continue proper watering and sunlight"]
    },
    "Tomato___healthy": {
        "plant": "Tomato",
        "disease": "Healthy",
        "description": "Your tomato plant is healthy! Keep maintaining good care.",
        "symptoms": ["None"],
        "solutions": ["Continue proper watering and sunlight"]
    },
    "Tomato_Bacterial_spot": {
        "plant": "Tomato",
        "disease": "Bacterial Spot",
        "description": "Xanthomonas bacteria cause small lesions on leaves and fruits.",
        "symptoms": ["Small, dark, water-soaked spots on leaves","Rough, pitted fruit"],
        "solutions": ["Use copper-based fungicides","Avoid overhead watering"]
    },
    "Tomato_Early_blight": {
        "plant": "Tomato",
        "disease": "Early Blight",
        "description": "Alternaria solani fungus attacks older leaves first.",
        "symptoms": ["Brown spots with concentric rings","Yellowing and wilting leaves"],
        "solutions": [" Prune infected leaves"," Apply fungicides"]
    },
    "Tomato_Late_blight": {
        "plant": "Tomato",
        "disease": "Late Blight",
        "description": "Phytophthora infestans spreads rapidly in wet conditions.",
        "symptoms": ["Large brown patches on leaves","White mold on stems and leaves"],
        "solutions": ["Apply fungicides","Increase airflow"]
    },
    "Tomato_Leaf_Mold": {
        "plant": "Tomato",
        "disease": "Leaf Mold",
        "description": "Passalora fulva fungus grows on leaves, reducing photosynthesis.",
        "symptoms": ["Yellowish spots on upper leaves","Fuzzy mold on the undersides"],
        "solutions": ["Improve air circulation","Apply organic fungicides"]
    },
    "Tomato_Septoria_leaf_spot": {
        "plant": "Tomato",
        "disease": "Septoria Leaf Spot",
        "description": " Septoria lycopersici causes brown circular spots on leaves.",
        "symptoms": ["Tiny brown spots with yellow halos","Premature leaf drop"],
        "solutions": ["Prune lower infected leaves","Use fungicides"]
    },
    "Tomato_Spider_mites_Two_spotted_spider_mite": {
        "plant": "Tomato",
        "disease": " Spider Mite Infestation",
        "description": "Tiny spider mites suck plant juices, causing leaf damage.",
        "symptoms": ["Yellow spots and stippling on leaves","Webbing under leaves"],
        "solutions": [" Use neem oil or insecticidal soap","Increase humidity"]
    },
    "Tomato__Target_Spot": {
        "plant": "Tomato",
        "disease": "Target Spot",
        "description": "Caused by Corynespora cassiicola, leads to leaf necrosis.",
        "symptoms": ["Brown circular lesions with darker centers","Leaf wilting"],
        "solutions": ["Use organic fungicides"," Prune lower leaves"]
    },
    "Tomato__Tomato_YellowLeaf__Curl_Virus": {
        "plant": "Tomato",
        "disease": "Tomato Yellow Leaf Curl Virus",
        "description": "A viral disease spread by whiteflies, causing stunted growth.",
        "symptoms": ["Curling and yellowing leaves","Stunted growth"],
        "solutions": ["Control whiteflies with insecticides","Use virus-resistant seeds"]
    },
    "Tomato__Tomato_mosaic_virus": {
        "plant": "Tomato",
        "disease": "Tomato Mosaic Virus",
        "description": "A viral infection affecting leaf and fruit quality.",
        "symptoms": ["Mottled, distorted leaves","Reduced fruit yield"],
        "solutions": ["Remove infected plants","Disinfect gardening tools"]
    }
    
}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        flash('No file part', 'danger')
        return redirect(request.url)

    file = request.files['file']
    if file.filename == '':
        flash('No selected file', 'danger')
        return redirect(request.url)

    if file:
        img_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(img_path)

        
        img = image.load_img(img_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.0  

        # Make prediction
        output = model.predict(img_array)
        predicted_class = np.argmax(output)
        predicted_class_label = class_labels[predicted_class]

       
        disease_details = disease_info.get(predicted_class_label, {
            "plant": "Unknown",
            "disease": "Unknown",
            "description": "No information available.",
            "symptoms": ["Unknown"],
            "solutions": ["No solution available"]
        })

        return render_template(
            'result.html', 
            result=predicted_class_label, 
            image_file=file.filename,
            disease_details=disease_details
        )

if __name__ == '__main__':
    app.run(debug=True)
