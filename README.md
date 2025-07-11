#  Plant Disease Classifier 

A deep learning-powered web app that detects diseases in **tomato**, **potato**, and **bell pepper** leaves using image classification with CNN. Built with TensorFlow and Flask, this project helps farmers and researchers identify common crop diseases from uploaded leaf images.

---

##  Features

- Upload leaf images of tomato, potato, or bell pepper
- Classifies multiple diseases and healthy conditions
- Shows disease name, symptoms, and treatment suggestions
- Built as a simple Flask web application

---

##  Supported Crops and Diseases

| Plant         | Diseases Detected                             |
|--------------|-----------------------------------------------|
| Tomato       | Bacterial Spot, Early Blight, Late Blight, Healthy |
| Potato       | Early Blight, Late Blight, Healthy            |
| Bell Pepper  | Bacterial Spot, Healthy                       |

---

##  Technologies Used

- Python
- TensorFlow / Keras
- Flask (Web framework)
- HTML / CSS
- Pillow (for image handling)
- Pickle (for class label mapping)

---

##  Project Structure
```

plant-disease-classifier/
│
├── app.py # Flask application
├── plant_disease_model.keras # Trained Keras model
├── class_indices.pkl # Class label mappings
├── static/ # Static files (images, CSS)
├── templates/ # HTML templates
├── requirements.txt # Project dependencies
└── README.md # Project documentation

```
---

##  How to Run Locally

> Make sure Python is installed and your environment is activated.

1. **Clone this repository**

```bash
git clone https://github.com/UddhavYadav/plant-disease-classifier.git
cd plant-disease-classifier
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Run the app**
```bash
python app.py
```

4. **Visit in browser**
```
http://127.0.0.1:5000/

```