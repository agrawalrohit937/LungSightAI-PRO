import os
import cv2
import numpy as np
import datetime
from flask import Flask, render_template, request, jsonify, send_file
from flask_sqlalchemy import SQLAlchemy
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras import layers
import tensorflow as tf
from fpdf import FPDF
# --- 1. ADD THIS IMPORT ---
from tensorflow.keras.applications.resnet50 import preprocess_input

# --- CONFIGURATION ---
app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///database.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

MODEL_PATH = 'models/ResNet50_TL_final.keras'
UPLOAD_FOLDER = 'static/uploads'
REPORT_FOLDER = 'static/reports'
CLASS_NAMES = ['Covid', 'Normal', 'Viral Pneumonia']

for folder in [UPLOAD_FOLDER, REPORT_FOLDER]:
    os.makedirs(folder, exist_ok=True)

# --- DATABASE MODEL ---
class ScanRecord(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    patient_name = db.Column(db.String(100), nullable=False)
    age = db.Column(db.Integer, nullable=False)
    prediction = db.Column(db.String(50), nullable=False)
    confidence = db.Column(db.Float, nullable=False)
    date = db.Column(db.DateTime, default=datetime.datetime.now)
    image_path = db.Column(db.String(200))

with app.app_context():
    db.create_all()

# --- AI ENGINE ---
print("⚡ Loading Neural Network System...")

# --- 2. UPDATED LOADING LOGIC ---
# We pass 'custom_objects' so Keras knows what 'preprocess_input' is inside the model
model = load_model(MODEL_PATH, custom_objects={"preprocess_input": preprocess_input})

def get_gradcam(img_array, model, class_idx):
    # Auto-detect last conv layer
    last_conv_layer = next(x.name for x in reversed(model.layers) if isinstance(x, layers.Conv2D))
    grad_model = tf.keras.models.Model([model.inputs], [model.get_layer(last_conv_layer).output, model.output])
    
    with tf.GradientTape() as tape:
        conv_outputs, preds = grad_model(img_array)
        loss = preds[:, class_idx]
    
    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-10)
    return heatmap.numpy()

# --- ROUTES ---

@app.route('/')
def landing():
    return render_template('landing.html')

@app.route('/dashboard')
def dashboard():
    # Fetch last 5 records for the history table
    history = ScanRecord.query.order_by(ScanRecord.date.desc()).limit(5).all()
    return render_template('dashboard.html', history=history)

@app.route('/analyze', methods=['POST'])
def analyze():
    file = request.files['file']
    name = request.form['name']
    age = request.form['age']
    
    filename = f"{datetime.datetime.now().timestamp()}_{file.filename}"
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)

    # Preprocess & Predict
    img = image.load_img(filepath, target_size=(224, 224))
    img_array = np.expand_dims(image.img_to_array(img), axis=0)
    
    # NOTE: ResNet preprocessing is handled inside the model via the Lambda layer
    # so we just pass the raw image array (0-255)
    
    preds = model.predict(img_array)
    top_idx = np.argmax(preds[0])
    confidence = float(preds[0][top_idx])
    prediction = CLASS_NAMES[top_idx]

    # GradCAM
    heatmap = get_gradcam(img_array, model, top_idx)
    original_cv = cv2.imread(filepath)
    heatmap_resized = cv2.resize(heatmap, (original_cv.shape[1], original_cv.shape[0]))
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    heatmap_colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(original_cv, 0.6, heatmap_colored, 0.4, 0)
    
    overlay_filename = f"gradcam_{filename}"
    overlay_path = os.path.join(UPLOAD_FOLDER, overlay_filename)
    cv2.imwrite(overlay_path, overlay)

    # Save to DB
    new_record = ScanRecord(patient_name=name, age=age, prediction=prediction, 
                            confidence=round(confidence*100, 2), image_path=filename)
    db.session.add(new_record)
    db.session.commit()

    return jsonify({
        'status': 'success',
        'prediction': prediction,
        'confidence': round(confidence*100, 2),
        'original_url': filepath,
        'overlay_url': overlay_path,
        'record_id': new_record.id,
        'probs': [float(p) for p in preds[0]]
    })

@app.route('/download_report/<int:record_id>')
def download_report(record_id):
    record = ScanRecord.query.get_or_404(record_id)
    
    # Generate PDF
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(200, 10, txt="LungSight AI - Diagnostic Report", ln=True, align='C')
    
    pdf.set_font("Arial", size=12)
    pdf.ln(10)
    pdf.cell(200, 10, txt=f"Patient Name: {record.patient_name}", ln=True)
    pdf.cell(200, 10, txt=f"Patient Age: {record.age}", ln=True)
    pdf.cell(200, 10, txt=f"Date: {record.date.strftime('%Y-%m-%d %H:%M')}", ln=True)
    pdf.ln(10)
    
    pdf.set_font("Arial", 'B', 14)
    if record.prediction == 'Normal':
        pdf.set_text_color(0, 128, 0) # Green
    else:
        pdf.set_text_color(255, 0, 0) # Red
        
    pdf.cell(200, 10, txt=f"AI Diagnosis: {record.prediction}", ln=True)
    pdf.set_text_color(0, 0, 0)
    pdf.cell(200, 10, txt=f"Confidence Score: {record.confidence}%", ln=True)
    
    # Add Image
    img_path = os.path.join(UPLOAD_FOLDER, f"gradcam_{record.image_path}")
    if os.path.exists(img_path):
        pdf.image(img_path, x=10, y=100, w=100)
    
    report_path = os.path.join(REPORT_FOLDER, f"report_{record_id}.pdf")
    pdf.output(report_path)
    
    return send_file(report_path, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True, port=5000)