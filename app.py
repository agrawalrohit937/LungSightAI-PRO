import os
import gc
import cv2
import numpy as np
import datetime
import tensorflow as tf

from flask import Flask, render_template, request, jsonify, send_file
from flask_sqlalchemy import SQLAlchemy
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras import layers
from tensorflow.keras.applications.resnet50 import preprocess_input
from fpdf import FPDF

# ---------------- SYSTEM SAFETY ----------------
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# ---------------- FLASK CONFIG ----------------
app = Flask(__name__)
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///database.db"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
db = SQLAlchemy(app)

# ---------------- PATHS ----------------
MODEL_PATH = "models/ResNet50_TL_best.keras"
UPLOAD_FOLDER = "static/uploads"
REPORT_FOLDER = "static/reports"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(REPORT_FOLDER, exist_ok=True)

# ---------------- CLASS NAMES ----------------
CLASS_NAMES = ["COVID19", "NORMAL", "PNEUMONIA"]

# ---------------- DATABASE MODEL ----------------
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

# ---------------- LOAD MODEL (LAMBDA SAFE) ----------------
print("⚡ Loading Neural Network System...")

model = load_model(
    MODEL_PATH,
    custom_objects={"preprocess_input": preprocess_input},
    compile=False
)

print("✅ Model Loaded Successfully")

# ------------------------------------------------------------------
# 🚫 GRAD-CAM DISABLED (VERY IMPORTANT FOR RENDER FREE TIER)
# ------------------------------------------------------------------
# Render Free (512MB) me ResNet50 + GradCAM = OOM
# Isliye grad_model ko forcefully None rakh rahe hain

grad_model = None
print("⚠️ Grad-CAM TEMPORARILY DISABLED FOR FREE TIER")

# ---------------- HELPER FUNCTIONS ----------------
def generate_gradcam(img_array, class_index):
    """
    Free tier ke liye blank heatmap return karega
    """
    return np.zeros((224, 224))

# ---------------- ROUTES ----------------
@app.route("/")
def landing():
    return render_template("landing.html")

@app.route("/dashboard")
def dashboard():
    history = ScanRecord.query.order_by(ScanRecord.date.desc()).limit(5).all()
    return render_template("dashboard.html", history=history)

@app.route("/analyze", methods=["POST"])
def analyze():
    try:
        file = request.files["file"]
        name = request.form["name"]
        age = int(request.form["age"])

        filename = f"{datetime.datetime.now().timestamp()}_{file.filename}"
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)

        # -------- PREPROCESS (MODEL HANDLE KARTA HAI) --------
        img = image.load_img(filepath, target_size=(224, 224))
        img_array = np.expand_dims(image.img_to_array(img), axis=0)

        # -------- PREDICTION --------
        preds = model.predict(img_array, verbose=0)[0]
        class_index = int(np.argmax(preds))
        confidence = float(preds[class_index])
        prediction = CLASS_NAMES[class_index]

        # -------- DUMMY GRAD-CAM OVERLAY (BLANK) --------
        original = cv2.imread(filepath)
        blank_heatmap = np.zeros_like(original)
        overlay = cv2.addWeighted(original, 1.0, blank_heatmap, 0.0, 0)

        overlay_name = f"gradcam_{filename}"
        overlay_path = os.path.join(UPLOAD_FOLDER, overlay_name)
        cv2.imwrite(overlay_path, overlay)

        # -------- SAVE TO DB --------
        record = ScanRecord(
            patient_name=name,
            age=age,
            prediction=prediction,
            confidence=round(confidence * 100, 2),
            image_path=filename
        )
        db.session.add(record)
        db.session.commit()

        # -------- MEMORY CLEANUP --------
        del img_array, preds, overlay
        gc.collect()

        return jsonify({
            "status": "success",
            "prediction": prediction,
            "confidence": round(confidence * 100, 2),
            "original_url": filepath,
            "overlay_url": overlay_path,
            "record_id": record.id
        })

    except Exception as e:
        print("❌ Error:", e)
        return jsonify({"status": "error", "message": str(e)})

@app.route("/download_report/<int:record_id>")
def download_report(record_id):
    record = ScanRecord.query.get_or_404(record_id)

    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.cell(200, 10, "LungSight AI - Diagnostic Report", ln=True, align="C")

    pdf.set_font("Arial", size=12)
    pdf.ln(10)
    pdf.cell(200, 10, f"Patient Name: {record.patient_name}", ln=True)
    pdf.cell(200, 10, f"Age: {record.age}", ln=True)
    pdf.cell(200, 10, f"Date: {record.date.strftime('%Y-%m-%d %H:%M')}", ln=True)

    pdf.ln(10)
    pdf.set_font("Arial", "B", 14)
    pdf.cell(200, 10, f"Diagnosis: {record.prediction}", ln=True)
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, f"Confidence: {record.confidence}%", ln=True)

    report_path = os.path.join(REPORT_FOLDER, f"report_{record_id}.pdf")
    pdf.output(report_path)

    return send_file(report_path, as_attachment=True)

# ---------------- ENTRY POINT ----------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
