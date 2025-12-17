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

# ---------------- CLASS NAMES (MUST MATCH TRAINING) ----------------
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

# ---------------- GRAD-CAM (GLOBAL, RAM SAFE) ----------------
try:
    last_conv_layer_name = next(
        layer.name for layer in reversed(model.layers)
        if isinstance(layer, layers.Conv2D)
    )

    grad_model = tf.keras.models.Model(
        [model.inputs],
        [model.get_layer(last_conv_layer_name).output, model.output]
    )
    print("✅ Grad-CAM Initialized")

except Exception as e:
    print("⚠️ Grad-CAM Disabled:", e)
    grad_model = None

# ---------------- HELPER FUNCTIONS ----------------
def generate_gradcam(img_array, class_index):
    if grad_model is None:
        return np.zeros((224, 224))

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = predictions[:, class_index]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]

    heatmap = tf.reduce_sum(conv_outputs * pooled_grads, axis=-1)
    heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + 1e-8)

    return heatmap.numpy()

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

        # -------- PREPROCESS (NO MANUAL preprocess_input HERE) --------
        img = image.load_img(filepath, target_size=(224, 224))
        img_array = np.expand_dims(image.img_to_array(img), axis=0)

        # -------- PREDICTION --------
        preds = model.predict(img_array, verbose=0)[0]
        class_index = int(np.argmax(preds))
        confidence = float(preds[class_index])
        prediction = CLASS_NAMES[class_index]

        # -------- GRAD-CAM --------
        heatmap = generate_gradcam(img_array, class_index)
        original = cv2.imread(filepath)

        heatmap = cv2.resize(heatmap, (original.shape[1], original.shape[0]))
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

        overlay = cv2.addWeighted(original, 0.6, heatmap, 0.4, 0)
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
        del img_array, preds, heatmap, overlay
        gc.collect()
        tf.keras.backend.clear_session()

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

    img_path = os.path.join(UPLOAD_FOLDER, f"gradcam_{record.image_path}")
    if os.path.exists(img_path):
        pdf.image(img_path, x=10, y=120, w=100)

    report_path = os.path.join(REPORT_FOLDER, f"report_{record_id}.pdf")
    pdf.output(report_path)

    return send_file(report_path, as_attachment=True)

# ---------------- ENTRY POINT ----------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
