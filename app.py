import os
import datetime
from flask import Flask, render_template, request, jsonify, send_file
from flask_sqlalchemy import SQLAlchemy
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
UPLOAD_FOLDER = "static/uploads"
REPORT_FOLDER = "static/reports"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(REPORT_FOLDER, exist_ok=True)

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

# ---------------- AI DISABLED (DEMO MODE) ----------------
print("⚠️ AI prediction disabled (Render free-tier demo mode)")

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
    """
    Demo mode: inference intentionally disabled
    """
    return jsonify({
        "status": "error",
        "title": "AI Inference Disabled",
        "message": (
            "Real-time AI prediction is temporarily unavailable on the live demo "
            "due to free-tier server memory limits.\n\n"
            "✔ Full prediction and Grad-CAM work correctly in local or paid environments.\n"
            "✔ This deployment demonstrates UI, workflow, and system design."
        )
    }), 503

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
