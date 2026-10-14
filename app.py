from flask import Flask, render_template, request, jsonify, send_file
from pathlib import Path
from datetime import datetime, timezone
from urllib.parse import urlparse, parse_qs
from werkzeug.utils import secure_filename
import sqlite3, io, secrets, re, os, json

app = Flask(__name__)
ROOT = Path(__file__).resolve().parent
DB = ROOT / "medicines.db"
UPLOADS = ROOT / "uploads"
UPLOADS.mkdir(exist_ok=True)
ALLOWED = {".jpg", ".jpeg", ".png", ".webp"}
MAX_BYTES = 8 * 1024 * 1024
MODEL_XML = ROOT / "models" / "mobilenetv4.xml"
CLASSES_FILE = ROOT / "models" / "medicine_classes.json"
compiled = None
model_labels = []
model_message = "OpenVINO model has not been loaded."


def db_conn():
    c = sqlite3.connect(DB)
    c.row_factory = sqlite3.Row
    return c


def init_db():
    with db_conn() as c:
        c.execute("""CREATE TABLE IF NOT EXISTS medicines(
          medicine_id TEXT PRIMARY KEY, name TEXT NOT NULL, batch_number TEXT NOT NULL,
          manufacturer TEXT NOT NULL, manufacture_date TEXT, expiry_date TEXT NOT NULL,
          image_class TEXT NOT NULL, created_at TEXT NOT NULL)""")
        c.execute("""CREATE TABLE IF NOT EXISTS scans(
          scan_id INTEGER PRIMARY KEY AUTOINCREMENT, medicine_id TEXT NOT NULL,
          scanned_at TEXT NOT NULL, result TEXT NOT NULL, image_label TEXT,
          FOREIGN KEY(medicine_id) REFERENCES medicines(medicine_id))""")


def init_model():
    global compiled, model_labels, model_message
    try:
        if not MODEL_XML.is_file() or not MODEL_XML.with_suffix(".bin").is_file():
            raise FileNotFoundError("MobileNetV4 XML/BIN model files are missing from models/.")
        if not CLASSES_FILE.is_file():
            raise FileNotFoundError("models/medicine_classes.json is missing.")
        import openvino as ov
        with CLASSES_FILE.open("r", encoding="utf-8") as f:
            model_labels = json.load(f)
        core = ov.Core()
        compiled = core.compile_model(str(MODEL_XML), "CPU")
        model_message = f"MobileNetV4 loaded on OpenVINO CPU ({len(model_labels)} classes)."
    except Exception as e:
        compiled = None
        model_message = f"Model unavailable: {e}"


def run_openvino(path):
    """Use the same RGB, 224x224, ImageNet-normalized preprocessing as validation."""
    if compiled is None:
        return {"configured": False, "label": None, "confidence": None, "message": model_message}
    try:
        import numpy as np
        from PIL import Image
        with Image.open(path) as im:
            image = im.convert("RGB").resize((224, 224), Image.Resampling.BILINEAR)
            arr = np.asarray(image, dtype=np.float32) / 255.0
        mean = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)
        arr = (arr - mean) / std
        arr = np.transpose(arr, (2, 0, 1))[None, ...].astype(np.float32)
        result = compiled({compiled.input(0): arr})
        scores = np.asarray(result[compiled.output(0)]).reshape(-1)
        # Convert logits to probabilities for a readable confidence score.
        shifted = scores - np.max(scores)
        probs = np.exp(shifted) / np.sum(np.exp(shifted))
        ix = int(np.argmax(probs))
        label = model_labels[ix] if ix < len(model_labels) else f"class_{ix}"
        return {"configured": True, "label": label,
                "confidence": round(float(probs[ix]), 4),
                "message": ""}
    except Exception as e:
        return {"configured": True, "label": None, "confidence": None,
                "message": f"Inference failed: {e}"}


def new_id():
    return "MED-" + secrets.token_hex(8).upper()


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/api/health")
def health():
    return jsonify({"database": "connected", "openvino_model_loaded": compiled is not None,
                    "openvino_message": model_message})


@app.route("/api/medicines", methods=["GET"])
def list_medicines():
    with db_conn() as c:
        rows = c.execute("SELECT medicine_id,name,batch_number,manufacturer,expiry_date FROM medicines ORDER BY created_at DESC").fetchall()
    return jsonify([dict(r) for r in rows])


@app.route("/api/medicines", methods=["POST"])
def register_medicine():
    data = request.get_json(silent=True) or request.form
    name = str(data.get("name", "")).strip()
    batch = str(data.get("batch_number", "")).strip()
    manufacturer = str(data.get("manufacturer", "")).strip()
    mfg = str(data.get("manufacture_date", "")).strip()
    expiry = str(data.get("expiry_date", "")).strip()
    image_class = str(data.get("image_class", "")).strip() or name
    if not all([name, batch, manufacturer, expiry]):
        return jsonify({"message": "Name, batch number, manufacturer and expiry date are required."}), 400
    if len(name) > 150 or len(batch) > 100 or len(manufacturer) > 150:
        return jsonify({"message": "One or more fields are too long."}), 400
    medicine_id = new_id()
    try:
        with db_conn() as c:
            c.execute("""INSERT INTO medicines
                (medicine_id,name,batch_number,manufacturer,manufacture_date,expiry_date,image_class,created_at)
                VALUES(?,?,?,?,?,?,?,?)""",
                (medicine_id, name, batch, manufacturer, mfg, expiry, image_class, datetime.now(timezone.utc).isoformat()))
    except sqlite3.IntegrityError:
        return jsonify({"message": "ID collision. Please retry."}), 409
    return jsonify({"message": "Medicine registered. Print this QR and attach it to exactly one package.",
                    "medicine": {"medicine_id": medicine_id, "name": name, "batch_number": batch,
                                 "manufacturer": manufacturer, "expiry_date": expiry,
                                 "qr_url": f"/api/qr/{medicine_id}.png"}}), 201


@app.route("/api/qr/<medicine_id>.png")
def qr_image(medicine_id):
    import qrcode
    with db_conn() as c:
        row = c.execute("SELECT medicine_id FROM medicines WHERE medicine_id=?", (medicine_id,)).fetchone()
    if row is None:
        return jsonify({"message": "Medicine ID not found."}), 404
    img = qrcode.make(row["medicine_id"])
    buf = io.BytesIO(); img.save(buf, format="PNG"); buf.seek(0)
    return send_file(buf, mimetype="image/png", download_name=f"{medicine_id}.png")


@app.route("/api/verify", methods=["POST"])
def verify():
    qr_data = (request.form.get("qr_data") or "").strip()
    image = request.files.get("image")
    if not qr_data:
        return jsonify({"message": "Scan or enter a QR ID."}), 400
    if not image or not image.filename:
        return jsonify({"message": "Upload a package image."}), 400
    ext = Path(secure_filename(image.filename)).suffix.lower()
    if ext not in ALLOWED:
        return jsonify({"message": "Allowed: JPG, JPEG, PNG, WEBP."}), 400
    image.stream.seek(0, os.SEEK_END)
    size = image.stream.tell()
    image.stream.seek(0)
    if size == 0 or size > MAX_BYTES:
        return jsonify({"message": "Image must be non-empty and at most 8 MB."}), 400

    parsed = urlparse(qr_data)
    query = parse_qs(parsed.query)
    id_from_url = query.get("id", [None])[0] or query.get("medicine_id", [None])[0]
    medicine_id = (id_from_url or (parsed.fragment if parsed.scheme else qr_data)).strip().upper()
    if not re.fullmatch(r"MED-[A-F0-9]{16}", medicine_id):
        return jsonify({
            "status": "NOT_REGISTERED",
            "message": "QR content is not a valid system medicine ID.",
            "medicine": {}, "image_screening": {}, "scan_count": 0,
            "checks": {"qr_registered": False}
        }), 200

    path = UPLOADS / (secrets.token_hex(12) + ext)
    image.save(path)
    screening = run_openvino(path)

    with db_conn() as c:
        medicine = c.execute(
            "SELECT * FROM medicines WHERE medicine_id=?", (medicine_id,)
        ).fetchone()

        if medicine is None:
            return jsonify({
                "status": "NOT_REGISTERED",
                "message": "This identifier is not registered in the local demo database.",
                "medicine": {}, "image_screening": screening, "scan_count": 0,
                "checks": {"qr_registered": False}
            }), 200

        prior_count = c.execute(
            "SELECT COUNT(*) FROM scans WHERE medicine_id=?", (medicine_id,)
        ).fetchone()[0]
        count = prior_count + 1
        details = {k: medicine[k] for k in (
            "name", "batch_number", "manufacturer", "expiry_date", "image_class"
        )}

        expected = (medicine["image_class"] or medicine["name"] or "").strip()
        predicted = (screening.get("label") or "").strip()
        class_match = bool(predicted and expected and predicted.casefold() == expected.casefold())
        today = datetime.now().date()
        expiry_date = None
        try:
            expiry_date = datetime.fromisoformat((medicine["expiry_date"] or "").strip()[:10]).date()
        except (ValueError, TypeError):
            pass
        expired = expiry_date is not None and expiry_date < today
        expiry_unknown = expiry_date is None
        confidence = screening.get("confidence")
        model_ready = bool(screening.get("configured") and predicted)

        checks = {
            "qr_registered": True,
            "package_class_match": class_match if model_ready else None,
            "expiry_valid": (not expired) if not expiry_unknown else None,
            "first_scan": prior_count == 0,
            "model_available": model_ready,
            "model_confidence": confidence
        }

        if prior_count > 0:
            status = "SUSPICIOUS"
            message_text = "This identifier has been scanned before."
        elif expired:
            status = "EXPIRED"
            message_text = "The registered expiry date has passed. Do not treat this package as verified."
        elif not model_ready:
            status = "UNVERIFIED"
            message_text = "The QR ID is registered, but the AI package check could not be completed."
        elif not class_match:
            status = "MISMATCH"
            message_text = f"The registered expected class ('{expected}') does not match the model prediction ('{predicted}'). Review the image and record."
        elif confidence is not None and confidence < 0.35:
            status = "UNVERIFIED"
            message_text = "The package class matched, but the model score is too low for this prototype's verification rule."
        else:
            status = "VERIFIED_DEMO"
            message_text = "The QR is registered, the predicted package class matches the expected class, and the record is not expired." \
            ""

        c.execute(
            "INSERT INTO scans(medicine_id,scanned_at,result,image_label) VALUES(?,?,?,?)",
            (medicine_id, datetime.now(timezone.utc).isoformat(), status, screening.get("label"))
        )

    notes = [
        screening.get("message") or "",
        ""
    ]
    if expiry_unknown:
        notes.append("Expiry date could not be parsed; expiry was not confirmed.")
    if confidence is not None and confidence < 0.35:
        notes.append("Low model score; treat the prediction cautiously.")

    return jsonify({
        "status": status,
        "message": message_text,
        "medicine": details,
        "image_screening": screening,
        "scan_count": count,
        "checks": checks,
        "note": " ".join(x for x in notes if x)
    })


if __name__ == "__main__":
    init_db()
    init_model()
    print(f"Open http://127.0.0.1:5000 | {model_message}")
    app.run(host="127.0.0.1", port=5000, debug=True)
