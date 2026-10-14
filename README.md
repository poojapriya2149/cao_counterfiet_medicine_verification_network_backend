# MedVerify — Integrated Flask + QR + OpenVINO

This project combines the MedVerify web interface, Flask/SQLite QR registration and scan-history backend, and the trained MobileNetV4 OpenVINO model. It does **not** retrain the model.

## Run on Windows (PowerShell)

Open this folder in VS Code, then run each command separately:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe app.py
```

Open http://127.0.0.1:5000. Model status: http://127.0.0.1:5000/api/health

## Included
- `app.py`: Flask routes, QR generation, SQLite records and scan history, OpenVINO inference.
- `templates/index.html`, `static/app.js`, `static/style.css`: web UI.
- `models/mobilenetv4.xml`, `models/mobilenetv4.bin`, `models/medicine_classes.json`: trained model and labels.
- `medicines.db`: supplied SQLite database (currently empty of medicine records).

## Verification statuses
- `VERIFIED_DEMO`: the QR ID exists, the model prediction exactly matches the registered expected image class, the record is not expired, and this is the first recorded scan. The UI explicitly says this is verified against demo records only.
- `MISMATCH`: the model's predicted class differs from the expected class.
- `EXPIRED`: the registered expiry date has passed.
- `SUSPICIOUS`: the ID has been scanned previously; this is a review signal, not proof of counterfeiting.
- `UNVERIFIED`: the model check is unavailable or the prediction score is below the prototype threshold.
- `NOT_REGISTERED`: the QR identifier is invalid or absent from the local database.

A `VERIFIED_DEMO` status is a prototype rule, not real-world proof of authenticity. Use trusted manufacturer/supply-chain records and secure, uniquely issued package identifiers before making real authenticity claims.

## Important interpretation
MobileNetV4 predicts one of the 50 medicine/package classes. It does **not** classify a product as genuine or counterfeit. QR verification checks whether the ID exists in this local database. A repeated scan is flagged for review, not automatically labelled counterfeit. The backend also checks expected class matching and expiry. A registered QR and an AI class prediction are supporting signals, not regulatory proof of authenticity.

The preprocessing follows the training/evaluation pipeline: RGB, 224×224, tensor scaling to [0,1], then ImageNet mean/std normalization. Confidence is computed with softmax over model outputs; treat it as a model score, not a calibrated probability.

## Notes
- The browser QR scanner loads `html5-qrcode` from a CDN, so camera scanning requires internet access; manual ID entry remains available.
- This is a local academic prototype. Before deployment, add authenticated admin access, CSRF protection, rate limiting, upload cleanup/retention rules, stronger file validation, and security testing.
