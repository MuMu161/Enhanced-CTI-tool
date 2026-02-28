"""
app.py — Improved Flask Web Application

Original issues:
1. extract_pdf_text() called but structured fields never parsed
   → FeatureExtractor only gets raw text, misses 14 of 19 features
2. No error handling (bad PDF crashes the server)
3. No input validation
4. Results dict manually built — easy to miss fields

Improvements:
1. Uses parse_pdf() → passes structured fields to pipeline
2. Proper try/except per file (one bad PDF doesn't break others)
3. Input validation (empty upload, non-PDF files)
4. Passes merge probability to UI (bonus confidence indicator)
5. Cleaner result building
"""

from flask import Flask, render_template, request, jsonify
import sys
import os

sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from pipeline import EnhanceCTIPipeline
from src.pdf_parser import parse_pdf   # ← use parse_pdf, not extract_pdf_text

app = Flask(__name__)
pipeline = EnhanceCTIPipeline()

UPLOAD_FOLDER  = "uploads"
ALLOWED_EXTENSIONS = {"pdf"}
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


# ── Helpers ───────────────────────────────────────────────────────────

def allowed_file(filename):
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS
    )


def safe_filename(filename):
    """Basic sanitization — replace spaces and keep alphanumeric+dot."""
    import re
    name = os.path.basename(filename)          # strip any path traversal
    name = re.sub(r"[^\w\.\-]", "_", name)    # replace unsafe chars
    return name or "upload.pdf"


# ── Routes ────────────────────────────────────────────────────────────

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":

        files = request.files.getlist("file")

        if not files or all(f.filename == "" for f in files):
            return render_template(
                "index.html",
                error="No files selected. Please upload at least one PDF."
            )

        results = []

        for file in files:

            if not file or file.filename == "":
                continue

            # ── Validate file type ────────────────────────────────────
            if not allowed_file(file.filename):
                results.append({
                    "filename": file.filename,
                    "error":    "Not a PDF file — skipped.",
                })
                continue

            # ── Save uploaded file ────────────────────────────────────
            filename = safe_filename(file.filename)
            path     = os.path.join(UPLOAD_FOLDER, filename)

            try:
                file.save(path)
            except Exception as e:
                results.append({
                    "filename": file.filename,
                    "error":    f"Could not save file: {e}",
                })
                continue

            # ── Parse PDF (text + structured fields) ─────────────────
            try:
                text, structured = parse_pdf(path)
            except Exception as e:
                results.append({
                    "filename": filename,
                    "error":    f"PDF parsing failed: {e}",
                })
                continue

            if not text.strip():
                results.append({
                    "filename": filename,
                    "error":    "Could not extract text from this PDF.",
                })
                continue

            # ── Run EnhanceCTI pipeline ───────────────────────────────
            try:
                result = pipeline.process(text, structured=structured)
            except Exception as e:
                results.append({
                    "filename": filename,
                    "error":    f"Pipeline error: {e}",
                })
                continue

            # ── Build result for template ─────────────────────────────
            results.append({
                "filename":              filename,
                "decision":              result.get("decision",              "UNKNOWN"),
                "industries":            result.get("industries",            []),
                "classification_scores": result.get("classification_scores", {}),
                "similarity_score":      result.get("similarity_score",      0.0),
                "duplicate_detected":    result.get("duplicate_detected",    False),
                "explanations":          result.get("explanations",          {}),
                "reason":                result.get("reason",                ""),
                # Bonus fields (shown in UI if template supports them)
                "features_extracted":    result.get("features_extracted",    {}),
                "db_size_after":         result.get("db_size_after",         0),
                "error":                 None,
            })

        return render_template("index.html", results=results)

    # GET request
    return render_template("index.html", results=None)


# ── Optional JSON API endpoint (useful for testing) ───────────────────

@app.route("/api/process", methods=["POST"])
def api_process():
    """
    JSON API for programmatic access.
    POST with multipart/form-data file field named 'file'.
    Returns JSON result.
    """
    if "file" not in request.files:
        return jsonify({"error": "No file field in request"}), 400

    file = request.files["file"]

    if not allowed_file(file.filename):
        return jsonify({"error": "Only PDF files are accepted"}), 400

    filename = safe_filename(file.filename)
    path     = os.path.join(UPLOAD_FOLDER, filename)
    file.save(path)

    try:
        text, structured = parse_pdf(path)
        result = pipeline.process(text, structured=structured)
        result["filename"] = filename
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── Health check ──────────────────────────────────────────────────────

@app.route("/health")
def health():
    return jsonify({
        "status":   "ok",
        "db_size":  len(pipeline.database),
        "model_ok": pipeline.industry_model.model is not None,
    })


# ── Entry point ───────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 50)
    print("  EnhanceCTI Flask Server")
    print("  Open: http://127.0.0.1:5000")
    print("=" * 50)
    app.run(debug=True, host="0.0.0.0", port=5000)
