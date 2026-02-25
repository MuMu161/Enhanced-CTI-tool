from flask import Flask, render_template, request
from pipeline import EnhanceCTIPipeline
from src.pdf_parser import extract_pdf_text
import sys
import os

sys.path.append(os.path.abspath(os.path.dirname(__file__)))

app = Flask(__name__)
pipeline = EnhanceCTIPipeline()

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route("/", methods=["GET", "POST"])
def index():

    if request.method == "POST":

        files = request.files.getlist("file")   # ✅ allow multiple files
        results = []

        for file in files:
            path = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(path)

            text = extract_pdf_text(path)

            result = pipeline.process(text)   # ✅ new structured return

            results.append({
                "filename": file.filename,
                "decision": result["decision"],
                "industries": result["industries"],
                "classification_scores": result["classification_scores"],
                "similarity_score": result["similarity_score"],
                "duplicate_detected": result["duplicate_detected"],
                "explanations": result["explanations"]
            })

        return render_template(
            "index.html",
            results=results
        )

    return render_template("index.html")

if __name__ == "__main__":
    print("Starting Flask server...")
    app.run(debug=True)
