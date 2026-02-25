try:
    import PyPDF2
except ImportError:
    print("⚠ PyPDF2 not installed")


def extract_pdf_text(path):
    text = ""

    try:
        with open(path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                text += page.extract_text() or ""
    except Exception as e:
        print("PDF extraction error:", e)

    return text
