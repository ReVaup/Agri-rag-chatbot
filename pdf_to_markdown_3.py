import os
import gc
from docling.document_converter import DocumentConverter

PDF_DIR = "D:/Deltacubes/python_programming/pdfs"
OUTPUT_DIR = "D:/Deltacubes/python_programming/markdowns"

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("🚀 Starting Docling Conversion...")

# ✅ Create ONE converter (important)
converter = DocumentConverter()

files = [f for f in os.listdir(PDF_DIR) if f.endswith(".pdf")]

# 🔥 Optional: process small → large (reduces crashes)
files = sorted(files, key=lambda x: os.path.getsize(os.path.join(PDF_DIR, x)))

for file in files:
    try:
        pdf_path = os.path.join(PDF_DIR, file)
        md_path = os.path.join(OUTPUT_DIR, file.replace(".pdf", ".md"))

        # ✅ Skip already processed
        if os.path.exists(md_path):
            continue

        print(f"📄 Processing: {file}")

        # 🔥 Convert
        result = converter.convert(pdf_path)

        markdown = result.document.export_to_markdown()

        # ✅ Save
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(markdown)

        print(f"✅ Saved: {file}")

        # 🔥 CRITICAL: free memory
        del result
        del markdown
        gc.collect()

    except Exception as e:
        print(f"❌ Failed: {file} → {e}")
        continue

print("🎉 All Done!")