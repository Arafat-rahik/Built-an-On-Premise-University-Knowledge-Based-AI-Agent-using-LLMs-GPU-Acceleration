import fitz
from pathlib import Path
import easyocr
import numpy as np
import cv2
import re
import torch
from tqdm import tqdm

input_folder = Path("data")
output_folder = Path("text_data")
output_folder.mkdir(exist_ok=True, parents=True)

gpu_available = torch.cuda.is_available()
ocr_reader = easyocr.Reader(['bn','en'], gpu=gpu_available)

def clean_text(text):
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'(\d+)\)', r'\n\n\1)', text)
    text = re.sub(r'\n{2,}', '\n\n', text)
    return text.strip()

for pdf in tqdm(list(input_folder.glob("*.pdf")), desc="PDFs"):
    doc = fitz.open(pdf)
    full_text = ""

    for page_number, page in enumerate(doc, start=1):
        text = page.get_text("text").strip()

        if len(text) < 50:  # fallback OCR
            pix = page.get_pixmap(dpi=300)
            img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)

            if pix.n == 4:
                img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
            elif pix.n == 1:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

            try:
                ocr_result = ocr_reader.readtext(img, detail=0, paragraph=True)
                text = " ".join(ocr_result)
                print(f"OCR applied: {pdf.name}, page {page_number}")
            except Exception as e:
                print(f"Error OCRing {pdf.name}, page {page_number}: {e}")

        full_text += text + "\n\n"

    full_text = clean_text(full_text)
    output_file = output_folder / f"{pdf.stem}.txt"
    output_file.write_text(full_text, encoding="utf-8")
    print(f"Converted: {pdf.name} ✅")
