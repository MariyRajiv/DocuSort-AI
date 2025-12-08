import os
import io
import uuid
import logging
import re
import difflib
import requests
from datetime import datetime, timezone
from pathlib import Path
from typing import List

import numpy as np
import cv2
from PIL import Image
import PyPDF2
from docx import Document as DocxDocument
from pptx import Presentation
from pptx.enum.shapes import MSO_SHAPE_TYPE
from pdf2image import convert_from_bytes

from fastapi import FastAPI, File, UploadFile, APIRouter
from starlette.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from motor.motor_asyncio import AsyncIOMotorClient
import easyocr

# ----------------------------
# ENV + Setup
# ----------------------------
ROOT_DIR = Path(__file__).parent

if (ROOT_DIR / ".env").exists():
    from dotenv import load_dotenv
    load_dotenv(ROOT_DIR / ".env")

client = AsyncIOMotorClient(os.getenv("MONGO_URL"))
db = client[os.getenv("DB_NAME", "documents")]

app = FastAPI()
api = APIRouter(prefix="/api")

HF_API_KEY = os.getenv("HF_API_KEY")
HF_EMBED_URL = "https://api-inference.huggingface.co/pipeline/feature-extraction/sentence-transformers/all-MiniLM-L6-v2"

HEADERS = {"Authorization": f"Bearer {HF_API_KEY}"}

# EasyOCR
reader = easyocr.Reader(['en'], gpu=False)

logging.basicConfig(level=logging.INFO)


# ----------------------------
# Document Keywords
# ----------------------------
DOCUMENT_PATTERNS = {
    "Resume": {"keywords": ["experience", "education", "skills", "projects", "work history",
                             "employment", "certifications", "curriculum vitae",
                             "b.tech", "internship", "summary", "achievements"]},

    "Invoice": {"keywords": ["invoice", "invoice no", "bill to", "amount due", "total",
                             "subtotal", "payment", "gst", "vat", "price", "quantity",
                             "description", "receipt"]},

    "Contract": {"keywords": ["contract", "agreement", "party a", "party b", "terms",
                              "conditions", "confidentiality", "governing law",
                              "witness", "signatures", "obligations", "liability"]},

    "Report": {"keywords": ["report", "site report", "project details", "project report",
                            "summary", "description", "details", "findings", "assessment",
                            "audit", "inspections", "issues", "priority", "status",
                            "comments", "assigned to", "due date", "renovation", "construction",
                            "work log", "progress", "project reference", "in progress"]}
}


# ----------------------------
# Models
# ----------------------------
class ClassificationResult(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    file_name: str
    file_type: str
    file_size: int
    document_type: str
    summary: str
    reason: str
    confidence: float
    extracted_text: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


# ----------------------------
# OCR Utils
# ----------------------------
def preprocess_image(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.convertScaleAbs(gray, alpha=1.4, beta=10)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY)
    return thresh


def extract_text_from_image(raw):
    try:
        image = Image.open(io.BytesIO(raw)).convert("RGB")
        arr = np.array(image)
        processed = preprocess_image(arr)
        text = reader.readtext(processed, detail=0)

        if not text:
            text = reader.readtext(arr, detail=0)

        return " ".join(str(t) for t in text).strip()
    except Exception:
        return ""


def extract_text_from_pdf(raw):
    try:
        reader_pdf = PyPDF2.PdfReader(io.BytesIO(raw))
        text = ""
        for p in reader_pdf.pages:
            text += p.extract_text() or ""
        return text.strip()
    except Exception:
        return ""


def extract_text_from_pdf_with_ocr(raw):
    try:
        text = extract_text_from_pdf(raw)
        if text.strip():
            return text

        pages = convert_from_bytes(raw)
        full = ""
        for page in pages:
            arr = np.array(page)
            processed = preprocess_image(arr)
            result = reader.readtext(processed, detail=0)
            full += " ".join(str(t) for t in result) + "\n"

        return full.strip()
    except Exception:
        return ""


def extract_text_from_docx(raw):
    try:
        doc = DocxDocument(io.BytesIO(raw))
        return "\n".join(p.text for p in doc.paragraphs)
    except Exception:
        return ""


def extract_text_from_pptx(raw):
    try:
        prs = Presentation(io.BytesIO(raw))
        texts = []
        for slide in prs.slides:
            for shape in slide.shapes:
                extract_text_from_shape(shape, texts)
        return "\n".join(t.strip() for t in texts if t.strip())
    except Exception:
        return ""


def extract_text_from_shape(shape, texts):
    if getattr(shape, "has_text_frame", False):
        texts.append(shape.text_frame.text)
    if shape.shape_type == MSO_SHAPE_TYPE.GROUP:
        for subshape in shape.shapes:
            extract_text_from_shape(subshape, texts)


# ----------------------------
# Email Detection
# ----------------------------
def is_email(text):
    t = text.lower()

    headers = ["from:", "to:", "subject:", "cc:", "bcc:", "date:"]
    header_count = sum(1 for h in headers if h in t)

    emails = re.findall(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", t)
    email_count = len(emails)

    greetings = ["\nhi ", "\nhello ", "\ndear "]
    signoffs = ["regards", "sincerely", "best regards", "thanks"]

    score = (
        min(header_count, 3) * 1.0
        + min(email_count, 3) * 0.6
        + (1 if any(g in t for g in greetings) else 0)
        + (1 if any(s in t for s in signoffs) else 0)
    )

    return score >= 2.5


# ----------------------------
# HuggingFace API Embedding (NO local Torch)
# ----------------------------
def embed_text(text: str):
    if not text:
        return np.zeros((384,))

    response = requests.post(
        HF_EMBED_URL,
        headers=HEADERS,
        json={"inputs": text[:500]},
        timeout=30
    )

    arr = np.array(response.json()[0])
    return arr / (np.linalg.norm(arr) + 1e-9)


# ----------------------------
# Keyword + Semantic Ranking
# ----------------------------
def keyword_score(text, keywords):
    t = text.lower()
    return sum(1 for k in keywords if k in t) / len(keywords)


def fuzzy_keyword_score(text, keywords, threshold=0.8):
    t = text.lower().split()
    score = 0
    for k in keywords:
        if difflib.get_close_matches(k.lower(), t, n=1, cutoff=threshold):
            score += 1
    return score / len(keywords)


def semantic_score(text, label):
    examples = {
        "Resume": "resume cv work experience education skills projects profile",
        "Invoice": "invoice billing payment gst total amount due",
        "Contract": "contract legal agreement terms conditions parties obligations",
        "Report": "site report project report inspection issues findings summary"
    }

    try:
        emb1 = embed_text(text)
        emb2 = embed_text(examples[label])
        return float(np.dot(emb1, emb2)) * 0.3
    except Exception:
        return 0.0


# ----------------------------
# Main Classifier
# ----------------------------
def classify(text: str):
    t = text.lower()

    if len(t) < 10:
        return "Others", "Too little text", 0.1

    # Resume first
    r_kw = keyword_score(text, DOCUMENT_PATTERNS["Resume"]["keywords"])
    r_sem = semantic_score(text, "Resume")
    if r_kw + r_sem > 0.55:
        return "Resume", "Resume-like structure detected", min(1, r_kw + r_sem)

    # Email detection
    if is_email(text):
        return "Email", "Email style detected", 0.92

    # Other types
    scores = {}
    for doc_type, data in DOCUMENT_PATTERNS.items():
        if doc_type == "Invoice":
            k = fuzzy_keyword_score(text, data["keywords"])
        else:
            k = keyword_score(text, data["keywords"])
        s = semantic_score(text, doc_type)
        total = k + s
        scores[doc_type] = total

    best = max(scores, key=lambda x: scores[x])
    if scores[best] < 0.35:
        return "Others", "Low match scores", scores[best]

    return best, "Matched text patterns", min(scores[best], 1.0)


# ----------------------------
# API
# ----------------------------
@api.post("/classify")
async def classify_files(files: List[UploadFile] = File(...)):
    results = []

    for f in files:
        raw = await f.read()
        name = f.filename or "unknown"
        ext = name.split(".")[-1].lower()

        if ext == "pdf":
            text = extract_text_from_pdf_with_ocr(raw)
        elif ext in ["docx", "doc"]:
            text = extract_text_from_docx(raw)
        elif ext in ["ppt", "pptx"]:
            text = extract_text_from_pptx(raw)
        elif ext in ["jpg", "jpeg", "png", "bmp", "tiff", "webp"]:
            text = extract_text_from_image(raw)
        elif ext == "txt":
            text = raw.decode(errors="ignore")
        else:
            text = extract_text_from_image(raw)

        doc_type, reason, conf = classify(text)
        summary = f"{doc_type} containing {len(text.split())} words"

        result = ClassificationResult(
            file_name=name,
            file_type=ext.upper(),
            file_size=len(raw),
            document_type=doc_type,
            summary=summary,
            reason=reason,
            confidence=conf,
            extracted_text=text[:5000]
        )

        await db.classifications.insert_one(result.model_dump())
        results.append(result)

    return results


@api.get("/history")
async def history():
    return await db.classifications.find({}, {"_id": 0}).sort("timestamp", -1).limit(50).to_list(50)


app.include_router(api)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

@app.on_event("shutdown")
async def shutdown_db_client():
    client.close()


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("server:app", host="0.0.0.0", port=port)
