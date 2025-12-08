import os
import io
import uuid
import logging
import re

from pptx import Presentation
from pptx.enum.shapes import MSO_SHAPE_TYPE
from datetime import datetime, timezone
from pathlib import Path
from typing import List

import numpy as np
import cv2
from PIL import Image
import PyPDF2
from docx import Document as DocxDocument

from fastapi import FastAPI, File, UploadFile, APIRouter
from starlette.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from pdf2image import convert_from_bytes

from motor.motor_asyncio import AsyncIOMotorClient
import easyocr
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F

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

reader = easyocr.Reader(['en'], gpu=False)

# HuggingFace model for embeddings
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
hf_model_name = "sentence-transformers/all-MiniLM-L6-v2"
tokenizer = AutoTokenizer.from_pretrained(hf_model_name)
model = AutoModel.from_pretrained(hf_model_name).to(device)

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
# Response Model
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
# OCR & Extraction
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

def extract_text_from_pdf_with_ocr(raw):
    try:
        text = extract_text_from_pdf(raw)
        if text.strip():
            return text.strip()
        pages = convert_from_bytes(raw)
        ocr_text = ""
        for page in pages:
            arr = np.array(page)
            processed = preprocess_image(arr)
            result = reader.readtext(processed, detail=0)
            ocr_text += " ".join(str(t) for t in result) + "\n"
        return ocr_text.strip()
    except Exception as e:
        logging.error(f"PDF OCR error: {e}")
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
# Email Detector
# ----------------------------
def is_email(text: str, debug: bool = False) -> bool:
    t = (text or "").lower()
    headers = ["from:", "to:", "subject:", "cc:", "bcc:", "date:"]
    header_count = sum(1 for h in headers if h in t)
    emails = re.findall(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", t)
    email_count = len(emails)
    greetings = ["\nhi ", "\nhello ", "\ndear ", "\ndear "]
    signoffs = ["regards", "sincerely", "best regards", "thanks", "thank you"]
    has_greeting = any(g in t for g in greetings)
    has_signoff = any(s in t for s in signoffs)
    paragraphs = [p.strip() for p in t.split("\n\n") if p.strip()]
    long_paragraphs = sum(1 for p in paragraphs if len(p.split()) > 20)
    resume_indicators = ["education", "experience", "skills", "projects",
                         "work experience", "objective", "curriculum vitae",
                         "summary", "certifications", "references"]
    resume_matches = sum(1 for w in resume_indicators if w in t)
    score = 0.0
    score += min(header_count, 3) * 1.0
    score += min(email_count, 3) * 0.6
    score += 1.0 if has_greeting else 0.0
    score += 1.0 if has_signoff else 0.0
    score += min(long_paragraphs, 2) * 0.8
    score -= min(resume_matches, 3) * 1.2
    if debug:
        print("EMAIL DEBUG:", f"headers={header_count}", f"emails={email_count}",
              f"greeting={has_greeting}", f"signoff={has_signoff}",
              f"long_paras={long_paragraphs}", f"resume_matches={resume_matches}",
              f"score={score:.2f}")
    return score >= 2.5

# ----------------------------
# Keyword + Semantic Scoring
# ----------------------------
def keyword_score(text, keywords):
    t = text.lower()
    count = sum(1 for k in keywords if k.lower() in t)
    return min(0.5, count / len(keywords))

def embed_text(text: str):
    """Get sentence embedding from HuggingFace model."""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    # Mean pooling
    embeddings = outputs.last_hidden_state.mean(dim=1)
    embeddings = F.normalize(embeddings, p=2, dim=1)
    return embeddings.cpu().numpy()

def semantic_score(text, label):
    examples = {
        "Resume": "resume cv curriculum vitae work experience skills education profile",
        "Invoice": "invoice bill payment receipt items charges gst total amount",
        "Contract": "legal agreement contract terms conditions parties obligations signatures",
        "Report": "project report site inspection report construction report progress report summary details issues assessment findings status updates"
    }
    if label not in examples:
        return 0.0
    try:
        emb1 = embed_text(text[:400])
        emb2 = embed_text(examples[label])
        sim = np.dot(emb1, emb2.T)[0][0]
        return sim * 0.3
    except Exception:
        return 0.0

# ----------------------------
# Classification Core
# ----------------------------
def classify(text: str):
    text_l = text.lower().strip()
    if not text_l or len(text_l) < 10:
        return "Others", "Very little text found", 0.2
    resume_keywords = DOCUMENT_PATTERNS["Resume"]["keywords"]
    resume_kw = keyword_score(text, resume_keywords)
    resume_sem = semantic_score(text, "Resume")
    resume_score = resume_kw + resume_sem
    if resume_score >= 0.55:
        return "Resume", "Strong resume patterns detected", min(resume_score, 1.0)
    if is_email(text):
        return "Email", "Email structure detected", 0.9
    scores = {}
    reasons = {}
    for doc_type, data in DOCUMENT_PATTERNS.items():
        k = keyword_score(text, data["keywords"])
        s = semantic_score(text, doc_type)
        total = k + s
        if doc_type == "Report":
            report_indicators = ["project details", "site report", "project report",
                                 "status:", "priority:", "due date", "assigned to",
                                 "description:", "summary:", "issues", "in progress",
                                 "client name", "client email", "auditor", "site",
                                 "renovation", "inspection"]
            bonus = sum(1 for x in report_indicators if x in text_l)
            total += min(0.30, bonus * 0.05)
        scores[doc_type] = total
        reasons[doc_type] = f"Keyword score={k:.2f}, semantic score={s:.2f}"
    best = max(scores, key=lambda x: scores[x])
    best_score = scores[best]
    if best_score < 0.40:
        return "Others", "Low confidence across all document types", best_score
    return best, "Matched linguistic + semantic patterns", min(best_score, 1.0)

# ----------------------------
# API Routes
# ----------------------------
@api.post("/classify")
async def classify_files(files: List[UploadFile] = File(...)):
    results = []
    for file in files:
        raw = await file.read()
        name = file.filename or "unknown"
        ext = name.split(".")[-1].lower() if "." in name else ""
        if ext == "pdf":
            text = extract_text_from_pdf_with_ocr(raw)
        elif ext in ["docx", "doc"]:
            text = extract_text_from_docx(raw)
        elif ext in ["pptx", "ppt"]:
            text = extract_text_from_pptx(raw)
        elif ext in ["jpg", "jpeg", "png", "webp", "tiff", "bmp"]:
            text = extract_text_from_image(raw)
        elif ext in ["txt"]:
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
        doc = result.model_dump()
        doc["timestamp"] = doc["timestamp"].isoformat()
        await db.classifications.insert_one(doc)
        results.append(result)
    return results

@api.get("/history")
async def history():
    items = await db.classifications.find({}, {"_id": 0}).sort("timestamp", -1).limit(50).to_list(50)
    for x in items:
        if isinstance(x["timestamp"], str):
            x["timestamp"] = datetime.fromisoformat(x["timestamp"])
    return items

app.include_router(api)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@app.on_event("shutdown")
async def shutdown_db_client():
    client.close()

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("server:app", host="0.0.0.0", port=port)
