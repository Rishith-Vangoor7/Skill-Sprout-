"""
SkillSprout FastAPI Backend
Endpoints: /api/upload-resume, /api/recommend, /api/courses
"""

import os
import re
import json
import math
import io
from typing import List, Optional
from pathlib import Path

import torch
import torch.nn as nn
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ─── Try to import optional document libs ─────────────────────────────────────
try:
    import pypdf
    HAS_PYPDF = True
except ImportError:
    HAS_PYPDF = False

try:
    from docx import Document as DocxDocument
    HAS_DOCX = True
except ImportError:
    HAS_DOCX = False

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
ARTIFACT_DIR = BASE_DIR / "model_artifacts"

VOCAB_PATH = ARTIFACT_DIR / "vocab.json"
COURSE_MAP_PATH = ARTIFACT_DIR / "course_map.json"
MODEL_PATH = ARTIFACT_DIR / "bilstm_attention_model.pth"

# ─── Model Constants (must match train.py) ────────────────────────────────────
EMBED_DIM = 128
HIDDEN_DIM = 128
NUM_LSTM_LAYERS = 2
ATTN_HEADS = 4
DROPOUT = 0.3
MAX_SEQ_LEN = 20
TOP_K = 10

# ─── App ──────────────────────────────────────────────────────────────────────
app = FastAPI(title="SkillSprout API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Global state ─────────────────────────────────────────────────────────────
_vocab_data: dict = {}
_course_map: dict = {}
_model = None
_device = torch.device("cpu")
_all_course_vecs = None   # pre-computed for fast retrieval


# ─── Model Definition (same as train.py) ──────────────────────────────────────
class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_model, num_heads=4):
        super().__init__()
        assert d_model % num_heads == 0
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x, mask=None):
        B, T, D = x.size()
        H = self.num_heads
        dk = self.d_k
        Q = self.W_q(x).view(B, T, H, dk).transpose(1, 2)
        K = self.W_k(x).view(B, T, H, dk).transpose(1, 2)
        V = self.W_v(x).view(B, T, H, dk).transpose(1, 2)
        scores = (Q @ K.transpose(-2, -1)) / math.sqrt(dk)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attn = torch.softmax(scores, dim=-1)
        out = (attn @ V).transpose(1, 2).contiguous().view(B, T, D)
        return self.out_proj(out), attn


class SkillEncoder(nn.Module):
    def __init__(self, vocab_size, embed_dim=EMBED_DIM, hidden_dim=HIDDEN_DIM,
                 num_layers=NUM_LSTM_LAYERS, num_heads=ATTN_HEADS, dropout=DROPOUT):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            embed_dim, hidden_dim, num_layers=num_layers,
            batch_first=True, bidirectional=True, dropout=dropout
        )
        lstm_out_dim = hidden_dim * 2
        self.attn = ScaledDotProductAttention(lstm_out_dim, num_heads=num_heads)
        self.layer_norm = nn.LayerNorm(lstm_out_dim)
        self.proj = nn.Sequential(
            nn.Linear(lstm_out_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        pad_mask = (x != 0).float()
        emb = self.embed(x)
        lstm_out, _ = self.lstm(emb)
        attn_out, _ = self.attn(lstm_out)
        out = self.layer_norm(lstm_out + attn_out)
        mask = pad_mask.unsqueeze(-1)
        pooled = (out * mask).sum(1) / mask.sum(1).clamp(min=1)
        return self.proj(pooled)


class SkillSproutModel(nn.Module):
    def __init__(self, vocab_size, num_courses,
                 embed_dim=EMBED_DIM, hidden_dim=HIDDEN_DIM):
        super().__init__()
        self.encoder = SkillEncoder(vocab_size, embed_dim, hidden_dim)
        self.course_embed = nn.Embedding(num_courses, hidden_dim)

    def forward(self, skill_seq):
        return self.encoder(skill_seq)


# ─── Startup ──────────────────────────────────────────────────────────────────
@app.on_event("startup")
def startup_event():
    global _vocab_data, _course_map, _model, _device, _all_course_vecs

    # Load vocab
    if VOCAB_PATH.exists():
        with open(VOCAB_PATH) as f:
            _vocab_data = json.load(f)
        print(f"Loaded vocab: {len(_vocab_data['vocab'])} tokens")
    else:
        print("WARNING: vocab.json not found – run ml/train.py first")

    # Load course map
    if COURSE_MAP_PATH.exists():
        with open(COURSE_MAP_PATH) as f:
            _course_map = json.load(f)
        print(f"Loaded course_map: {len(_course_map)} courses")
    else:
        print("WARNING: course_map.json not found – run ml/train.py first")

    # Load model
    if MODEL_PATH.exists() and _vocab_data and _course_map:
        checkpoint = torch.load(MODEL_PATH, map_location="cpu")
        _model = SkillSproutModel(
            vocab_size=checkpoint["vocab_size"],
            num_courses=checkpoint["num_courses"],
        )
        _model.load_state_dict(checkpoint["model_state_dict"])
        _model.eval()
        print(f"Loaded model: vocab={checkpoint['vocab_size']}, courses={checkpoint['num_courses']}")
        print(f"  Metrics: {checkpoint.get('metrics', {})}")

        # Pre-compute all course vectors for fast retrieval
        _precompute_course_vecs()
    else:
        print("WARNING: Model not loaded – run ml/train.py first")


def _precompute_course_vecs():
    """Encode every course's skill sequence once at startup."""
    global _all_course_vecs
    if _model is None or not _course_map:
        return
    tok2id = _vocab_data["tok2id"]
    seqs = []
    for idx in range(len(_course_map)):
        course = _course_map[str(idx)]
        ids = _skills_to_ids(course["skills"], tok2id)
        seqs.append(ids)
    seq_tensor = torch.tensor(seqs, dtype=torch.long)
    with torch.no_grad():
        bs = 256
        vecs = []
        for start in range(0, len(seqs), bs):
            vecs.append(_model(seq_tensor[start:start + bs]))
        _all_course_vecs = torch.cat(vecs, dim=0)   # N x H
    print(f"Pre-computed {_all_course_vecs.shape[0]} course vectors")


# ─── Helpers ──────────────────────────────────────────────────────────────────
def _parse_skills(raw: str) -> List[str]:
    skills = [s.strip().lower() for s in str(raw).split(",") if s.strip()]
    skills = [re.sub(r"[^a-z0-9 ]", "", s).strip() for s in skills]
    return [s for s in skills if s]


def _skills_to_ids(skills: List[str], tok2id: dict, max_len: int = MAX_SEQ_LEN) -> List[int]:
    ids = [tok2id.get(s, tok2id.get("<UNK>", 1)) for s in skills[:max_len]]
    ids += [0] * (max_len - len(ids))
    return ids[:max_len]


def _extract_text_from_pdf(file_bytes: bytes) -> str:
    if not HAS_PYPDF:
        raise HTTPException(500, "pypdf not installed")
    reader = pypdf.PdfReader(io.BytesIO(file_bytes))
    text = " ".join(page.extract_text() or "" for page in reader.pages)
    return text


def _extract_text_from_docx(file_bytes: bytes) -> str:
    if not HAS_DOCX:
        raise HTTPException(500, "python-docx not installed")
    doc = DocxDocument(io.BytesIO(file_bytes))
    text = " ".join(p.text for p in doc.paragraphs)
    return text


def _extract_skills_from_text(text: str) -> List[str]:
    """Match text against known vocab skills via substring matching."""
    if not _vocab_data:
        return []
    tok2id = _vocab_data["tok2id"]
    text_lower = text.lower()
    # Remove special chars
    text_clean = re.sub(r"[^a-z0-9 \n]", " ", text_lower)
    found = []
    for skill in tok2id.keys():
        if skill in ("<PAD>", "<UNK>"):
            continue
        if re.search(r"\b" + re.escape(skill) + r"\b", text_clean):
            found.append(skill)
    return list(set(found))[:50]


SAMPLE_PROFILE_SKILLS = [
    "python", "machine learning", "data analysis", "sql",
    "statistics", "data visualization", "deep learning",
    "tensorflow", "pandas", "communication"
]


# ─── Pydantic Models ──────────────────────────────────────────────────────────
class RecommendRequest(BaseModel):
    skills: List[str]
    top_k: Optional[int] = TOP_K
    subject_filter: Optional[str] = None
    level_filter: Optional[str] = None


class CourseResult(BaseModel):
    id: int
    title: str
    subject: str
    institution: str
    level: str
    duration: str
    skills: List[str]
    rating: float
    reviews: int
    match_score: float
    matched_skills: List[str]
    learning_product: str


# ─── Routes ───────────────────────────────────────────────────────────────────
@app.get("/")
def root():
    return {"message": "SkillSprout API is running"}


@app.get("/api/health")
def health():
    return {
        "model_loaded": _model is not None,
        "vocab_size": len(_vocab_data.get("vocab", [])),
        "num_courses": len(_course_map),
    }


@app.post("/api/upload-resume")
async def upload_resume(file: Optional[UploadFile] = File(None)):
    """
    Extract skills from uploaded PDF/DOCX.
    If no file provided, returns sample profile skills.
    """
    if file is None:
        return {"skills": SAMPLE_PROFILE_SKILLS, "source": "sample"}

    file_bytes = await file.read()
    filename = file.filename or ""
    ext = filename.rsplit(".", 1)[-1].lower()

    if ext == "pdf":
        text = _extract_text_from_pdf(file_bytes)
    elif ext in ("docx", "doc"):
        text = _extract_text_from_docx(file_bytes)
    else:
        raise HTTPException(400, f"Unsupported file type: .{ext}")

    skills = _extract_skills_from_text(text)
    if not skills:
        skills = SAMPLE_PROFILE_SKILLS

    return {"skills": skills, "source": filename}


@app.post("/api/recommend")
def recommend(req: RecommendRequest) -> List[CourseResult]:
    """
    Run inference: encode user skills -> BiLSTM -> attention -> dot-product ranking.
    Returns Top-K ranked courses with match percentages.
    """
    if not _vocab_data or not _course_map:
        raise HTTPException(503, "Model artifacts not loaded. Run ml/train.py first.")

    tok2id = _vocab_data["tok2id"]
    user_skills = _parse_skills(", ".join(req.skills))

    # Pre-filter candidate indices based on subject/level criteria
    candidate_indices = []
    for idx_str, course in _course_map.items():
        if req.subject_filter and req.subject_filter.lower() not in course["subject"].lower():
            continue
        if req.level_filter and req.level_filter.lower() not in course["level"].lower():
            continue
        candidate_indices.append(int(idx_str))

    if not candidate_indices:
        return []

    results = []

    if _model is not None and _all_course_vecs is not None:
        # ── Neural inference path ──────────────────────────────────────────
        ids = _skills_to_ids(user_skills, tok2id)
        seq_tensor = torch.tensor([ids], dtype=torch.long)
        with torch.no_grad():
            user_vec = _model(seq_tensor)          # 1 x H
        all_scores = (user_vec @ _all_course_vecs.T).squeeze(0)   # N

        cand_tensor = torch.tensor(candidate_indices, dtype=torch.long)
        cand_scores = all_scores[cand_tensor]

        top_k = min(req.top_k or TOP_K, len(candidate_indices))
        topk_res = torch.topk(cand_scores, top_k)
        top_cand_indices = topk_res.indices.tolist()
        score_values = topk_res.values.tolist()

        # Normalise scores to [0, 1]
        min_s, max_s = min(score_values), max(score_values)
        rng = max_s - min_s if max_s != min_s else 1.0

        user_skill_set = set(user_skills)

        for cand_idx, raw_score in zip(top_cand_indices, score_values):
            idx = candidate_indices[cand_idx]
            course = _course_map.get(str(idx))
            if not course:
                continue

            course_skills = set(course["skills"])
            matched = list(user_skill_set & course_skills)
            # Blend model score with explicit skill overlap
            model_score = (raw_score - min_s) / rng
            overlap_score = len(matched) / max(len(user_skill_set), 1)
            blend = 0.6 * model_score + 0.4 * overlap_score
            match_pct = round(min(blend * 100, 99.9), 1)

            results.append(CourseResult(
                id=course["id"],
                title=course["title"],
                subject=course["subject"],
                institution=course["institution"],
                level=course["level"],
                duration=course["duration"],
                skills=course["skills"][:15],
                rating=course["rating"],
                reviews=course["reviews"],
                match_score=match_pct,
                matched_skills=matched[:10],
                learning_product=course["learning_product"],
            ))
    else:
        # ── Fallback: pure skill-overlap scoring ──────────────────────────
        user_skill_set = set(user_skills)
        scored = []
        for idx in candidate_indices:
            course = _course_map[str(idx)]
            course_skills = set(course["skills"])
            matched = user_skill_set & course_skills
            score = len(matched) / max(len(user_skill_set), 1)
            scored.append((score, matched, course))
        scored.sort(key=lambda x: x[0], reverse=True)

        for score, matched, course in scored[: req.top_k or TOP_K]:
            results.append(CourseResult(
                id=course["id"],
                title=course["title"],
                subject=course["subject"],
                institution=course["institution"],
                level=course["level"],
                duration=course["duration"],
                skills=course["skills"][:15],
                rating=course["rating"],
                reviews=course["reviews"],
                match_score=round(score * 100, 1),
                matched_skills=list(matched)[:10],
                learning_product=course["learning_product"],
            ))

    return results


@app.get("/api/courses")
def get_courses(
    search: Optional[str] = None,
    subject: Optional[str] = None,
    level: Optional[str] = None,
    limit: int = 50,
    offset: int = 0,
):
    """Serve course catalog with optional search/filter."""
    courses = list(_course_map.values())

    if search:
        s = search.lower()
        courses = [
            c for c in courses
            if s in c["title"].lower()
            or s in c["subject"].lower()
            or any(s in sk for sk in c["skills"])
        ]
    if subject:
        courses = [c for c in courses if subject.lower() in c["subject"].lower()]
    if level:
        courses = [c for c in courses if level.lower() in c["level"].lower()]

    total = len(courses)
    page = courses[offset: offset + limit]

    return {
        "total": total,
        "offset": offset,
        "limit": limit,
        "courses": page,
    }


@app.get("/api/subjects")
def get_subjects():
    subjects = sorted(set(c["subject"] for c in _course_map.values() if c["subject"]))
    return {"subjects": subjects}


@app.get("/api/levels")
def get_levels():
    levels = sorted(set(c["level"] for c in _course_map.values() if c["level"]))
    return {"levels": levels}
