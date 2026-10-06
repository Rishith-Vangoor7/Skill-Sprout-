"""
SkillSprout – Bi-LSTM + Scaled Dot-Product Self-Attention Training Script
Trains a course recommendation model from Coursera data.
"""

import os
import sys
import json
import math
import random
import re
from collections import Counter

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# ─── Paths ───────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "public")
ARTIFACT_DIR = os.path.join(BASE_DIR, "backend", "model_artifacts")
os.makedirs(ARTIFACT_DIR, exist_ok=True)

# ─── Config ───────────────────────────────────────────────────────────────────
SEED = 42
TEST_SPLIT = 0.10
EMBED_DIM = 128
HIDDEN_DIM = 128
NUM_LSTM_LAYERS = 2
ATTN_HEADS = 4
DROPOUT = 0.3
BATCH_SIZE = 64
EPOCHS = 30
LR = 1e-3
MAX_SEQ_LEN = 20
TOP_K = 10
MIN_SKILL_FREQ = 2

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


# ─── Data Loading ─────────────────────────────────────────────────────────────
def load_data():
    print("📂 Loading CSV files …")
    p1 = pd.read_csv(os.path.join(DATA_DIR, "Coursera_full_part1.csv"))
    p2 = pd.read_csv(os.path.join(DATA_DIR, "Coursera_full_part2.csv"))
    df = pd.concat([p1, p2], ignore_index=True)
    df.dropna(subset=["Gained Skills", "Title"], inplace=True)
    df.drop_duplicates(subset=["Title"], inplace=True)
    df.reset_index(drop=True, inplace=True)
    print(f"   Total courses after merge & dedup: {len(df)}")
    return df


def parse_skills(raw: str):
    """Split a comma-separated skill string into a clean list."""
    skills = [s.strip().lower() for s in str(raw).split(",") if s.strip()]
    # Normalise: keep alpha-numeric + spaces
    skills = [re.sub(r"[^a-z0-9 ]", "", s).strip() for s in skills]
    return [s for s in skills if s]


# ─── Vocabulary ───────────────────────────────────────────────────────────────
def build_vocab(df):
    counter = Counter()
    for raw in df["Gained Skills"]:
        counter.update(parse_skills(raw))
    # Keep only skills that appear at least MIN_SKILL_FREQ times
    vocab = ["<PAD>", "<UNK>"] + [
        s for s, c in counter.most_common() if c >= MIN_SKILL_FREQ
    ]
    tok2id = {tok: idx for idx, tok in enumerate(vocab)}
    print(f"   Vocabulary size: {len(vocab)}")
    return vocab, tok2id


def skills_to_ids(skills, tok2id, max_len=MAX_SEQ_LEN):
    ids = [tok2id.get(s, tok2id["<UNK>"]) for s in skills[:max_len]]
    # Pad / truncate
    ids += [tok2id["<PAD>"]] * (max_len - len(ids))
    return ids[:max_len]


# ─── Dataset ──────────────────────────────────────────────────────────────────
class CourseDataset(Dataset):
    """
    For each course i, the INPUT is its skill sequence (teacher-forcing),
    and the TARGET is the set of courses that share >= 1 skill (collaborative-style).
    We sample one positive target per course per epoch.
    """

    def __init__(self, df, tok2id, course_skill_sets):
        self.df = df
        self.tok2id = tok2id
        self.course_skill_sets = course_skill_sets
        # Pre-compute skill tensors
        self.skill_tensors = []
        for raw in df["Gained Skills"]:
            skills = parse_skills(raw)
            ids = skills_to_ids(skills, tok2id)
            self.skill_tensors.append(torch.tensor(ids, dtype=torch.long))
        # Pre-compute positive pairs
        self.pairs = self._build_pairs()

    def _build_pairs(self):
        pairs = []
        n = len(self.df)
        for i in range(n):
            set_i = self.course_skill_sets[i]
            for j in range(n):
                if i == j:
                    continue
                if set_i & self.course_skill_sets[j]:
                    pairs.append((i, j))
        # Limit to manageable size
        random.shuffle(pairs)
        return pairs[:50000]

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        i, j = self.pairs[idx]
        return self.skill_tensors[i], self.skill_tensors[j]


# ─── Model ────────────────────────────────────────────────────────────────────
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

        Q = self.W_q(x).view(B, T, H, dk).transpose(1, 2)  # B,H,T,dk
        K = self.W_k(x).view(B, T, H, dk).transpose(1, 2)
        V = self.W_v(x).view(B, T, H, dk).transpose(1, 2)

        scores = (Q @ K.transpose(-2, -1)) / math.sqrt(dk)   # B,H,T,T
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
        lstm_out_dim = hidden_dim * 2           # bidirectional
        self.attn = ScaledDotProductAttention(lstm_out_dim, num_heads=num_heads)
        self.layer_norm = nn.LayerNorm(lstm_out_dim)
        self.proj = nn.Sequential(
            nn.Linear(lstm_out_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        # x: B x T
        pad_mask = (x != 0).float()                     # B x T
        emb = self.embed(x)                              # B x T x E
        lstm_out, _ = self.lstm(emb)                     # B x T x 2H
        attn_out, _ = self.attn(lstm_out)                # B x T x 2H
        out = self.layer_norm(lstm_out + attn_out)       # residual

        # Mean-pool over non-padding positions
        mask = pad_mask.unsqueeze(-1)                    # B x T x 1
        pooled = (out * mask).sum(1) / mask.sum(1).clamp(min=1)  # B x 2H
        return self.proj(pooled)                          # B x H


class SkillSproutModel(nn.Module):
    def __init__(self, vocab_size, num_courses,
                 embed_dim=EMBED_DIM, hidden_dim=HIDDEN_DIM):
        super().__init__()
        self.encoder = SkillEncoder(vocab_size, embed_dim, hidden_dim)
        self.course_embed = nn.Embedding(num_courses, hidden_dim)
        nn.init.xavier_uniform_(self.course_embed.weight)

    def forward(self, skill_seq):
        return self.encoder(skill_seq)       # B x H

    def score_all(self, skill_vec):
        """Dot-product score against all course embeddings."""
        return skill_vec @ self.course_embed.weight.T    # B x num_courses


# ─── Training ─────────────────────────────────────────────────────────────────
def train(model, loader, optimizer, device):
    model.train()
    total_loss = 0.0
    criterion = nn.CrossEntropyLoss()

    for skill_i, skill_j in loader:
        skill_i = skill_i.to(device)
        skill_j = skill_j.to(device)

        vec_i = model(skill_i)               # B x H
        vec_j = model(skill_j)               # B x H

        # In-batch contrastive: logits = B x B, diagonal is positive
        logits = vec_i @ vec_j.T             # B x B
        labels = torch.arange(logits.size(0), device=device)
        loss = criterion(logits, labels) + criterion(logits.T, labels)

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()

    return total_loss / max(len(loader), 1)


# ─── Evaluation ───────────────────────────────────────────────────────────────
@torch.no_grad()
def evaluate(model, df, tok2id, course_skill_sets, test_indices, device, k=TOP_K):
    model.eval()
    num_courses = len(df)

    # Precompute all course embeddings in one pass
    all_skill_seqs = []
    for raw in df["Gained Skills"]:
        skills = parse_skills(raw)
        ids = skills_to_ids(skills, tok2id)
        all_skill_seqs.append(ids)

    seq_tensor = torch.tensor(all_skill_seqs, dtype=torch.long, device=device)
    # Batch encode to avoid OOM
    bs = 256
    all_vecs = []
    for start in range(0, num_courses, bs):
        vecs = model(seq_tensor[start:start + bs])
        all_vecs.append(vecs)
    all_vecs = torch.cat(all_vecs, dim=0)          # N x H

    # Score matrix via dot-product of encoded skill vectors
    scores_mat = all_vecs @ all_vecs.T              # N x N

    recall_list, prec_list = [], []
    for idx in test_indices:
        true_pos = {
            j for j in range(num_courses)
            if j != idx and course_skill_sets[idx] & course_skill_sets[j]
        }
        if not true_pos:
            continue
        row_scores = scores_mat[idx].cpu()
        row_scores[idx] = -1e9  # exclude self
        top_k_indices = torch.topk(row_scores, k).indices.tolist()
        hits = len(set(top_k_indices) & true_pos)
        recall_list.append(hits / len(true_pos))
        prec_list.append(hits / k)

    recall = float(np.mean(recall_list)) if recall_list else 0.0
    precision = float(np.mean(prec_list)) if prec_list else 0.0
    return recall, precision


# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    df = load_data()

    # Build vocab
    vocab, tok2id = build_vocab(df)
    vocab_path = os.path.join(ARTIFACT_DIR, "vocab.json")
    with open(vocab_path, "w") as f:
        json.dump({"vocab": vocab, "tok2id": tok2id}, f)
    print(f"   Saved vocab -> {vocab_path}")

    # Build per-course skill sets
    course_skill_sets = []
    for raw in df["Gained Skills"]:
        course_skill_sets.append(set(parse_skills(raw)))

    # Build course map
    course_map = {}
    for idx, row in df.iterrows():
        course_map[str(idx)] = {
            "id": idx,
            "title": row.get("Title", ""),
            "subject": row.get("Subject", ""),
            "institution": row.get("Institution", ""),
            "level": row.get("Level", ""),
            "duration": row.get("Duration", ""),
            "skills": parse_skills(str(row.get("Gained Skills", ""))),
            "rating": float(row.get("Rate", 0) or 0),
            "reviews": int(str(row.get("Reviews", 0)).replace(",", "") or 0),
            "learning_product": row.get("Learning Product", ""),
        }

    course_map_path = os.path.join(ARTIFACT_DIR, "course_map.json")
    with open(course_map_path, "w") as f:
        json.dump(course_map, f)
    print(f"   Saved course_map ({len(course_map)} courses) -> {course_map_path}")

    # Train / test split
    all_indices = list(range(len(df)))
    random.shuffle(all_indices)
    split = int(len(all_indices) * (1 - TEST_SPLIT))
    train_indices = set(all_indices[:split])
    test_indices = all_indices[split:]

    # Build dataset from train indices
    train_df = df.iloc[list(train_indices)].reset_index(drop=True)
    train_skill_sets = [course_skill_sets[i] for i in list(train_indices)]
    dataset = CourseDataset(train_df, tok2id, train_skill_sets)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True,
                        num_workers=0, drop_last=True)

    # Model
    model = SkillSproutModel(len(vocab), len(df)).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    print("\nTraining ...")
    for epoch in range(1, EPOCHS + 1):
        loss = train(model, loader, optimizer, device)
        scheduler.step()
        if epoch % 5 == 0 or epoch == 1:
            print(f"   Epoch {epoch:3d}/{EPOCHS}  loss={loss:.4f}  lr={scheduler.get_last_lr()[0]:.6f}")

    # Evaluation
    print("\nEvaluating on test split ...")
    recall, precision = evaluate(
        model, df, tok2id, course_skill_sets, test_indices, device, k=TOP_K
    )
    print(f"   Recall@{TOP_K}   = {recall:.4f}")
    print(f"   Precision@{TOP_K} = {precision:.4f}")

    # Save model
    model_path = os.path.join(ARTIFACT_DIR, "bilstm_attention_model.pth")
    torch.save({
        "model_state_dict": model.state_dict(),
        "vocab_size": len(vocab),
        "num_courses": len(df),
        "embed_dim": EMBED_DIM,
        "hidden_dim": HIDDEN_DIM,
        "metrics": {"recall_at_10": recall, "precision_at_10": precision},
    }, model_path)
    print(f"\nModel saved -> {model_path}")
    print("Training complete!")


if __name__ == "__main__":
    main()
