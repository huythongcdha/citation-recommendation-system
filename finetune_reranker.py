# finetune_pubmedbert_triplet_hybrid.py
# -*- coding: utf-8 -*-

import warnings
warnings.filterwarnings("ignore")
import os
from pathlib import Path
import math
import random
from typing import List, Dict, Any, Tuple, Iterable

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from transformers import (
    AutoTokenizer, AutoConfig, BertForSequenceClassification,
    get_cosine_schedule_with_warmup
)
from peft import LoraConfig, get_peft_model

import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.dataset as ds

from tqdm.auto import tqdm
import faiss

# =============== CONFIG (điều chỉnh đường dẫn theo máy bạn) ===============
# Triplet parquet (anchor= citing, pos= cited, negs= list<=8 ids)
TRIPLET_PATH = Path(r"E:\Citation Recommendation\data\sample\triplet_sample_citations_100k_K_8.parquet")

# Papers metadata: partitioned parquet trong thư mục có 256 subfolders part=00..ff
PAPERS_ROOT = Path(r"E:\Citation Recommendation\data\stage\papers_with_abstracts_cleaned")

# FAISS index & BM25 index
FAISS_INDEX_PATH = Path(r"E:\Citation Recommendation\data\indexes\ivf_flat_ip_11m.faiss")
BM25_INDEX_DIR   = Path(r"E:\Citation Recommendation\data\bm25\lucene-index-papers_11m")

# SPECTER2 (retriever) cho hybrid val MRR (query side)
S2_BASE      = "allenai/specter2_base"
S2_ADAPTER_TA   = "allenai/specter2"              # doc-style (title+abstract)
S2_ADAPTER_ADHQ = "allenai/specter2_adhoc_query"  # ad-hoc query (context)

# PubMedBERT cross-encoder (reranker)
CE_MODEL_NAME = "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext"
MAX_LEN = 512

# Train setup
PER_DEVICE_TRAIN_BS = 32
GRAD_ACCUM_STEPS    = 2
NUM_EPOCHS          = 3
NUM_WORKERS         = 8
PIN_MEMORY          = True

# Optim & Scheduler
LR            = 2e-4
WEIGHT_DECAY  = 0.01
BETAS         = (0.9, 0.98)
ADAM_EPS      = 1e-8
WARMUP_RATIO  = 0.06
FP16          = True
MAX_GRAD_NORM = 1.0

# LoRA
LORA_CFG = dict(
    r=8, lora_alpha=16, lora_dropout=0.1, bias="none",
    target_modules=["query","key","value","output.dense"]
)

# Token budgets & truncate order
BUDGET = {
    "query":    {"title": 30, "abstract": 150, "context": 80,  "authors": 20},
    "document": {"title": 30, "abstract": 160, "authors": 20,  "venue": 10},
}
TRUNC_ORDER = [
    ("document", "venue"),
    ("document", "authors"),
    ("query",    "authors"),
    ("query",    "abstract"),
    ("document", "abstract"),
    ("query",    "context"),
    ("query",    "title"),
    ("document", "title"),
]
SPECIAL_PREFIXES = {
    "query": {
        "title":   "Title: ",
        "abstract":"Abstract: ",
        "context": "Context: ",
        "authors": "Authors: ",
    },
    "document": {
        "title":   "Title: ",
        "abstract":"Abstract: ",
        "authors": "Authors: ",
        "venue":   "Venue: ",
    }
}

# Validation (hybrid) config
TOPK_RETRIEVER = 100
ALPHA_HYBRID   = 0.5        # blend α * zscore(FAISS) + (1-α) * zscore(BM25)
RRF_K          = 60         # nếu cần RRF (không dùng ở đây)
W_TA, W_CTX    = 0.5, 0.5   # early fusion SPECTER2

# Misc
SEED = 42
OUT_DIR = Path(r"E:\Citation Recommendation\models\pubmedbert_lora_triplet_hybrid")
OUT_DIR.mkdir(parents=True, exist_ok=True)
BEST_CKPT = OUT_DIR / "best_100k_by_MRR.pt"


# =================== Utils ===================
def seed_everything(seed=SEED):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
seed_everything(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =================== PartitionedMetadataLoader ===================
class PartitionedMetadataLoader:
    """
    papers_with_abstracts_cleaned/
      part=00/
         xxx.parquet
      ...
      part=ff/
    Partition bằng (corpusid & 255) -> hex 2 ký tự.
    Đọc theo nhóm corpusid để giảm số lần mở file.
    """
    def __init__(self, root: Path, required_cols: List[str] = None, cache_parts: int = 16):
        self.root = root
        self.required_cols = required_cols or ["corpusid","title","abstract","authors_concat","venue","year"]
        self.cache_parts = cache_parts
        self._cache: Dict[str, pd.DataFrame] = {}  # LRU đơn giản theo kích thước

    def _part_of(self, corpusid: int) -> str:
        return f"part={corpusid & 0xff:02x}"

    def _load_part_df(self, part: str) -> pd.DataFrame:
        # Cache đơn giản
        if part in self._cache:
            return self._cache[part]

        part_dir = self.root / part
        if not part_dir.exists():
            df = pd.DataFrame(columns=self.required_cols)
        else:
            files = sorted(part_dir.glob("*.parquet"))
            if not files:
                df = pd.DataFrame(columns=self.required_cols)
            else:
                # ✅ Đọc schema bằng PyArrow thay vì dùng nrows=1
                try:
                    schema = pq.read_schema(str(files[0]))
                    avail_cols = set(schema.names)
                except Exception:
                    # fallback an toàn: đọc full lần đầu (ít gặp)
                    tmp = pd.read_parquet(files[0])
                    avail_cols = set(tmp.columns)

                cols_to_read = [c for c in self.required_cols if c in avail_cols]

                # Nếu thư mục part có nhiều file, có thể gộp nhanh bằng dataset:
                # (an toàn/nhanh hơn khi part có nhiều mảnh)
                try:
                    # Ưu tiên đọc cả thư mục part một lần nếu có nhiều file
                    if len(files) > 1:
                        import pyarrow.dataset as ds
                        table = ds.dataset(str(part_dir)).to_table(columns=cols_to_read)
                    else:
                        table = pq.read_table(str(files[0]), columns=cols_to_read)
                    df = table.to_pandas()
                except Exception:
                    # Fallback: đọc bằng pandas (một file)
                    df = pd.read_parquet(files[0], columns=cols_to_read)

        # Chuẩn hoá kiểu & bổ sung cột còn thiếu
        if "corpusid" in df.columns:
            df["corpusid"] = pd.to_numeric(df["corpusid"], errors="coerce").astype("Int64").astype("int64", errors="ignore")

        for c in self.required_cols:
            if c not in df.columns:
                df[c] = "" if c != "year" else np.nan

        # Cache (LRU thô sơ)
        if len(self._cache) >= self.cache_parts:
            self._cache.pop(next(iter(self._cache)))
        self._cache[part] = df
        return df

    def get_many(self, corpusids: Iterable[int]) -> pd.DataFrame:
        corpusids = list(map(int, corpusids))
        by_part: Dict[str, List[int]] = {}
        for cid in corpusids:
            by_part.setdefault(self._part_of(cid), []).append(cid)

        frames = []
        for part, ids in by_part.items():
            df_part = self._load_part_df(part)
            if df_part.empty:
                continue
            sub = df_part[df_part["corpusid"].isin(ids)]
            frames.append(sub)
        if not frames:
            return pd.DataFrame(columns=self.required_cols)
        out = pd.concat(frames, axis=0, ignore_index=True)
        # đảm bảo đầy đủ cột
        for c in self.required_cols:
            if c not in out.columns:
                out[c] = "" if c not in ("year",) else np.nan
        return out


# =================== CE Tokenizer & packing ===================
def build_ce_tokenizer():
    tok = AutoTokenizer.from_pretrained(CE_MODEL_NAME, use_fast=True)
    assert tok.sep_token_id is not None and tok.cls_token_id is not None
    return tok

ce_tokenizer = build_ce_tokenizer()

def _encode_text(text: str) -> List[int]:
    return ce_tokenizer.encode(text, add_special_tokens=False, truncation=False)

def _budgeted_tokens(piece: str, side: str, field: str) -> List[int]:
    prefix = SPECIAL_PREFIXES[side][field]
    return _encode_text(prefix + (piece or ""))

def _pack_and_truncate(example: Dict[str, str]) -> Dict[str, torch.Tensor]:
    q_title    = _budgeted_tokens(example["q_title"],    "query",    "title")
    q_abs      = _budgeted_tokens(example["q_abstract"], "query",    "abstract")
    q_ctx      = _budgeted_tokens(example["q_context"],  "query",    "context")
    q_auth     = _budgeted_tokens(example["q_authors"],  "query",    "authors")

    d_title    = _budgeted_tokens(example["d_title"],    "document", "title")
    d_abs      = _budgeted_tokens(example["d_abstract"], "document", "abstract")
    d_auth     = _budgeted_tokens(example["d_authors"],  "document", "authors")
    d_venue    = _budgeted_tokens(example["d_venue"],    "document", "venue")

    parts = {
        ("query","title"): q_title,    ("query","abstract"): q_abs,
        ("query","context"): q_ctx,    ("query","authors"): q_auth,
        ("document","title"): d_title, ("document","abstract"): d_abs,
        ("document","authors"): d_auth,("document","venue"): d_venue,
    }

    def assemble():
        ids = [ce_tokenizer.cls_token_id]; seg = [0]
        for key in [("query","title"),("query","abstract"),("query","context"),("query","authors")]:
            ids.extend(parts[key]); seg.extend([0]*len(parts[key]))
            ids.append(ce_tokenizer.sep_token_id); seg.append(0)
        for key in [("document","title"),("document","abstract"),("document","authors"),("document","venue")]:
            ids.extend(parts[key]); seg.extend([1]*len(parts[key]))
            ids.append(ce_tokenizer.sep_token_id); seg.append(1)
        return ids, seg

    def clip_to_budget(side, field):
        parts[(side,field)] = parts[(side,field)][:BUDGET[side][field]]

    ids, seg = assemble()
    if len(ids) <= MAX_LEN:
        attn = [1]*len(ids)
        pad = MAX_LEN - len(ids)
        if pad>0:
            ids += [ce_tokenizer.pad_token_id]*pad
            seg += [0]*pad
            attn += [0]*pad
        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "token_type_ids": torch.tensor(seg, dtype=torch.long),
            "attention_mask": torch.tensor(attn, dtype=torch.long),
        }

    # Pass 1: cắt về đúng budget theo thứ tự
    for (side, field) in TRUNC_ORDER:
        ids, seg = assemble()
        if len(ids) <= MAX_LEN: break
        clip_to_budget(side, field)

    ids, seg = assemble()
    # Pass 2: nếu vẫn >512, cắt step nhỏ
    def step_trim(side, field, step=4):
        if len(parts[(side,field)]) > step:
            parts[(side,field)] = parts[(side,field)][:-step]
        else:
            parts[(side,field)] = parts[(side,field)][:1]

    while len(ids) > MAX_LEN:
        for (side, field) in TRUNC_ORDER:
            if len(ids) <= MAX_LEN: break
            step_trim(side, field, step=4)
            ids, seg = assemble()

    attn = [1]*len(ids)
    pad = MAX_LEN - len(ids)
    if pad>0:
        ids += [ce_tokenizer.pad_token_id]*pad
        seg += [0]*pad
        attn += [0]*pad
    return {
        "input_ids": torch.tensor(ids, dtype=torch.long),
        "token_type_ids": torch.tensor(seg, dtype=torch.long),
        "attention_mask": torch.tensor(attn, dtype=torch.long),
    }


# =================== Triplet Dataset (dùng PartitionedMetadataLoader) ===================
class TripletDataset(Dataset):
    def __init__(self, triplet_df: pd.DataFrame, meta_loader: PartitionedMetadataLoader):
        self.df = triplet_df.reset_index(drop=True)
        self.meta = meta_loader

    def __len__(self): return len(self.df)

    def _fetch_many(self, ids: Iterable[int]) -> pd.DataFrame:
        return self.meta.get_many(ids).set_index("corpusid")

    def __getitem__(self, i):
        r = self.df.iloc[i]
        citing_id = int(r["citing_id"])
        cited_id  = int(r["cited_id"])

        # ---- PATCH: xử lý non_cited_id an toàn ----
        val = r["non_cited_id"]
        if isinstance(val, (list, tuple)):
            seq = list(val)
        elif isinstance(val, np.ndarray):
            seq = val.tolist()
        elif pd.isna(val):
            seq = []
        else:
            seq = [val]
        neg_ids = [int(x) for x in seq][:8]
        # -------------------------------------------

        need_ids = [citing_id, cited_id] + neg_ids
        meta_df  = self._fetch_many(need_ids)

        def safe_get(pid: int):
            if pid not in meta_df.index:
                return {"title":"","abstract":"","authors_concat":"","venue":"","year":np.nan}
            row = meta_df.loc[pid]
            return {
                "title": str(row.get("title","") or ""),
                "abstract": str(row.get("abstract","") or ""),
                "authors_concat": str(row.get("authors_concat","") or ""),
                "venue": str(row.get("venue","") or ""),
                "year": row.get("year", np.nan),
            }

        qmeta = safe_get(citing_id)
        pmeta = safe_get(cited_id)
        nmeta = [safe_get(n) for n in neg_ids]

        sample = {
            "citing_id": citing_id,
            "cited_id":  cited_id,
            "neg_ids":   neg_ids,
            "q_title":   qmeta["title"],
            "q_abstract":qmeta["abstract"],
            "q_context": str(r.get("context","") or ""),
            "q_authors": qmeta["authors_concat"],
            "pos_title":    pmeta["title"],
            "pos_abstract": pmeta["abstract"],
            "pos_authors":  pmeta["authors_concat"],
            "pos_venue":    pmeta["venue"],
            "neg_docs": nmeta,
        }
        return sample


class CollatedBatch:
    def __init__(self, input_ids, token_type_ids, attention_mask, group_index, pos_index, neg_counts):
        self.input_ids = input_ids
        self.token_type_ids = token_type_ids
        self.attention_mask = attention_mask
        self.group_index = group_index
        self.pos_index = pos_index
        self.neg_counts = neg_counts

def collate_triplet(batch: List[Dict[str,Any]]) -> CollatedBatch:
    pairs = []
    group_idx = []; pos_idx = []; neg_cnts = []
    cursor = 0
    for ex in batch:
        query = {
            "q_title": ex["q_title"],
            "q_abstract": ex["q_abstract"],
            "q_context": ex["q_context"],
            "q_authors": ex["q_authors"],
        }
        # positive
        docp = {"d_title": ex["pos_title"], "d_abstract": ex["pos_abstract"],
                "d_authors": ex["pos_authors"], "d_venue": ex["pos_venue"]}
        tpos = _pack_and_truncate({**query, **docp}); pairs.append(tpos)

        # negatives
        K = len(ex["neg_docs"])
        for d in ex["neg_docs"]:
            docn = {"d_title": d["title"], "d_abstract": d["abstract"],
                    "d_authors": d["authors_concat"], "d_venue": d["venue"]}
            tneg = _pack_and_truncate({**query, **docn}); pairs.append(tneg)

        group_idx.append(cursor); pos_idx.append(cursor); neg_cnts.append(K)
        cursor += (1+K)

    ids = torch.stack([p["input_ids"] for p in pairs], 0)
    tti = torch.stack([p["token_type_ids"] for p in pairs], 0)
    att = torch.stack([p["attention_mask"] for p in pairs], 0)
    return CollatedBatch(ids, tti, att,
                         torch.tensor(group_idx, dtype=torch.long),
                         torch.tensor(pos_idx,   dtype=torch.long),
                         torch.tensor(neg_cnts,  dtype=torch.long))


# =================== Split by citing year ===================
def attach_citing_year(df_triplet: pd.DataFrame, meta_loader: PartitionedMetadataLoader) -> pd.DataFrame:
    yrs = meta_loader.get_many(df_triplet["citing_id"].astype("int64")).loc[:, ["corpusid","year"]]
    yrs = yrs.rename(columns={"corpusid":"citing_id","year":"citing_year"})
    out = df_triplet.merge(yrs, on="citing_id", how="left")
    return out

def split_by_year(df: pd.DataFrame):
    train = df[(df["citing_year"].fillna(0) < 2025)]
    val  = df[(df["citing_year"] == 2025)]
    test   = df[(df["citing_year"] == 2026)]
    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)


# =================== Build CE model + LoRA ===================
def build_ce_model():
    config = AutoConfig.from_pretrained(CE_MODEL_NAME, num_labels=1)
    base = BertForSequenceClassification.from_pretrained(CE_MODEL_NAME, config=config)
    base.gradient_checkpointing_enable()
    lcfg = LoraConfig(
        r=LORA_CFG["r"], lora_alpha=LORA_CFG["lora_alpha"], lora_dropout=LORA_CFG["lora_dropout"],
        bias=LORA_CFG["bias"], target_modules=LORA_CFG["target_modules"], task_type="SEQ_CLS"
    )
    model = get_peft_model(base, lcfg)
    model.print_trainable_parameters()
    return model.to(device)


# =================== Triplet loss ===================
margin = 0.3
mrloss = nn.MarginRankingLoss(margin=margin, reduction='mean')

def triplet_loss_from_logits(logits: torch.Tensor,
                             group_index: torch.Tensor,
                             pos_index: torch.Tensor,
                             neg_counts: torch.Tensor) -> torch.Tensor:
    losses = []
    y = torch.ones(1, device=logits.device, dtype=logits.dtype)
    B = group_index.size(0)
    for i in range(B):
        start = group_index[i].item()
        pidx  = pos_index[i].item()
        kneg  = neg_counts[i].item()
        if kneg == 0: continue
        s_pos = logits[pidx]
        s_negs = logits[(start+1):(start+1+kneg)]
        loss = mrloss(s_pos.expand_as(s_negs), s_negs, y.expand_as(s_negs))
        losses.append(loss)
    if not losses:
        return torch.tensor(0.0, device=logits.device, requires_grad=True)
    return torch.stack(losses).mean()


# =================== Optim & Sched ===================
def build_optim_sched(model, train_steps: int):
    optim = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=LR, betas=BETAS, eps=ADAM_EPS, weight_decay=WEIGHT_DECAY
    )
    warmup = int(WARMUP_RATIO * train_steps)
    sched  = get_cosine_schedule_with_warmup(optim, num_warmup_steps=warmup, num_training_steps=train_steps)
    scaler = torch.amp.GradScaler('cuda', enabled=FP16)
    return optim, sched, scaler


# =================== SPECTER2 + FAISS + BM25 (for validation MRR) ===================
from adapters import AutoAdapterModel  # AdapterHub
from transformers import AutoTokenizer as HFTokenizer

def l2norm_vecs(x: np.ndarray) -> np.ndarray:
    if x.ndim == 1: return (x / (np.linalg.norm(x)+1e-12)).astype(np.float32)
    return (x / (np.linalg.norm(x, axis=1, keepdims=True)+1e-12)).astype(np.float32)

@torch.no_grad()
def s2_encode(texts: List[str], model: AutoAdapterModel, tokenizer: HFTokenizer,
              batch_size=512, max_len=512) -> np.ndarray:
    out = []
    for i in range(0, len(texts), batch_size):
        enc = tokenizer(texts[i:i+batch_size], max_length=max_len, truncation=True,
                        padding=True, return_tensors="pt")
        enc = {k: v.to(device) for k,v in enc.items()}
        with torch.amp.autocast('cuda', enabled=FP16):
            hidden = model(**enc).last_hidden_state
            v = hidden[:,0,:].detach().cpu().float().numpy()  # CLS pooling
            out.append(l2norm_vecs(v))
    return np.vstack(out)

def build_retriever_objects():
    # SPECTER2 query encoders
    tok = HFTokenizer.from_pretrained(S2_BASE)
    m_ta = AutoAdapterModel.from_pretrained(S2_BASE);   m_ta.load_adapter(S2_ADAPTER_TA, source="hf", load_as="a_ta", set_active=True);   m_ta.eval().to(device)
    m_ah = AutoAdapterModel.from_pretrained(S2_BASE);   m_ah.load_adapter(S2_ADAPTER_ADHQ, source="hf", load_as="a_ah", set_active=True); m_ah.eval().to(device)

    # FAISS
    index = faiss.read_index(FAISS_INDEX_PATH.as_posix())
    try:
        index.nprobe = min(getattr(index, "nlist", 128), 128)
    except Exception:
        pass

    # BM25 (pyserini)
    try:
        from pyserini.search.lucene import LuceneSearcher
        bm25 = LuceneSearcher(BM25_INDEX_DIR.as_posix())
        # default BM25 params ok; tune nếu cần
    except Exception as e:
        print("⚠️ BM25 SimpleSearcher load failed:", e)
        bm25 = None

    return tok, m_ta, m_ah, index, bm25

def zscore(x: np.ndarray) -> np.ndarray:
    if x.size == 0: return x
    mu, sd = x.mean(), x.std()
    return np.zeros_like(x) if sd==0 else (x-mu)/sd

def faiss_search_batch(index, Q: np.ndarray, topk: int) -> Tuple[np.ndarray, np.ndarray]:
    D, I = index.search(Q.astype(np.float32), topk)
    return I.astype(np.int64), D.astype(np.float32)

def hybrid_candidates(query_rows: List[Dict[str,Any]],
                      tok, m_ta, m_ah, faiss_index, bm25, topk=TOPK_RETRIEVER, alpha=ALPHA_HYBRID):
    # Build query strings
    texts_ta, texts_ctx, texts_bm = [], [], []
    for q in query_rows:
        title, abstract, context = q["title"], q["abstract"], q["context"]
        ta = title if title else ""
        if (abstract or "").strip():
            ta = f"{ta} {tok.sep_token} {abstract.strip()}" if ta else abstract.strip()
        texts_ta.append(ta if ta else "")
        texts_ctx.append(context if context else (title or ""))
        texts_bm.append(context if context else (title or ""))

    Q_ta  = s2_encode(texts_ta,  m_ta, tok)
    Q_ctx = s2_encode(texts_ctx, m_ah, tok)
    Q_early = l2norm_vecs(W_TA * Q_ta + W_CTX * Q_ctx)

    I_faiss, S_faiss = faiss_search_batch(faiss_index, Q_early, topk)

    # BM25
    N = len(query_rows)
    I_bm = np.zeros((N, topk), dtype=np.int64)
    S_bm = np.zeros((N, topk), dtype=np.float32)
    if bm25 is not None:
        for i, qtext in enumerate(texts_bm):
            hits = bm25.search(qtext, topk)
            if not hits: continue
            docids = [int(h.docid) for h in hits]
            scores = [float(h.score) for h in hits]
            k = min(topk, len(docids))
            I_bm[i,:k] = np.asarray(docids[:k], dtype=np.int64)
            S_bm[i,:k] = np.asarray(scores[:k], dtype=np.float32)

    # Hybrid blend per row
    I_hybrid = np.zeros_like(I_faiss)
    for i in range(N):
        cand_ids = set(I_faiss[i].tolist()) | set(I_bm[i].tolist())
        if -1 in cand_ids: cand_ids.discard(-1)
        cand_ids = list(cand_ids)
        sf, sb = [], []
        for cid in cand_ids:
            f_mask = (I_faiss[i]==cid)
            s_f = S_faiss[i][f_mask][0] if f_mask.any() else 0.0
            b_mask = (I_bm[i]==cid)
            s_b = S_bm[i][b_mask][0] if b_mask.any() else 0.0
            sf.append(s_f); sb.append(s_b)
        sf = np.asarray(sf, dtype=np.float32)
        sb = np.asarray(sb, dtype=np.float32)
        s = alpha * zscore(sf) + (1.0-alpha) * zscore(sb)
        order = np.argsort(-s)
        I_hybrid[i,:] = np.asarray([cand_ids[j] for j in order[:topk]], dtype=np.int64)

    return I_hybrid  # shape: (N, topk)


# =================== Build val queries & compute MRR ===================
def compute_mrr(val_df: pd.DataFrame, meta_loader: PartitionedMetadataLoader,
                tok, m_ta, m_ah, faiss_index, bm25) -> float:
    """
    Cho mỗi citing trong val:
      - Query fields: title, abstract, context (từ citing paper)
      - Hybrid top-100, bỏ citing_id rồi tính RR theo cited_id
    """
    # Chuẩn bị query rows
    citing_ids = val_df["citing_id"].astype("int64").tolist()
    cited_ids  = val_df["cited_id"].astype("int64").tolist()
    contexts   = val_df["context"].fillna("").astype(str).tolist()

    # Lấy title/abstract của citing
    cmeta = meta_loader.get_many(citing_ids).set_index("corpusid")
    queries = []
    for i, cid in enumerate(citing_ids):
        row = cmeta.loc[cid] if cid in cmeta.index else None
        title = str(row["title"]) if row is not None and pd.notna(row["title"]) else ""
        abstract = str(row["abstract"]) if row is not None and pd.notna(row["abstract"]) else ""
        queries.append({"title": title, "abstract": abstract, "context": contexts[i]})

    I_h = hybrid_candidates(queries, tok, m_ta, m_ah, faiss_index, bm25, topk=TOPK_RETRIEVER, alpha=ALPHA_HYBRID)

    # Tính RR (loại citing_id)
    rrs = []
    for i in range(len(queries)):
        cand = I_h[i]
        citing_id = citing_ids[i]
        cand = cand[cand != citing_id]
        # tìm rank của cited_id
        target = cited_ids[i]
        pos = np.where(cand == target)[0]
        if len(pos)==0:
            rrs.append(0.0)
        else:
            rrs.append(1.0 / (pos[0] + 1))
    return float(np.mean(rrs))


# =================== Training loop (bật/tắt gradient đúng chuẩn) ===================
def train_one_epoch(loader, model, optim, sched, scaler) -> float:
    model.train(True)
    epoch_loss, n_steps = 0.0, 0

    optim.zero_grad(set_to_none=True)
    prev = torch.is_grad_enabled()
    torch.set_grad_enabled(True)
    try:
        for batch in tqdm(loader, desc="Train"):
            input_ids      = batch.input_ids.to(device, non_blocking=True)
            token_type_ids = batch.token_type_ids.to(device, non_blocking=True)
            attention_mask = batch.attention_mask.to(device, non_blocking=True)

            with torch.amp.autocast('cuda', enabled=FP16):
                outputs = model(input_ids=input_ids,
                                token_type_ids=token_type_ids,
                                attention_mask=attention_mask,
                                return_dict=True)
                logits = outputs.logits.squeeze(-1)
                loss = triplet_loss_from_logits(
                    logits, batch.group_index.to(device),
                    batch.pos_index.to(device), batch.neg_counts.to(device)
                )
            # grad accumulation
            loss_to_backprop = loss / GRAD_ACCUM_STEPS
            scaler.scale(loss_to_backprop).backward()

            if (n_steps + 1) % GRAD_ACCUM_STEPS == 0:
                scaler.unscale_(optim)
                torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
                scaler.step(optim)
                scaler.update()
                optim.zero_grad(set_to_none=True)
                sched.step()

            epoch_loss += loss.item()
            n_steps += 1

        # leftover
        if n_steps % GRAD_ACCUM_STEPS != 0:
            scaler.unscale_(optim)
            torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
            scaler.step(optim)
            scaler.update()
            optim.zero_grad(set_to_none=True)
            sched.step()
    finally:
        torch.set_grad_enabled(prev)

    return epoch_loss / max(1, n_steps)

@torch.no_grad()
def evaluate_one_epoch(loader, model) -> float:
    model.train(False)
    epoch_loss, n_steps = 0.0, 0

    prev = torch.is_grad_enabled()
    torch.set_grad_enabled(False)
    try:
        for batch in tqdm(loader, desc="Val"):
            input_ids      = batch.input_ids.to(device, non_blocking=True)
            token_type_ids = batch.token_type_ids.to(device, non_blocking=True)
            attention_mask = batch.attention_mask.to(device, non_blocking=True)

            with torch.amp.autocast('cuda', enabled=FP16):
                outputs = model(input_ids=input_ids,
                                token_type_ids=token_type_ids,
                                attention_mask=attention_mask,
                                return_dict=True)
                logits = outputs.logits.squeeze(-1)
                loss = triplet_loss_from_logits(
                    logits, batch.group_index.to(device),
                    batch.pos_index.to(device), batch.neg_counts.to(device)
                )
            epoch_loss += loss.item()
            n_steps += 1
    finally:
        torch.set_grad_enabled(prev)

    return epoch_loss / max(1, n_steps)


# =================== Main ===================
def main():
    # Load triplet df
    triplet_df = pd.read_parquet(TRIPLET_PATH)
    # Metadata loader
    meta_loader = PartitionedMetadataLoader(PAPERS_ROOT)

    # Attach citing_year & split
    triplet_df = attach_citing_year(triplet_df, meta_loader)
    train_df, val_df, test_df = split_by_year(triplet_df)
    print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

    # Datasets & loaders
    train_ds = TripletDataset(train_df, meta_loader)
    val_ds   = TripletDataset(val_df,   meta_loader)

    train_loader = DataLoader(train_ds, batch_size=PER_DEVICE_TRAIN_BS, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY,
                              persistent_workers=True, collate_fn=collate_triplet, prefetch_factor=2)
    val_loader   = DataLoader(val_ds, batch_size=PER_DEVICE_TRAIN_BS, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY,
                              persistent_workers=True, collate_fn=collate_triplet, prefetch_factor=2)

    # Build CE model + optim/sched
    model = build_ce_model()
    steps_per_epoch = math.ceil(len(train_loader) / GRAD_ACCUM_STEPS)
    total_steps = steps_per_epoch * NUM_EPOCHS
    optim, sched, scaler = build_optim_sched(model, total_steps)

    # Build retriever objects for val MRR
    s2_tok, s2_ta, s2_adhq, faiss_index, bm25 = build_retriever_objects()

    best_mrr = -1.0
    for epoch in range(1, NUM_EPOCHS+1):
        print(f"\n=== Epoch {epoch}/{NUM_EPOCHS} ===")
        tr_loss = train_one_epoch(train_loader, model, optim, sched, scaler)
        vl_loss = evaluate_one_epoch(val_loader, model)
        print(f"Train loss: {tr_loss:.4f} | Val loss: {vl_loss:.4f}")

        # ====== Validation by Hybrid MRR ======
        # Chuẩn bị val queries: title, abstract, context từ citing
        # context đã có trong val_df; title/abstract lấy bằng meta_loader
        # Để giảm thời gian, có thể limit số val (ví dụ 2k), ở đây dùng toàn bộ
        val_mrr = compute_mrr(val_df, meta_loader, s2_tok, s2_ta, s2_adhq, faiss_index, bm25)
        print(f"Val MRR (Hybrid BM25 + S2-FAISS early, top{TOPK_RETRIEVER}, α={ALPHA_HYBRID}): {val_mrr:.5f}")

        # Save best by MRR
        if val_mrr > best_mrr:
            best_mrr = val_mrr
            torch.save(model.state_dict(), BEST_CKPT)
            print("✅ Saved best checkpoint by MRR:", BEST_CKPT)

    print("Training done. Best Val MRR:", best_mrr)


if __name__ == "__main__":
    main()
