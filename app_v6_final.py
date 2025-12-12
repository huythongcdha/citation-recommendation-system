# app.py
# -*- coding: utf-8 -*-
import os
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
import faiss
import torch
import re
from html import escape
import pyarrow.parquet as pq

# -------- Optional fuzzy matching --------
try:
    from rapidfuzz import fuzz
    _HAS_RAPIDFUZZ = True
except Exception:
    _HAS_RAPIDFUZZ = False

# ==================== CONFIG ====================
FAISS_INDEX_PATH    = Path(r"E:\Citation Recommendation\data\indexes\ivf_flat_ip_11m.faiss")
PAPERS_META_ROOT    = Path(r"E:\Citation Recommendation\data\stage\papers_with_abstracts_cleaned_with_url")
RERANKER_CKPT_PATH  = Path(r"E:\Citation Recommendation\models\pubmedbert_lora_triplet_hybrid\best_100k_by_MRR.pt")

BM25_INDEX_DIR      = Path(r"E:\Citation Recommendation\data\bm25\lucene-index-papers_11m")

# SPECTER2 retriever
S2_BASE      = "allenai/specter2_base"
ADAPTER_TA   = "allenai/specter2"              # doc-style (title+abstract)
ADAPTER_ADHQ = "allenai/specter2_adhoc_query"  # ad-hoc (context)
S2_MAX_LEN   = 512

TOPK_SHOW    = 10
TOPK_MAX_UI  = 25      # maximum display candidates
TOPK_POOL    = 25     # candidate pool from retriever

# Hybrid z-score fuse: s = α*z(FAISS) + (1-α)*z(BM25)
ALPHA_DEFAULT = 0.5

# Reranker (PubMedBERT CE + LoRA)
MODEL_NAME   = "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext"
MAX_LEN      = 512
FP16         = True

# Truncate BUDGET
BUDGET = {
    "query":    {"title": 30, "abstract": 150, "context": 80,  "authors": 20},
    "document": {"title": 30, "abstract": 160, "authors": 20, "venue": 10},
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
    "query":    {"title":"Title: ", "abstract":"Abstract: ", "context":"Context: ", "authors":"Authors: "},
    "document": {"title":"Title: ", "abstract":"Abstract: ", "authors":"Authors: ", "venue":"Venue: "},
}

# ==================== UI BASE ====================
st.set_page_config(page_title="Citation Recommendation", page_icon="📚", layout="wide")
st.markdown(
    "<h1 style='font-size:1.9rem; margin-bottom:0.2rem;'>📚 Citation Recommendation</h1>"
    "<p style='color:#666; font-size:0.95rem;'>Hybrid Retriever (SPECTER2+FAISS / BM25) → PubMedBERT+LoRA Reranker with token-overlap explainability & RIS export.</p>",
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown("<h3 style='margin-bottom:0.3rem;'>Navigation</h3>", unsafe_allow_html=True)
    page = st.radio(" ", ["Search & Recommend", "Download history"], index=0, label_visibility="collapsed")
    st.markdown("---")

# ==================== Partitioned Metadata Loader ====================
class PartitionedMetadataLoader:
    """
    Metadata phân vùng theo part=00..ff dựa vào corpusid & 255.
    root/
      part=00/*.parquet
      ...
      part=ff/*.parquet
    """
    def __init__(self, root: Path, required_cols: Optional[List[str]] = None, cache_limit: int = 16):
        self.root = Path(root)
        self.required_cols = required_cols or ["corpusid","title","abstract","authors_concat","venue","year","url"]
        self._part_cache: dict[str, pd.DataFrame] = {}
        self._path_cache: dict[str, Optional[Path]] = {}
        self._lru_keys: List[str] = []
        self._cache_limit = cache_limit

    @staticmethod
    def _part_hex(corpusid: int) -> str:
        return f"{corpusid & 255:02x}"

    def _evict_if_needed(self):
        if len(self._part_cache) <= self._cache_limit:
            return
        key = self._lru_keys.pop(0)
        self._part_cache.pop(key, None)

    def _touch_lru(self, key: str):
        if key in self._lru_keys:
            self._lru_keys.remove(key)
        self._lru_keys.append(key)

    def _find_part_file(self, part_hex: str) -> Optional[Path]:
        if part_hex in self._path_cache:
            return self._path_cache[part_hex]
        part_dir = self.root / f"part={part_hex}"
        if not part_dir.exists():
            self._path_cache[part_hex] = None
            return None
        files = list(part_dir.glob("*.parquet"))
        p = files[0] if files else None
        self._path_cache[part_hex] = p
        return p

    def _load_part(self, part_hex: str) -> pd.DataFrame:
        if part_hex in self._part_cache:
            self._touch_lru(part_hex)
            return self._part_cache[part_hex]

        p = self._find_part_file(part_hex)
        if p is None:
            df = pd.DataFrame(columns=self.required_cols)
        else:
            schema = pq.read_schema(p)
            available_cols = set(schema.names)
            cols = [c for c in self.required_cols if c in available_cols]
            if cols:
                df = pd.read_parquet(p, columns=cols)
            else:
                df = pd.DataFrame(columns=self.required_cols)

        if "corpusid" in df.columns:
            df["corpusid"] = df["corpusid"].astype("int64", errors="ignore")
            df = df.set_index("corpusid", drop=False)

        self._part_cache[part_hex] = df
        self._touch_lru(part_hex)
        self._evict_if_needed()
        return df

    def get_by_id(self, corpusid: int) -> Optional[pd.Series]:
        part_hex = self._part_hex(int(corpusid))
        df = self._load_part(part_hex)
        if int(corpusid) in df.index:
            return df.loc[int(corpusid)]
        return None

    def get_many(self, corpusids: List[int]) -> pd.DataFrame:
        groups: Dict[str, List[int]] = {}
        for cid in corpusids:
            part_hex = self._part_hex(int(cid))
            groups.setdefault(part_hex, []).append(int(cid))
        frames = []
        for part_hex, ids in groups.items():
            df = self._load_part(part_hex)
            sub = df.loc[df.index.intersection(ids)]
            frames.append(sub)
        if frames:
            out = pd.concat(frames, axis=0)
        else:
            out = pd.DataFrame(columns=self.required_cols)
        return out

# ==================== Cached loaders ====================
@st.cache_resource(show_spinner=True)
def build_meta_loader(root: Path) -> PartitionedMetadataLoader:
    return PartitionedMetadataLoader(
        root,
        required_cols=["corpusid","title","abstract","authors_concat","venue","year","url"]
    )

@st.cache_resource(show_spinner=True)
def load_faiss_index(index_path: Path):
    faiss_index = faiss.read_index(index_path.as_posix())
    try:
        faiss_index.nprobe = min(getattr(faiss_index, "nlist", 128), 128)
    except Exception:
        pass
    return faiss_index

@st.cache_resource(show_spinner=True)
def load_specter2_dual():
    from adapters import AutoAdapterModel
    from transformers import AutoTokenizer as HFTokenizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = HFTokenizer.from_pretrained(S2_BASE)
    m_ta = AutoAdapterModel.from_pretrained(S2_BASE)
    m_ta.load_adapter(ADAPTER_TA, source="hf", load_as="a_ta", set_active=True)
    m_ta.eval().to(device)
    m_adhq = AutoAdapterModel.from_pretrained(S2_BASE)
    m_adhq.load_adapter(ADAPTER_ADHQ, source="hf", load_as="a_adhq", set_active=True)
    m_adhq.eval().to(device)
    return tok, m_ta, m_adhq, device

@st.cache_resource(show_spinner=True)
def load_bm25_searcher(index_dir: Path):
    from pyserini.search.lucene import LuceneSearcher
    searcher = LuceneSearcher(index_dir.as_posix())
    searcher.set_bm25(k1=1.2, b=0.75)
    return searcher

def load_reranker_model():
    from transformers import AutoConfig, BertForSequenceClassification, AutoTokenizer
    from peft import LoraConfig, get_peft_model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = AutoConfig.from_pretrained(MODEL_NAME, num_labels=1)
    base = BertForSequenceClassification.from_pretrained(MODEL_NAME, config=config)
    base.gradient_checkpointing_enable()
    lora_cfg = LoraConfig(
        r=8, lora_alpha=16, target_modules=["query","key","value","output.dense"],
        lora_dropout=0.1, bias="none", task_type="SEQ_CLS",
    )
    model = get_peft_model(base, lora_cfg).to(device)
    state = torch.load(RERANKER_CKPT_PATH.as_posix(), map_location=device)
    model.load_state_dict(state, strict=False)
    tok = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    model.eval()
    return model, tok, device

@st.cache_resource(show_spinner=False)
def _build_rer_model_and_tok():
    return load_reranker_model()

# ==================== Retriever utils ====================
def _l2norm(v: np.ndarray) -> np.ndarray:
    return v / (np.linalg.norm(v) + 1e-12)

@st.cache_resource(show_spinner=False)
def _build_rer_tokenizer():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    assert tok.sep_token_id is not None and tok.cls_token_id is not None
    return tok

_rer_tok = _build_rer_tokenizer()

def _encode_text(text: str) -> List[int]:
    return _rer_tok.encode(text, add_special_tokens=False, truncation=False)

def _budgeted_tokens(piece: str, side: str, field: str) -> List[int]:
    prefix = SPECIAL_PREFIXES[side][field]
    return _encode_text(prefix + (piece or ""))

def _pack_and_truncate(example: Dict[str, str]) -> Dict[str, torch.Tensor]:
    parts = {
        ("query","title"):    _budgeted_tokens(example["q_title"],    "query","title"),
        ("query","abstract"): _budgeted_tokens(example["q_abstract"], "query","abstract"),
        ("query","context"):  _budgeted_tokens(example["q_context"],  "query","context"),
        ("query","authors"):  _budgeted_tokens(example["q_authors"],  "query","authors"),
        ("document","title"):    _budgeted_tokens(example["d_title"],    "document","title"),
        ("document","abstract"): _budgeted_tokens(example["d_abstract"], "document","abstract"),
        ("document","authors"):  _budgeted_tokens(example["d_authors"],  "document","authors"),
        ("document","venue"):    _budgeted_tokens(example["d_venue"],    "document","venue"),
    }
    def assemble():
        ids = [_rer_tok.cls_token_id]; seg = [0]
        for key in [("query","title"),("query","abstract"),("query","context"),("query","authors")]:
            ids += parts[key]; seg += [0]*len(parts[key])
            ids += [_rer_tok.sep_token_id]; seg += [0]
        for key in [("document","title"),("document","abstract"),("document","authors"),("document","venue")]:
            ids += parts[key]; seg += [1]*len(parts[key])
            ids += [_rer_tok.sep_token_id]; seg += [1]
        return ids, seg
    def clip_to_budget(side, field):
        parts[(side,field)] = parts[(side,field)][:BUDGET[side][field]]
    ids, seg = assemble()
    if len(ids) > MAX_LEN:
        for side, field in TRUNC_ORDER:
            ids, seg = assemble()
            if len(ids) <= MAX_LEN:
                break
            clip_to_budget(side, field)
        ids, seg = assemble()
        def step_trim(side, field, step=4):
            cur = parts[(side,field)]
            parts[(side,field)] = cur[:-step] if len(cur) > step else cur[:1]
        while len(ids) > MAX_LEN:
            for side, field in TRUNC_ORDER:
                if len(ids) <= MAX_LEN:
                    break
                step_trim(side, field, 4)
                ids, seg = assemble()
    attn = [1]*len(ids)
    pad = MAX_LEN - len(ids)
    if pad > 0:
        ids += [_rer_tok.pad_token_id]*pad
        seg += [0]*pad
        attn += [0]*pad
    return {
        "input_ids": torch.tensor(ids, dtype=torch.long),
        "token_type_ids": torch.tensor(seg, dtype=torch.long),
        "attention_mask": torch.tensor(attn, dtype=torch.long),
    }

@torch.no_grad()
def s2_encode(texts: List[str], model_s2, tokenizer_s2, device, max_len=S2_MAX_LEN, bs=256) -> np.ndarray:
    outs = []
    for i in range(0, len(texts), bs):
        batch = texts[i:i+bs]
        enc = tokenizer_s2(batch, truncation=True, padding=True, max_length=max_len, return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items()}
        last = model_s2(**enc).last_hidden_state
        cls = last[:, 0, :].detach().cpu().float().numpy()
        cls /= (np.linalg.norm(cls, axis=1, keepdims=True) + 1e-12)
        outs.append(cls.astype(np.float32))
    return np.vstack(outs)

def encode_query_earlyfusion(title: str, abstract: str, context: str,
                             s2_tok, m_ta, m_adhq, device,
                             w_ta: float, w_ctx: float) -> np.ndarray:
    ta_text = (title or "").strip()
    if (abstract or "").strip():
        ta_text = (ta_text + " " + s2_tok.sep_token + " " + abstract.strip()) if ta_text else abstract.strip()
    q_ta  = s2_encode([ta_text], m_ta,   s2_tok, device)[0]
    q_ctx = s2_encode([context or (title or "")], m_adhq, s2_tok, device)[0]
    s = max(w_ta + w_ctx, 1e-8)
    w_ta_n = w_ta / s
    w_ctx_n = w_ctx / s
    q = w_ta_n * q_ta + w_ctx_n * q_ctx
    return _l2norm(q).astype(np.float32)

def faiss_search(index, q: np.ndarray, topk: int) -> Tuple[np.ndarray, np.ndarray]:
    D, I = index.search(q.reshape(1,-1), topk)
    return I[0].astype(np.int64), D[0].astype(np.float32)

def zscore(x: np.ndarray) -> np.ndarray:
    if x.size == 0:
        return x
    mu = x.mean()
    sd = x.std()
    return np.zeros_like(x) if sd == 0 else (x - mu) / sd

def bm25_search_ids_scores(searcher, text: str, topk: int) -> Tuple[np.ndarray, np.ndarray]:
    hits = searcher.search(text, topk)
    if not hits:
        return np.empty((0,), dtype=np.int64), np.empty((0,), dtype=np.float32)
    ids = np.array([int(h.docid) for h in hits], dtype=np.int64)
    scores = np.array([float(h.score) for h in hits], dtype=np.float32)
    return ids, scores

def hybrid_fuse_ids_scores(faiss_ids: np.ndarray, faiss_scores: np.ndarray,
                           bm25_ids: np.ndarray, bm25_scores: np.ndarray,
                           topk: int, alpha: float) -> Tuple[np.ndarray, np.ndarray]:
    cand = set(faiss_ids.tolist()) | set(bm25_ids.tolist())
    if -1 in cand:
        cand.discard(-1)
    cand = list(cand)
    if not cand:
        return faiss_ids[:0], faiss_scores[:0]
    s_f = np.zeros(len(cand), dtype=np.float32)
    s_b = np.zeros(len(cand), dtype=np.float32)
    for i, pid in enumerate(cand):
        m = (faiss_ids == pid)
        s_f[i] = faiss_scores[m][0] if m.any() else 0.0
        m2 = (bm25_ids == pid)
        s_b[i] = bm25_scores[m2][0] if m2.any() else 0.0
    sf = zscore(s_f)
    sb = zscore(s_b)
    s = alpha * sf + (1.0 - alpha) * sb
    order = np.argsort(-s)
    top = order[:topk]
    return np.array([cand[j] for j in top], dtype=np.int64), s[top].astype(np.float32)

# ==================== Reranker ====================
@torch.no_grad()
def reranker_score_pairs(model, pairs: List[Dict[str,str]], device, bs=256) -> np.ndarray:
    scores = []
    for i in range(0, len(pairs), bs):
        chunk = pairs[i:i+bs]
        packed = [_pack_and_truncate(p) for p in chunk]
        input_ids      = torch.stack([x["input_ids"] for x in packed]).to(device, non_blocking=True)
        token_type_ids = torch.stack([x["token_type_ids"] for x in packed]).to(device, non_blocking=True)
        attention_mask = torch.stack([x["attention_mask"] for x in packed]).to(device, non_blocking=True)
        with torch.amp.autocast('cuda', enabled=FP16):
            out = model(input_ids=input_ids,
                        token_type_ids=token_type_ids,
                        attention_mask=attention_mask,
                        return_dict=True)
            logits = out.logits.squeeze(-1).detach().float().cpu().numpy()
            probs = 1 / (1 + np.exp(-logits))  # sigmoid
        scores.append(probs)
    return np.concatenate(scores, axis=0)

# ==================== Explainability: token overlap (precomputed) ====================
_WORD_RE = re.compile(r"[A-Za-z0-9]+(?:[-_][A-Za-z0-9]+)*")

STOPWORDS = {
    "a","an","the","this","that","these","those",
    "in","on","at","to","from","for","of","by","with","into","onto","about",
    "and","or","but","so","yet",
    "is","are","was","were","be","been","being",
    "am","do","does","did","have","has","had",
    "i","you","we","they","he","she","it",
    "me","him","her","us","them","my","your","our","their",
    "can","could","should","would","may","might",
    "will","shall","also","just","only","than","then","when","where",
}

def is_content_token(t: str) -> bool:
    if not t:
        return False
    t = t.lower()
    if len(t) < 3:
        return False
    if t in STOPWORDS:
        return False
    if all(ch.isdigit() for ch in t):
        return False
    return True

def _norm_tokens(s: str) -> List[str]:
    if not s:
        return []
    toks = _WORD_RE.findall(s)
    return [t.lower() for t in toks if is_content_token(t)]

def _split_authors(s: str) -> List[str]:
    if not s:
        return []
    parts = re.split(r"[;,]| and ", s)
    parts = [p.strip() for p in parts if p.strip()]
    tokens = []
    for p in parts:
        full = p.lower()
        if is_content_token(full):
            tokens.append(full)
        last = p.split()[-1].lower()
        if is_content_token(last):
            tokens.append(last)
    return list(set(tokens))

def _build_overlap_sets(query_parts: dict, doc_parts: dict, fuzzy: bool = True) -> dict:
    q_tokens = {
        "title":    _norm_tokens(query_parts.get("title","")),
        "abstract": _norm_tokens(query_parts.get("abstract","")),
        "context":  _norm_tokens(query_parts.get("context","")),
        "authors":  _split_authors(query_parts.get("authors","")),
    }
    d_tokens = {
        "title":    _norm_tokens(doc_parts.get("title","")),
        "abstract": _norm_tokens(doc_parts.get("abstract","")),
        "authors":  _split_authors(doc_parts.get("authors","")),
        "venue":    _norm_tokens(doc_parts.get("venue","")),
    }

    def fuzzy_hit(tok, pool, th=85):
        if not _HAS_RAPIDFUZZ or not fuzzy:
            return tok in pool
        return any(fuzz.ratio(tok, t) >= th for t in pool)

    doc_all = set(d_tokens["title"] + d_tokens["abstract"] + d_tokens["venue"] + d_tokens["authors"])
    q_high = {k: set() for k in q_tokens}
    for k, toks in q_tokens.items():
        for t in toks:
            if fuzzy_hit(t, doc_all):
                q_high[k].add(t)

    q_all = set(q_tokens["title"] + q_tokens["abstract"] + q_tokens["context"] + q_tokens["authors"])
    d_high = {k: set() for k in d_tokens}
    for k, toks in d_tokens.items():
        for t in toks:
            if fuzzy_hit(t, q_all):
                d_high[k].add(t)

    return {"q": q_high, "d": d_high}

def _colorize_text_simple(text: str, hit_set: set[str],
                          palette=("rgba(255,225,0,0.35)", "rgba(255,180,0,0.65)")) -> str:
    if not text:
        return ""
    def span(tok: str) -> str:
        col = palette[1] if len(tok) >= 8 else palette[0]
        return f"<span style='background:{col}; padding:0 2px; border-radius:4px'>{escape(tok)}</span>"
    out = []
    for m in re.finditer(r"\w+|\W+", text):
        s = m.group(0)
        if s.strip() and s[0].isalnum():
            tok = s
            key = tok.lower()
            out.append(span(tok) if key in hit_set else escape(tok))
        else:
            out.append(escape(s))
    return "".join(out)

# ==================== RIS Builder & History ====================
def build_ris_record(title: str, authors: str, year: str, venue: str,
                     abstract: str, url: str) -> str:
    lines = ["TY  - JOUR"]
    if authors:
        parts = re.split(r";| and ", authors)
        if len(parts) == 1:
            comma_parts = [p.strip() for p in authors.split(",") if p.strip()]
            if len(comma_parts) > 1:
                parts = comma_parts
        for a in parts:
            a = a.strip()
            if a:
                lines.append(f"AU  - {a}")
    if title:
        lines.append(f"TI  - {title}")
    if venue:
        lines.append(f"T2  - {venue}")
    if year and str(year).lower() != "nan":
        lines.append(f"PY  - {year}")
    if abstract:
        lines.append(f"AB  - {abstract}")
    if url:
        lines.append(f"UR  - {url}")
    lines.append("ER  - ")
    return "\r\n".join(lines) + "\r\n"

def add_to_download_history(row: pd.Series):
    hist = st.session_state.setdefault("download_history", [])
    hist.append({
        "corpusid": int(row["corpusid"]),
        "title": row["title"],
        "authors": row["authors"],
        "year": str(row["year"]),
        "venue": row["venue"],
        "url": row["url"],
        "abstract": row["abstract"],
        "downloaded_at": datetime.now().isoformat(timespec="seconds"),
    })
    if len(hist) > 500:
        st.session_state["download_history"] = hist[-500:]

# ==================== SEARCH PIPELINE (reusable) ====================
def run_full_search_pipeline(query: Dict[str,str],
                             retr_mode: str,
                             w_ctx: float,
                             alpha: float):
    """
    query: {title, authors, abstract, context}
    retr_mode: "FAISS (SPECTER2 early fusion)" or "Hybrid (FAISS + BM25)"
    """
    in_title   = query.get("title","")
    in_authors = query.get("authors","")
    in_abstract= query.get("abstract","")
    in_local   = query.get("context","")

    # 1) Load heavy resources
    with st.spinner("Loading models, index & metadata loader..."):
        meta_loader = build_meta_loader(PAPERS_META_ROOT)
        faiss_index = load_faiss_index(FAISS_INDEX_PATH)
        s2_tok, s2_ta, s2_adhq, s2_device = load_specter2_dual()
        reranker_model, _, rr_device = _build_rer_model_and_tok()
        if retr_mode == "Hybrid (FAISS + BM25)":
            bm25 = load_bm25_searcher(BM25_INDEX_DIR)
        else:
            bm25 = None

    # 2) Encode query by SPECTER2
    with st.spinner("Encoding query (SPECTER2)..."):
        qvec = encode_query_earlyfusion(
            in_title, in_abstract, in_local,
            s2_tok, s2_ta, s2_adhq, s2_device,
            w_ta=1.0 - w_ctx, w_ctx=w_ctx,
        )

    # 3) FAISS search
    with st.spinner(f"Retrieving FAISS top-{TOPK_POOL}..."):
        faiss_ids, faiss_scores = faiss_search(faiss_index, qvec, topk=TOPK_POOL)

    # 4) BM25 + hybrid fuse
    if retr_mode == "Hybrid (FAISS + BM25)" and bm25 is not None:
        with st.spinner(f"Retrieving BM25 top-{TOPK_POOL}..."):
            qtext = in_local if in_local.strip() else (in_title or "")
            bm25_ids, bm25_scores = bm25_search_ids_scores(bm25, qtext, TOPK_POOL)
        with st.spinner("Fusing FAISS + BM25 (z-score)..."):
            retr_ids, retr_scores = hybrid_fuse_ids_scores(
                faiss_ids, faiss_scores,
                bm25_ids, bm25_scores,
                topk=TOPK_POOL,
                alpha=alpha,
            )
    else:
        retr_ids, retr_scores = faiss_ids, faiss_scores

    # 5) Load metadata for candidates
    with st.spinner("Fetching metadata for candidates..."):
        meta_df = meta_loader.get_many(retr_ids.tolist())
        if "corpusid" in meta_df.columns:
            meta_df = meta_df.set_index("corpusid", drop=False)

    rows = []
    for pid, rscore in zip(retr_ids, retr_scores):
        if pid in meta_df.index:
            r = meta_df.loc[pid]
            rows.append({
                "corpusid": int(pid),
                "title": str(r.get("title","") or ""),
                "authors": str(r.get("authors_concat","") or ""),
                "year": r.get("year",""),
                "venue": str(r.get("venue","") or ""),
                "abstract": str(r.get("abstract","") or ""),
                "url": str(r.get("url","") or ""),
                "retriever_score": float(rscore),
            })
    retr_df = pd.DataFrame(rows)

    # 6) Rerank
    with st.spinner("Re-ranking with PubMedBERT+LoRA..."):
        pairs = []
        for _, row in retr_df.iterrows():
            pairs.append({
                "q_title":    in_title,
                "q_abstract": in_abstract,
                "q_context":  in_local,
                "q_authors":  in_authors,
                "d_title":    row["title"],
                "d_abstract": row["abstract"],
                "d_authors":  row["authors"],
                "d_venue":    row["venue"],
            })
        rr_scores = reranker_score_pairs(reranker_model, pairs, rr_device, bs=256)
        retr_df["reranker_score"] = rr_scores

    # 7) Precompute token-overlap explanation cho tối đa TOPK_MAX_UI
    q_parts = {
        "title": in_title,
        "abstract": in_abstract,
        "context": in_local,
        "authors": in_authors,
    }
    with st.spinner(f"Precomputing token-overlap explanation for top-{TOPK_MAX_UI} candidates..."):
        explain_map: Dict[int, dict] = {}
        top_for_explain = retr_df.sort_values("reranker_score", ascending=False).head(TOPK_MAX_UI)
        for _, row in top_for_explain.iterrows():
            doc_parts = {
                "title": row["title"],
                "abstract": row["abstract"],
                "authors": row["authors"],
                "venue": row["venue"],
            }
            ov = _build_overlap_sets(q_parts, doc_parts, fuzzy=True)
            explain_map[int(row["corpusid"])] = ov

    # 8) Save vào session_state
    st.session_state["retr_df"] = retr_df
    st.session_state["q_parts"] = q_parts
    st.session_state["retr_mode_used"] = retr_mode
    st.session_state["explain_map"] = explain_map

    # Save query + config để auto-update khi đổi config
    st.session_state["query_saved"] = query
    st.session_state["retr_cfg_saved"] = {
        "retr_mode": retr_mode,
        "w_ctx": float(w_ctx),
        "alpha": float(alpha),
    }

# ==================== PAGE 1: SEARCH & RECOMMEND ====================
if page == "Search & Recommend":
    with st.sidebar:
        st.markdown("<h3 style='margin-top:0.3rem;'>Settings</h3>", unsafe_allow_html=True)
        retr_mode = st.radio(
            "Retriever mode",
            ["FAISS (SPECTER2 model)", "Hybrid (FAISS + BM25)"],
            index=1,
            key="retr_mode_radio",
        )
        top_k = st.slider("Top-K results (display)", 5, TOPK_MAX_UI, value=TOPK_SHOW, step=1, key="topk_slider")
        show_reranker_scores = st.toggle("Show reranker scores", value=True, key="toggle_rr")
        show_retriever_scores = st.toggle("Show retriever scores", value=False, key="toggle_ret")
        st.markdown("##### SPECTER2 fusion weights")
        w_ctx = st.slider("Weight for context", 0.0, 1.0, value=0.5, step=0.05, key="wctx_slider")
        w_ta_display = 1.0 - w_ctx
        st.markdown(
            f"<span style='font-size:0.85rem; color:#777;'>"
            f"Weight for title+abstract: <b>{w_ta_display:.2f}</b></span>",
            unsafe_allow_html=True,
        )
        st.markdown("##### Hybrid fusion")
        alpha = st.slider("α for FAISS (z-score fuse)", 0.0, 1.0, value=ALPHA_DEFAULT, step=0.05, key="alpha_slider")

    st.markdown("<h3 style='margin-top:0.5rem;'>Citing Paper Input</h3>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        in_title = st.text_input("Title", placeholder="Enter the title")
    with col2:
        in_authors = st.text_input("Authors (comma-separated)", placeholder="e.g., Alice Smith, Bob Lee")

    in_abstract = st.text_area("Abstract", placeholder="Paste the abstract", height=140)
    in_local_ctx = st.text_area("Local Context (citation context)", placeholder="Paste the local citation context...", height=140)

    run = st.button("🔎 Find citations", type="primary", use_container_width=True)

    # ---- DECIDE WHEN TO RUN SEARCH PIPELINE ----
    query_now = {
        "title": in_title,
        "authors": in_authors,
        "abstract": in_abstract,
        "context": in_local_ctx,
    }
    cfg_now = {
        "retr_mode": retr_mode,
        "w_ctx": float(w_ctx),
        "alpha": float(alpha),
    }

    should_run = False

    # Case 1: user click button
    if run:
        if not any([in_title.strip(), in_authors.strip(), in_abstract.strip(), in_local_ctx.strip()]):
            st.warning("Please enter at least one field (Title, Authors, Abstract, or Local Context).")
        else:
            should_run = True

    # Case 2: auto-update khi thay đổi config mà query vẫn như cũ & đã từng có kết quả
    else:
        prev_query = st.session_state.get("query_saved")
        prev_cfg   = st.session_state.get("retr_cfg_saved")
        if prev_query is not None and prev_cfg is not None:
            if prev_query == query_now and prev_cfg != cfg_now:
                # query unchanged, config changed -> auto rerun
                should_run = True

    if should_run:
        run_full_search_pipeline(
            query=query_now,
            retr_mode=retr_mode,
            w_ctx=w_ctx,
            alpha=alpha,
        )

    # ---- DISPLAY RESULTS (card layout) ----
    if "retr_df" in st.session_state:
        retr_df = st.session_state["retr_df"]
        q_parts = st.session_state.get("q_parts", {
            "title": "", "abstract": "", "context": "", "authors": ""
        })
        retr_mode_used = st.session_state.get("retr_mode_used", "FAISS (SPECTER2 early fusion)")
        explain_map = st.session_state.get("explain_map", {})

        # Sort
        sort_cols = []
        if show_reranker_scores:
            sort_cols.append(("reranker_score", False))
        if show_retriever_scores:
            sort_cols.append(("retriever_score", False))
        if not sort_cols:
            sort_cols = [("reranker_score", False)]

        out_df = retr_df.sort_values(
            by=[c for c,_ in sort_cols],
            ascending=[asc for _,asc in sort_cols]
        ).head(top_k).copy()

        st.markdown(
            f"<div style='margin-top:0.7rem; margin-bottom:0.3rem; font-size:0.95rem; "
            f"color:#555;'>"
            f"<b>{retr_mode_used}</b> · showing top-{len(out_df)} results "
            f"(sorted by {', '.join([c for c,_ in sort_cols])})"
            f"</div>",
            unsafe_allow_html=True,
        )

        st.markdown("<hr style='margin:0.3rem 0 0.8rem 0;'/>", unsafe_allow_html=True)
        st.markdown("<h3 style='font-size:1.1rem; margin-bottom:0.2rem;'>🧾 Candidates</h3>", unsafe_allow_html=True)

        for rank, (idx, row) in enumerate(out_df.iterrows(), start=1):
            row_cid = int(row["corpusid"])
            row_key = f"{row_cid}"
            title = row["title"]
            authors = row["authors"]
            year = str(row["year"])
            venue = row["venue"]
            url = row["url"]
            abstract_full = row["abstract"] if isinstance(row["abstract"], str) else ""
            max_abs_len = 1000
            abstract_short = abstract_full[:max_abs_len] + ("..." if len(abstract_full) > max_abs_len else "")

            with st.container(border=True):
                st.markdown(
                    "<div style='font-size:0.9rem; color:#888; margin-bottom:0.1rem;'>"
                    f"Rank #{rank}</div>",
                    unsafe_allow_html=True,
                )

                # Title + URL
                if url and isinstance(url, str) and url.startswith("http"):
                    st.markdown(
                        f"<div style='font-size:1.0rem; font-weight:600; margin-bottom:0.1rem;'>"
                        f"<a href='{escape(url)}' target='_blank' style='text-decoration:none; color:#1155cc;'>"
                        f"{escape(title)}</a></div>",
                        unsafe_allow_html=True
                    )
                else:
                    st.markdown(
                        f"<div style='font-size:1.0rem; font-weight:600; margin-bottom:0.1rem;'>{escape(title)}</div>",
                        unsafe_allow_html=True
                    )

                # Authors
                if authors:
                    st.markdown(
                        f"<div style='font-size:0.9rem; color:#555; margin-bottom:0.1rem;'><b>Authors:</b> {escape(authors)}</div>",
                        unsafe_allow_html=True
                    )

                # Year & Venue
                meta_bits = []
                if year and year != "nan":
                    meta_bits.append(f"<b>Year:</b> {escape(year)}")
                if venue:
                    meta_bits.append(f"<b>Venue:</b> {escape(venue)}")
                if meta_bits:
                    st.markdown(
                        "<div style='font-size:0.9rem; color:#666; margin-bottom:0.2rem;'>"
                        + " &nbsp;&nbsp;·&nbsp;&nbsp;".join(meta_bits)
                        + "</div>",
                        unsafe_allow_html=True
                    )

                # Scores
                score_html = "<div style='font-size:0.85rem; color:#444; margin-bottom:0.4rem;'>"
                badges = []
                if show_reranker_scores and "reranker_score" in row:
                    badges.append(
                        f"<span style='background:#ecf5ff; color:#1f4b99; padding:2px 6px; "
                        f"border-radius:999px; margin-right:6px;'>"
                        f"<b>Reranker</b>: {row['reranker_score']:.3f}</span>"
                    )
                if show_retriever_scores and "retriever_score" in row:
                    badges.append(
                        f"<span style='background:#f5f5f5; color:#555; padding:2px 6px; "
                        f"border-radius:999px; margin-right:6px;'>"
                        f"<b>Retriever</b>: {row['retriever_score']:.3f}</span>"
                    )
                score_html += "".join(badges) + "</div>"
                if badges:
                    st.markdown(score_html, unsafe_allow_html=True)

                # Abstract
                st.markdown(
                    "<div style='font-size:0.9rem; color:#444; line-height:1.5;'>"
                    "<b>Abstract</b><br/>" + escape(abstract_short) + "</div>",
                    unsafe_allow_html=True
                )

                st.markdown("<div style='height:0.4rem;'></div>", unsafe_allow_html=True)

                # RIS download
                ris_text = build_ris_record(
                    title=title,
                    authors=authors,
                    year=year,
                    venue=venue,
                    abstract=abstract_full,
                    url=url,
                )
                ris_bytes = ris_text.encode("utf-8")
                fname = f"citation_{row_cid}.ris"
                cols_btn = st.columns([0.35, 0.65])
                with cols_btn[0]:
                    clicked_ris = st.download_button(
                        "💾 Download .RIS for EndNote",
                        data=ris_bytes,
                        file_name=fname,
                        mime="application/x-research-info-systems",
                        key=f"ris_dl_{row_key}"
                    )
                    if clicked_ris:
                        add_to_download_history(row)

                # Explainability
                with st.expander("🔍 Explain this match (token overlap)"):
                    ov = explain_map.get(row_cid)
                    if ov is None:
                        st.info("No precomputed explanation for this candidate.")
                    else:
                        doc_parts = {
                            "title": title,
                            "abstract": abstract_full,
                            "authors": authors if isinstance(authors, str) else "",
                            "venue": venue if isinstance(venue, str) else "",
                        }

                        q_html = []
                        q_html.append("<div><b>Query — Title</b><br/>"    + _colorize_text_simple(q_parts["title"],    ov["q"]["title"])    + "</div><br/>")
                        q_html.append("<div><b>Query — Abstract</b><br/>" + _colorize_text_simple(q_parts["abstract"], ov["q"]["abstract"]) + "</div><br/>")
                        q_html.append("<div><b>Query — Context</b><br/>"  + _colorize_text_simple(q_parts["context"],  ov["q"]["context"])  + "</div><br/>")
                        q_html.append("<div><b>Query — Authors</b><br/>"  + _colorize_text_simple(q_parts["authors"],  ov["q"]["authors"])  + "</div>")

                        d_html = []
                        d_html.append("<div><b>Doc — Title</b><br/>"      + _colorize_text_simple(doc_parts["title"],   ov["d"]["title"])   + "</div><br/>")
                        d_html.append("<div><b>Doc — Abstract</b><br/>"   + _colorize_text_simple(doc_parts["abstract"],ov["d"]["abstract"])+ "</div><br/>")
                        d_html.append("<div><b>Doc — Authors</b><br/>"    + _colorize_text_simple(doc_parts["authors"], ov["d"]["authors"]) + "</div><br/>")
                        d_html.append("<div><b>Doc — Venue</b><br/>"      + _colorize_text_simple(doc_parts["venue"],   ov["d"]["venue"])   + "</div>")

                        c1, c2 = st.columns(2)
                        with c1:
                            st.markdown("##### Query highlights")
                            st.markdown("<div style='line-height:1.8; font-size:0.9rem;'>" + "".join(q_html) + "</div>", unsafe_allow_html=True)
                        with c2:
                            st.markdown("##### Candidate highlights")
                            st.markdown("<div style='line-height:1.8; font-size:0.9rem;'>" + "".join(d_html) + "</div>", unsafe_allow_html=True)

                st.markdown("<div style='height:0.4rem;'></div>", unsafe_allow_html=True)

        # # Debug panel
        # with st.expander("🔍 Debug (top candidates raw table)"):
        #     dbg = retr_df.sort_values(
        #         by=[c for c,_ in sort_cols],
        #         ascending=[asc for _,asc in sort_cols]
        #     ).head(TOPK_POOL).copy()
        #     st.dataframe(
        #         dbg[["corpusid","title","venue","year","url","retriever_score","reranker_score"]],
        #         use_container_width=True,
        #         hide_index=True
        #     )

# ==================== PAGE 2: DOWNLOAD HISTORY ====================
if page == "Download history":
    st.markdown("<h3 style='margin-top:0.5rem;'>📜 Download history (.RIS)</h3>", unsafe_allow_html=True)

    history = st.session_state.get("download_history", [])
    if not history:
        st.info("You haven't downloaded any .RIS citations yet.")
    else:
        history_sorted = sorted(history, key=lambda x: x["downloaded_at"], reverse=True)
        st.markdown(f"<div style='font-size:0.9rem; color:#555;'>Total records: <b>{len(history_sorted)}</b> (latest first)</div>", unsafe_allow_html=True)
        st.markdown("<hr style='margin:0.3rem 0 0.8rem 0;'/>", unsafe_allow_html=True)

        for i, rec in enumerate(history_sorted, start=1):
            with st.container(border=True):
                title = rec["title"]
                authors = rec["authors"]
                year = rec["year"]
                venue = rec["venue"]
                url = rec["url"]
                abstract_full = rec.get("abstract", "") or ""
                max_abs_len_hist = 600
                abstract_short = abstract_full[:max_abs_len_hist] + ("..." if len(abstract_full) > max_abs_len_hist else "")
                downloaded_at = rec.get("downloaded_at","")

                st.markdown(
                    f"<div style='font-size:0.85rem; color:#888;'>History #{i}</div>",
                    unsafe_allow_html=True
                )

                if url and isinstance(url, str) and url.startswith("http"):
                    st.markdown(
                        f"<div style='font-size:1.0rem; font-weight:600; margin-bottom:0.1rem;'>"
                        f"<a href='{escape(url)}' target='_blank' style='text-decoration:none; color:#1155cc;'>"
                        f"{escape(title)}</a></div>",
                        unsafe_allow_html=True
                    )
                else:
                    st.markdown(
                        f"<div style='font-size:1.0rem; font-weight:600; margin-bottom:0.1rem;'>{escape(title)}</div>",
                        unsafe_allow_html=True
                    )

                if authors:
                    st.markdown(
                        f"<div style='font-size:0.9rem; color:#555; margin-bottom:0.1rem;'><b>Authors:</b> {escape(authors)}</div>",
                        unsafe_allow_html=True
                    )

                meta_bits = []
                if year and year != "nan":
                    meta_bits.append(f"<b>Year:</b> {escape(year)}")
                if venue:
                    meta_bits.append(f"<b>Venue:</b> {escape(venue)}")
                if meta_bits:
                    st.markdown(
                        "<div style='font-size:0.9rem; color:#666; margin-bottom:0.1rem;'>"
                        + " &nbsp;&nbsp;·&nbsp;&nbsp;".join(meta_bits)
                        + "</div>",
                        unsafe_allow_html=True
                    )

                if downloaded_at:
                    st.markdown(
                        f"<div style='font-size:0.8rem; color:#999; margin-bottom:0.3rem;'>"
                        f"Downloaded at: {escape(downloaded_at)}</div>",
                        unsafe_allow_html=True
                    )

                st.markdown(
                    "<div style='font-size:0.9rem; color:#444; line-height:1.5; margin-bottom:0.3rem;'>"
                    "<b>Abstract</b><br/>" + escape(abstract_short) + "</div>",
                    unsafe_allow_html=True
                )

                ris_text = build_ris_record(
                    title=title,
                    authors=authors,
                    year=year,
                    venue=venue,
                    abstract=abstract_full,
                    url=url,
                )
                ris_bytes = ris_text.encode("utf-8")
                fname = f"citation_{rec['corpusid']}.ris"
                st.download_button(
                    "💾 Download .RIS again",
                    data=ris_bytes,
                    file_name=fname,
                    mime="application/x-research-info-systems",
                    key=f"ris_hist_{rec['corpusid']}_{i}"
                )

                st.markdown("<div style='height:0.4rem;'></div>", unsafe_allow_html=True)
