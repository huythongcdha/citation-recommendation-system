# -*- coding: utf-8 -*-
"""
Tạo triplet_sample_citations.parquet từ sample_citations.parquet bằng Hybrid Retriever:
- Encode query (early fusion) với SPECTER2 + 2 adapter (retrieval cho title+abstract, ad-hoc cho context)
- FAISS early search + BM25 search -> Hybrid (alpha=0.5, z-score)
- Lọc candidates: != citing_id, != cited_id, và (nếu có citing_year) year(candidate) < citing_year
- Chọn ngẫu nhiên tối đa 8 negatives -> lưu triplet: (citing_id, context, cited_id, non_cited_id[list])
"""

import os
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm

import faiss
import torch
from transformers import AutoTokenizer
from adapters import AutoAdapterModel

# BM25 (Pyserini)
from pyserini.search.lucene import LuceneSearcher

# Parquet IO (partitioned dataset)
import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.dataset as ds

# ============================== CONFIG ==============================
# --- Input datasets ---
SAMPLE_CITATIONS = Path(r"E:\Citation Recommendation\data\sample\sample_citations_100k.parquet")
PAPERS_DIR       = Path(r"E:\Citation Recommendation\data\stage\papers_with_abstracts_cleaned")
# --- Indexes ---
FAISS_INDEX_PATH = Path(r"E:\Citation Recommendation\data\indexes\ivf_flat_ip_11m.faiss")
BM25_INDEX_DIR   = Path(r"E:\Citation Recommendation\data\bm25\lucene-index-papers_11m")

# --- Output ---
OUT_PARQUET      = Path(r"E:\Citation Recommendation\data\sample\triplet_sample_citations_100k_K_8.parquet")

# --- SPECTER2 encoders ---
BASE_MODEL   = "allenai/specter2_base"
ADAPTER_TA   = "allenai/specter2"              # retrieval adapter (doc-style)
ADAPTER_ADHQ = "allenai/specter2_adhoc_query"  # ad-hoc query adapter

# --- Retriever params ---
MAX_LEN        = 512
BATCH_SIZE     = 128
TOPK_RETRIEVER = 500
ALPHA_HYBRID   = 0.5         # trọng số hybrid giữa FAISS early và BM25
RRF_K          = 60          # nếu dùng RRF (ở đây ta dùng z-score fusion nên không dùng)
NEG_PER_REC    = 8           # tối đa 8 negatives/record
W_TA, W_CTX    = 0.5, 0.5    # early fusion weights (title+abstract vs context)

# ============================== UTILS ==============================
def get_dataset_columns(path: Path) -> list[str]:
    """Lấy danh sách cột từ dataset (file hoặc thư mục partitioned) mà không đọc dữ liệu."""
    if path.is_dir():
        return list(ds.dataset(str(path)).schema.names)
    else:
        return list(pq.read_schema(str(path)).names)

def read_partitioned_papers_minimal(papers_dir: Path, columns: list[str]) -> pd.DataFrame:
    """
    Đọc partitioned Parquet (part=**) chỉ với các cột cần thiết.
    Trả về pandas DataFrame.
    """
    dset = ds.dataset(str(papers_dir))
    to_read = [c for c in columns if c in dset.schema.names]
    table = dset.to_table(columns=to_read)
    return table.to_pandas()

def l2norm_vecs(x: np.ndarray) -> np.ndarray:
    if x.ndim == 1:
        n = np.linalg.norm(x) + 1e-12
        return (x / n).astype(np.float32)
    n = np.linalg.norm(x, axis=1, keepdims=True) + 1e-12
    return (x / n).astype(np.float32)

@torch.no_grad()
def encode_texts(texts, tokenizer, model, device, batch_size=BATCH_SIZE, use_cls=True):
    """Encode list[str] -> (N,768), L2-normalize; dùng CLS pooling (mặc định SPECTER2)."""
    embs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        enc = tokenizer(
            batch, max_length=MAX_LEN, truncation=True,
            padding=True, return_tensors="pt"
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        out = model(**enc)
        hidden = out.last_hidden_state  # (B, L, 768)
        if use_cls:
            vec = hidden[:, 0, :]
        else:
            mask = enc["attention_mask"].unsqueeze(-1).float()
            vec = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1.0)
        vec = vec.detach().cpu().float().numpy()
        embs.append(l2norm_vecs(vec))
    return np.vstack(embs)

def load_specter2_models():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

    # Doc-style (TA)
    m_ta = AutoAdapterModel.from_pretrained(BASE_MODEL)
    m_ta.load_adapter(ADAPTER_TA, source="hf", load_as="a_ta", set_active=True)
    m_ta.eval().to(device)

    # Ad-hoc (CTX)
    m_adhq = AutoAdapterModel.from_pretrained(BASE_MODEL)
    m_adhq.load_adapter(ADAPTER_ADHQ, source="hf", load_as="a_adhq", set_active=True)
    m_adhq.eval().to(device)

    return tokenizer, m_ta, m_adhq, device

def load_faiss_index():
    if not FAISS_INDEX_PATH.exists():
        raise FileNotFoundError(f"FAISS index not found: {FAISS_INDEX_PATH}")
    index = faiss.read_index(FAISS_INDEX_PATH.as_posix())
    # nếu IVF, đặt nprobe hợp lý
    try:
        index.nprobe = min(getattr(index, "nlist", 128), 128)
    except Exception:
        pass
    return index

def faiss_search_batch(index, Q: np.ndarray, topk: int):
    D, I = index.search(Q.astype(np.float32), topk)
    return I.astype(np.int64), D.astype(np.float32)

def zscore(x: np.ndarray) -> np.ndarray:
    mu = x.mean()
    sigma = x.std()
    return np.zeros_like(x) if sigma == 0 else (x - mu) / sigma

# ============================== MAIN ==============================
def main():
    # 1) Load sample_citations
    need_cols_cit = [
        "citing_id","cited_id","context","citing_title","citing_abstract",
        "citing_authors_concat","citing_year","part"
    ]
    df_cit = pd.read_parquet(SAMPLE_CITATIONS, columns=need_cols_cit).fillna({
        "context":"", "citing_title":"", "citing_abstract":"", "citing_authors_concat":""
    })
    df_cit["citing_id"] = df_cit["citing_id"].astype("int64")
    df_cit["cited_id"]  = df_cit["cited_id"].astype("int64")

    # 2) Load papers metadata (partitioned dir) -> cần 'corpusid' và 'year' để filter thời gian
    cols_wanted = ["corpusid","title","abstract","authors_concat","venue","year"]
    cols_avail  = set(get_dataset_columns(PAPERS_DIR))
    if "year" not in cols_avail or "corpusid" not in cols_avail:
        raise KeyError(
            f"Papers dataset phải có cột 'corpusid' và 'year'. Schema hiện có: {sorted(cols_avail)}"
        )
    meta_cols = [c for c in cols_wanted if c in cols_avail]
    print("Đang đọc papers metadata (partitioned)...")
    df_meta = read_partitioned_papers_minimal(PAPERS_DIR, meta_cols)
    df_meta["corpusid"] = df_meta["corpusid"].astype("int64")
    meta_index = df_meta.set_index("corpusid")  # để tra nhanh year & join nếu cần

    # 3) Load models + indexes
    print("Loading SPECTER2 encoders...")
    tokenizer, model_ta, model_adhq, device = load_specter2_models()
    print("Loading FAISS index...")
    faiss_index = load_faiss_index()
    print("Loading BM25 LuceneSearcher...")
    if not BM25_INDEX_DIR.exists():
        raise FileNotFoundError(f"BM25 index directory not found: {BM25_INDEX_DIR}")
    bm25 = LuceneSearcher(str(BM25_INDEX_DIR))

    # 4) Chuẩn bị văn bản và encode queries (TA/CTX)
    print("Encoding SPECTER2 queries (TA/CTX)...")
    texts_ta, texts_ctx, texts_bm25 = [], [], []
    for _, row in df_cit.iterrows():
        title = str(row["citing_title"]).strip()
        abstract = str(row["citing_abstract"]).strip()
        context = str(row["context"]).strip()

        # TA: "title [SEP] abstract"
        ta = title if title else ""
        if abstract:
            ta = f"{ta} {tokenizer.sep_token} {abstract}" if ta else abstract
        texts_ta.append(ta if ta else "")

        # CTX: context (nếu trống thì fallback title)
        texts_ctx.append(context if context else (title or ""))

        # BM25 text: context (nếu trống thì fallback title)
        texts_bm25.append(context if context else (title or ""))

    Q_ta  = encode_texts(texts_ta,  tokenizer, model_ta,   device, batch_size=BATCH_SIZE)
    Q_ctx = encode_texts(texts_ctx, tokenizer, model_adhq, device, batch_size=BATCH_SIZE)
    Q_early = l2norm_vecs(W_TA * Q_ta + W_CTX * Q_ctx)    # (N,768)

    # 5) FAISS search (early)
    print("FAISS search (early)...")
    I_early, D_early = faiss_search_batch(faiss_index, Q_early, topk=TOPK_RETRIEVER)

    # 6) BM25 search
    print("BM25 searching...")
    N = len(df_cit)
    I_bm25 = np.zeros((N, TOPK_RETRIEVER), dtype=np.int64)
    S_bm25 = np.zeros((N, TOPK_RETRIEVER), dtype=np.float32)
    for i, qtext in tqdm(enumerate(texts_bm25), total=N, desc="BM25"):
        hits = bm25.search(qtext, TOPK_RETRIEVER)
        if not hits:
            continue
        docids = [int(h.docid) for h in hits]
        scores = [float(h.score) for h in hits]
        k = min(TOPK_RETRIEVER, len(docids))
        I_bm25[i, :k] = np.asarray(docids[:k], dtype=np.int64)
        S_bm25[i, :k] = np.asarray(scores[:k], dtype=np.float32)

    # 7) HYBRID = zscore(FAISS) & zscore(BM25) với alpha=0.5 (trộn id-chung)
    print("Hybrid (FAISS early + BM25, alpha=0.5)...")
    I_hybrid = np.zeros((N, TOPK_RETRIEVER), dtype=np.int64)
    for i in range(N):
        ids_faiss = I_early[i]
        ids_bm    = I_bm25[i]

        cand_ids = set(ids_faiss.tolist()) | set(ids_bm.tolist())
        if -1 in cand_ids:
            cand_ids.discard(-1)
        cand_ids = list(cand_ids)

        sf, sb = [], []
        for cid in cand_ids:
            # score FAISS: dùng D_early (cosine/IP); nếu vắng, 0.0
            mask_f = (ids_faiss == cid)
            sf.append(D_early[i][mask_f][0] if mask_f.any() else 0.0)
            # score BM25
            mask_b = (ids_bm == cid)
            sb.append(S_bm25[i][mask_b][0] if mask_b.any() else 0.0)
        sf = np.asarray(sf, dtype=np.float32)
        sb = np.asarray(sb, dtype=np.float32)

        s_f = zscore(sf)
        s_b = zscore(sb)
        s   = ALPHA_HYBRID * s_f + (1.0 - ALPHA_HYBRID) * s_b

        order = np.argsort(-s)
        top_ids = [cand_ids[j] for j in order[:TOPK_RETRIEVER]]
        I_hybrid[i] = np.asarray(top_ids, dtype=np.int64)

    # 8) Tạo TRIPLET: pick <=8 negatives sau lọc (không trùng citing/cited, và year < citing_year nếu có)
    print("Selecting negatives and writing triplets...")
    triplet_rows = []
    for i in tqdm(range(N), desc="Triplets"):
        citing_id = int(df_cit.iloc[i]["citing_id"])
        cited_id  = int(df_cit.iloc[i]["cited_id"])
        citing_yr = df_cit.iloc[i]["citing_year"]
        has_year  = pd.notna(citing_yr)

        # hybrid candidates
        cand = I_hybrid[i]
        # loại citing_id, cited_id
        cand = cand[(cand != citing_id) & (cand != cited_id)]
        # loại id không có trong meta
        cands_df = pd.DataFrame({"corpusid": cand})
        cands_df = cands_df[cands_df["corpusid"].isin(meta_index.index)]

        # ràng buộc: year(candidate) < citing_year (nếu có)
        if has_year:
            years = meta_index.loc[cands_df["corpusid"], "year"].values
            cands_df = cands_df.assign(year=years)
            cands_df = cands_df[pd.notna(cands_df["year"])]
            # nhiều dataset để year float -> ép int an toàn
            cands_df["year"] = cands_df["year"].astype("int64", errors="ignore")
            cands_df = cands_df[cands_df["year"] < int(citing_yr)]

        # chọn ngẫu nhiên tối đa 8
        if len(cands_df) > 0:
            take = min(NEG_PER_REC, len(cands_df))
            chosen = cands_df.sample(n=take, replace=False, random_state=None)["corpusid"].tolist()
        else:
            chosen = []

        triplet_rows.append({
            "citing_id": citing_id,
            "context": df_cit.iloc[i]["context"],
            "cited_id": cited_id,
            "non_cited_id": [int(x) for x in chosen],
        })

    # 9) Lưu Parquet với list<int64> cho non_cited_id
    citing_id_arr = pa.array([int(r["citing_id"]) for r in triplet_rows], type=pa.int64())
    context_arr   = pa.array([r["context"] for r in triplet_rows], type=pa.string())
    cited_id_arr  = pa.array([int(r["cited_id"]) for r in triplet_rows], type=pa.int64())
    non_cited_arr = pa.array(
        [pa.array(r["non_cited_id"], type=pa.int64()) for r in triplet_rows],
        type=pa.list_(pa.int64())
    )

    table = pa.table({
        "citing_id": citing_id_arr,
        "context": context_arr,
        "cited_id": cited_id_arr,
        "non_cited_id": non_cited_arr,
    })

    OUT_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, OUT_PARQUET)
    print(f"✅ Saved triplets: {OUT_PARQUET} ({table.num_rows:,} rows)")


if __name__ == "__main__":
    main()