# app_local_knn.py
import os, backoff, streamlit as st
from typing import List, Dict
from dotenv import load_dotenv
from pymongo import MongoClient
import openai, numpy as np

# ---------- .env ----------
load_dotenv()
MONGODB_URI = os.getenv("MONGODB_URI")
DB_NAME, COLL_NAME = "movie_rag", "small_movies"
EMBED_MODEL = "text-embedding-3-small"
openai.api_key = os.environ["OPENAI_API_KEY"]

# ---------- اتصال به MongoDB ----------
client = MongoClient(MONGODB_URI, serverSelectionTimeoutMS=5000)
coll   = client[DB_NAME][COLL_NAME]

# ---------- توابع کمکی ----------
@backoff.on_exception(backoff.expo,
                      (openai.RateLimitError,
                       openai.APIConnectionError,
                       openai.APIError),
                      max_time=60)
def embed_query(text: str) -> np.ndarray:
    """بردار واحد (unit-norm) پرسش کاربر را برمی‌گرداند."""
    vec = openai.embeddings.create(model=EMBED_MODEL, input=text).data[0].embedding
    v = np.asarray(vec, dtype=np.float32)
    v /= np.linalg.norm(v) or 1.0
    return v

def retrieve_local(question: str, k: int = 4) -> List[Dict]:
    """ ساده: شبیه‌ترین اسناد را با شباهت کسینوسی پیدا می‌کند."""
    q_vec = embed_query(question)

    # همهٔ اسناد موردنیاز را یک‌باره می‌خوانیم
    docs = list(coll.find({}, {"_id": 0, "title": 1, "plot": 1, "vector": 1}))

    # محاسبهٔ شباهت کسینوسی برای هر سند
    for d in docs:
        v = np.asarray(d["vector"], dtype=np.float32)
        v /= np.linalg.norm(v) or 1.0
        d["score"] = float(np.dot(q_vec, v))     # مقدار اسکالر

    # مرتب‌سازی و بازگشت k مورد برتر
    docs.sort(key=lambda x: x["score"], reverse=True)
    return docs[:k]

# ---------- رابط Streamlit ----------
st.title("🎬 Movie Retriever")


question = st.text_input("❓ پرسش خود را بنویسید …", placeholder="مثلاً: فیلم مستندی دربارهٔ چاقی در آمریکا")
k = st.slider("تعداد نتایج (k)", 1, 10, value=4)

if st.button("جستجو") and question.strip():
    with st.spinner("در حال جستجو…"):
        hits = retrieve_local(question, k=k)

    if not hits:
        st.warning("هیچ نتیجه‌ای پیدا نشد.")
    else:
        for idx, h in enumerate(hits, start=1):
            st.subheader(f"{idx}. {h['title']}  —  (score {h['score']:.3f})")
            st.write(h["plot"])
