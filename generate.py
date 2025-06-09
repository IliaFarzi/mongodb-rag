import os, backoff, streamlit as st
from typing import List, Dict
from dotenv import load_dotenv
from pymongo import MongoClient
import openai, numpy as np


# ───────── .env & configuration ─────────
load_dotenv()
MONGODB_URI = os.getenv("MONGODB_URI")
DB_NAME, COLL_NAME = "movie_rag", "small_movies"
EMBED_MODEL = "text-embedding-3-small"
LLM_MODEL   = "gpt-4o-mini"
openai.api_key = os.environ["OPENAI_API_KEY"]

client = MongoClient(MONGODB_URI, serverSelectionTimeoutMS=5000)
coll   = client[DB_NAME][COLL_NAME]

# ───────── Helper functions ─────────
@backoff.on_exception(backoff.expo,
                      (openai.RateLimitError,
                       openai.APIConnectionError,
                       openai.APIError),
                      max_time=60)
def embed_query(text: str) -> np.ndarray:
    vec = openai.embeddings.create(model=EMBED_MODEL, input=text).data[0].embedding
    v = np.asarray(vec, dtype=np.float32)
    v /= np.linalg.norm(v) or 1.0
    return v

def retrieve_local(question: str, k: int = 4) -> List[Dict]:
    q_vec = embed_query(question)
    docs = list(coll.find({}, {"_id": 0, "title": 1, "plot": 1, "vector": 1}))
    for d in docs:
        v = np.asarray(d["vector"], dtype=np.float32)
        v /= np.linalg.norm(v) or 1.0
        d["score"] = float(np.dot(q_vec, v))
    docs.sort(key=lambda x: x["score"], reverse=True)
    return docs[:k]

@backoff.on_exception(backoff.expo,
                      (openai.RateLimitError,
                       openai.APIConnectionError,
                       openai.APIError),
                      max_time=60)
def generate_answer(question: str, passages: List[Dict]) -> str:
    context = "\n\n---\n\n".join(
        f"Title: {p['title']}\nPlot: {p['plot']}" for p in passages
    )

    system_prompt = (
        "You are a helpful movie assistant. "
        "Answer ONLY using the provided context and cite the movie titles in brackets."
    )
    user_prompt = (
        f"{system_prompt}\n\nContext:\n{context}\n\n"
        f"Question: {question}\nAnswer (Farsi preferred if user is Farsi):"
    )

    resp = openai.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": user_prompt}],
        temperature=0.2,
    )
    return resp.choices[0].message.content.strip()

# ───────── Streamlit UI ─────────
st.title("🎬 Movie Retriever & Generator")
st.write(
    "سؤال بپرسید،  خلاصه‌ها را پیدا می‌کند و "
    "سپس مدل زبانی (LLM) با تکیه بر همان متون پاسخ می‌دهد."
)

question = st.text_input("❓ پرسش خود را بنویسید …",
                         placeholder="مثلاً: فیلم مستندی دربارهٔ چاقی در آمریکا")
k = st.slider("تعداد نتایج (k)", 1, 10, value=4)

if st.button("بپرس!") and question.strip():
    with st.spinner("🔍 در حال بازیابی اسناد…"):
        docs = retrieve_local(question, k=k)

    if not docs:
        st.warning("هیچ سندی پیدا نشد.")
    else:
        st.subheader("متن‌های بازیابی‌شده (Context)")
        for i, d in enumerate(docs, 1):
            with st.expander(f"{i}. {d['title']}  (score {d['score']:.3f})"):
                st.write(d["plot"])

        with st.spinner("✍️ در حال تولید پاسخ…"):
            answer = generate_answer(question, docs)

        st.subheader("پاسخ مدل")
        st.write(answer)
