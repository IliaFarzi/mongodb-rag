import os
from typing import List

from pymongo import MongoClient, UpdateOne
import openai
import backoff
from dotenv import load_dotenv

load_dotenv()
# ---------- تنظیمات ----------
DB_NAME = "movie_rag"
COLL_NAME = "small_movies"

openai.api_key = os.getenv("OPENAI_API_KEY") 
EMBED_MODEL = "text-embedding-3-small"         

# ---------- توابع کمکی ----------
@backoff.on_exception(
    backoff.expo,
    (openai.RateLimitError, openai.APIConnectionError),
    max_time=60
)
def embed_plot(text: str) -> List[float]:
    """ارسال plot به OpenAI و دریافت بردار."""
    response = openai.embeddings.create(model=EMBED_MODEL, input=text)
    return response.data[0].embedding   # type: ignore

# ---------- پردازش دسته‌ای ----------
client = MongoClient(os.getenv("MONGODB_URI"))
coll = client[DB_NAME][COLL_NAME]

cursor = coll.find({}, projection=["fullplot"])   # فقط plot لازم داریم
ops: List[UpdateOne] = []
batch_size=10

for doc in cursor:
    plot_text = doc.get("fullplot")
    if not plot_text:          # اگر سند plot ندارد، رد می‌کنیم
        continue

    vector = embed_plot(plot_text)      # تماس با OpenAI
    ops.append(
       UpdateOne({"_id": doc["_id"]},
       {"$set": {"vector": vector}})
        )

    if len(ops) >= batch_size:
        coll.bulk_write(ops)
        ops.clear()
        print("+", end="", flush=True)

if ops:                              # ته‌مانده
    coll.bulk_write(ops)

print("\nDONE.")

