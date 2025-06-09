# Simple RAG System with MongoDB & OpenAI

A lightweight, end‑to‑end **retrieve‑augmented generation (RAG)** demo that couples
OpenAI embeddings with MongoDB Atlas Vector Search and exposes a chat‑style UI
through Streamlit.

---

## ✨ Key Features

* **Vector ingestion** – Converts text fields in MongoDB documents to dense
  vectors and stores them alongside your data.
* **Semantic retrieval** – Fast, high‑precision nearest‑neighbour search powered
  by Atlas Vector Search.
* **Context‑aware generation** – OpenAI completions leverage retrieved
  documents to produce grounded answers.
* **Minimal UI** – One‑file Streamlit app for rapid prototyping and interactive
  exploration.

---

## 🚀 Quick start

### 1 · install

```bash
pip install -r requirements.txt
```

### 2 · Configure environment

Copy **`env.example`** to **`.env`** and fill in:


### 3 · Ingest sample data

This ships with the public [**sample\_mflix**](https://www.mongodb.com/docs/atlas/sample-data/sample-mflix/) movie dataset.
Run `embedding.py` to embed the documents’ `plot` field and upsert vectors:



### 4 · Launch the RAG UI

```bash
streamlit run generate.py
```

Point your browser to `http://localhost:8501` and start asking *movie‑related*
questions—e.g. *“Which Steven Spielberg films involve aliens?”*  The model will
retrieve relevant plots and cite them in its response.

