import fitz  # PyMuPDF
import re
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from openai import OpenAI
import tkinter as tk
from tkinter import filedialog, scrolledtext
from flask import Flask, request, jsonify
import threading


# -------------------------- Core Functions --------------------------

def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    full_text = ""
    for page in doc:
        full_text += page.get_text()
    return full_text

def clean_text(text):
    text = re.sub(r'\n+', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def chunk_text(text, chunk_size=800, overlap=200):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks

def embed_chunks(chunks, model_name='sentence-transformers/all-MiniLM-L6-v2'):
    model = SentenceTransformer(model_name)
    embeddings = model.encode(chunks, show_progress_bar=True)
    return model, embeddings

def create_faiss_index(embeddings):
    dim = embeddings[0].shape[0]
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings).astype('float32'))
    return index

def search_index(query, embedder, index, chunks, k=3):
    query_embedding = embedder.encode([query]).astype('float32')
    D, I = index.search(query_embedding, k)
    results = [(chunks[i], D[0][j]) for j, i in enumerate(I[0])]
    return results

def generate_answer_with_openai(query, context):
    prompt = f"""You are a helpful assistant. Use the following context to answer the question and answer in the language of the context.

Context:
{context}

Question:
{query}

Answer:"""

    client = OpenAI(api_key="YOUR_API_KEY_HERE") 

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
    )

    return response.choices[0].message.content

# -------------------------- Evaluation Function --------------------------

def evaluate_system(query, expected_keywords, pdf_path):
    text = clean_text(extract_text_from_pdf(pdf_path))
    chunks = chunk_text(text)
    embedder, embeddings = embed_chunks(chunks)
    index = create_faiss_index(embeddings)
    top_chunks = search_index(query, embedder, index, chunks, k=3)
    context = "\n\n".join([chunk for chunk, _ in top_chunks])
    answer = generate_answer_with_openai(query, context)

    match = all(keyword.lower() in answer.lower() for keyword in expected_keywords)
    return {"query": query, "expected_keywords": expected_keywords, "answer": answer, "passed": match}

# -------------------------- Flask API --------------------------

app = Flask(__name__)
loaded_chunks = []
embedder_model = None
faiss_index = None

@app.route("/upload", methods=["POST"])
def upload_pdf():
    global loaded_chunks, embedder_model, faiss_index
    file = request.files["file"]
    file.save("uploaded.pdf")
    text = clean_text(extract_text_from_pdf("uploaded.pdf"))
    loaded_chunks = chunk_text(text)
    embedder_model, embeddings = embed_chunks(loaded_chunks)
    faiss_index = create_faiss_index(embeddings)
    return jsonify({"status": "PDF processed and indexed"})

@app.route("/ask", methods=["POST"])
def ask_question():
    global loaded_chunks, embedder_model, faiss_index
    data = request.json
    query = data.get("query", "")
    top_chunks = search_index(query, embedder_model, faiss_index, loaded_chunks, k=3)
    context = "\n\n".join([chunk for chunk, _ in top_chunks])
    answer = generate_answer_with_openai(query, context)
    return jsonify({"query": query, "answer": answer})

def run_flask_app():
    app.run(port=5000)

# -------------------------- GUI --------------------------

def run_query():
    query = entry.get()
    text_widget.delete('1.0', tk.END)

    text_widget.insert(tk.END, "📄 Extracting text...\n")
    text = extract_text_from_pdf(pdf_path)
    text = clean_text(text)
    chunks = chunk_text(text)

    text_widget.insert(tk.END, "🔎 Embedding chunks...\n")
    embedder, embeddings = embed_chunks(chunks)

    text_widget.insert(tk.END, "📁 Creating FAISS index...\n")
    index = create_faiss_index(embeddings)

    text_widget.insert(tk.END, f"🔍 Searching for top matching chunks for: {query}\n")
    top_chunks = search_index(query, embedder, index, chunks, k=3)

    context = "\n\n".join([chunk for chunk, dist in top_chunks])
    for i, (chunk, dist) in enumerate(top_chunks):
        text_widget.insert(tk.END, f"\nChunk {i+1} (distance={dist:.4f}):\n{chunk}\n")

    text_widget.insert(tk.END, "\n🌐 Querying OpenAI API...\n")
    answer = generate_answer_with_openai(query, context)
    text_widget.insert(tk.END, "\n🧠 Final Answer:\n" + answer)

# -------------------------- Main --------------------------

pdf_path = ""  # You can make this file dialog based if desired

if __name__ == "__main__":
    # Start Flask API in a background thread
    threading.Thread(target=run_flask_app, daemon=True).start()

    # Launch GUI
    root = tk.Tk()
    root.title("PDF-Based Question Answering System")

    frame = tk.Frame(root)
    frame.pack(padx=10, pady=10)

    label = tk.Label(frame, text="Enter your question:")
    label.pack()

    entry = tk.Entry(frame, width=80)
    entry.pack()

    btn = tk.Button(frame, text="Submit", command=run_query)
    btn.pack(pady=5)

    text_widget = scrolledtext.ScrolledText(root, wrap=tk.WORD, width=100, height=30)
    text_widget.pack(padx=10, pady=10)

    root.mainloop()
