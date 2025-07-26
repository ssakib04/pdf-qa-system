📄 PDF-based Question Answering System using LLM Embeddings
This project allows users to upload a PDF document and ask questions about its contents in Bangla and English. The system retrieves relevant context chunks using semantic search and answers queries using a language model.

🚀 Setup Guide
Clone this repository
bash
CopyEdit
git clone https://github.com/yourusername/pdf-qa-system.git
cd pdf-qa-system


Create a virtual environment
bash
CopyEdit
python -m venv venv
source venv/bin/activate  # On Windows use venv\Scripts\activate


Install dependencies
bash
CopyEdit
pip install -r requirements.txt


Run the application
bash
CopyEdit
python app.py



🧰 Used Tools, Libraries, and Packages
Tool/Library
Purpose
PyMuPDF (fitz)
Extract text from PDF
langchain
Text splitting, embeddings, retrieval
sentence-transformers
Embedding generation using all-MiniLM-L6-v2
faiss
Efficient vector similarity search
tkinter
Basic GUI interface
shutil, os
File handling
openai
LLM for answering (if implemented)


🧪 Sample Queries and Outputs
✅ English:
Query: "How many hearts does an octopus have?"
Answer: An octopus has three hearts.

✅ Bangla:
Query: "অনুপমের ভাষায় সুপুরুষ কাকে বলা হয়েছে?"
Answer: শুম্ভুনাথ।

🧾 API Documentation
Note: If you've added an API, here's a placeholder:
POST /ask
Ask a question from uploaded PDF.
Request:
json
CopyEdit
{
  "question": "How many hearts does an octopus have?"
}

Response:
json
CopyEdit
{
  "answer": "An octopus has three hearts."
}


📊 Evaluation Matrix
We use the langchain QAEvalChain to evaluate the performance of answers using metrics like:
Relevance

🧠 Design Decisions
📘 What method or library did you use to extract the text, and why?
We used PyMuPDF (fitz) for PDF parsing due to its speed, accuracy, and good support for maintaining layout. However, some PDFs with complex formatting (e.g., multi-column or scanned pages) still pose challenges.

🧩 What chunking strategy did you choose?
We used paragraph-based chunking with a character limit (e.g., 500–1000 characters) using Langchain's RecursiveCharacterTextSplitter. This maintains semantic integrity and gives better retrieval accuracy than fixed-length or sentence-based splitting.

🧬 What embedding model did you use?
We used sentence-transformers/all-MiniLM-L6-v2 for embedding generation because:
It’s fast and lightweight.
It performs well in semantic similarity tasks.
It captures contextual meaning and relationships across both English and Bangla (to a reasonable extent).

🧮 How are you comparing the query with stored chunks?
We use cosine similarity with FAISS for fast nearest-neighbor search. Chunks most similar to the query are retrieved and passed to the LLM to construct a final answer.

🔍 How do you ensure meaningful comparison?
By embedding both queries and chunks using the same transformer model, we ensure they exist in the same semantic space. We also:
Normalize text,
Strip non-textual noise,
Choose context-rich chunks.
If the query is vague or lacks context, the system may return general or irrelevant results — this can be improved with query rephrasing, prompt engineering, or reranking.

🎯 Do the results seem relevant?
Generally, yes — especially when queries are specific and aligned with the content. Relevance can be improved by:
Better chunking (e.g., hybrid methods),
Using stronger embedding models like text-embedding-3-small,
Adding reranking or multi-hop reasoning.

