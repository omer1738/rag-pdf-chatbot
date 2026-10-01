# 📚 RAG PDF Chatbot

Chat with your PDF documents. Upload one or more PDFs, ask questions in plain English, and get answers based **only on what is written in your files**.

Built with **Python, Streamlit, LangChain, FAISS and Groq**.

---

## ✨ Features

- 📄 Upload **one or many PDFs** at once
- 🔍 **Semantic search**: finds the right passages by meaning, not just keywords
- ⚡ **Streaming answers**: text appears word by word, like ChatGPT
- 🛡️ **Honest answers**: if the answer is not in the PDF, the bot says so instead of guessing
- 💬 Clean **chat interface** with conversation history
- 📥 **Download** your conversation as a CSV file
- 🔧 **Switch AI models** from the sidebar without touching the code

---

## 🧠 How it works

This project uses **RAG (Retrieval-Augmented Generation)**: look up the relevant text first, then let the AI write the answer.

```
 PDF files
    │
    ▼
 1. Extract text            (PyPDF2)
    │
    ▼
 2. Split into chunks       (1000 characters, 200 overlap)
    │
    ▼
 3. Create embeddings       (sentence-transformers/all-MiniLM-L6-v2, runs locally)
    │
    ▼
 4. Store in vector DB      (FAISS, in memory)
    │
    ──────────  when you ask a question  ──────────
    │
    ▼
 5. Embed the question
    │
    ▼
 6. Retrieve the 4 most similar chunks
    │
    ▼
 7. Send chunks + question to the LLM   (Groq)
    │
    ▼
 8. Stream the answer to the screen
```

Only the 4 relevant chunks are sent to the LLM, never the whole document.

---

## 🛠️ Tech stack

| Part | Tool |
|---|---|
| Interface | [Streamlit](https://streamlit.io/) |
| PDF reading | [PyPDF2](https://pypi.org/project/PyPDF2/) |
| Text splitting and chains | [LangChain](https://www.langchain.com/) |
| Embeddings | [Hugging Face](https://huggingface.co/) `all-MiniLM-L6-v2` |
| Vector store | [FAISS](https://github.com/facebookresearch/faiss) |
| LLM | [Groq](https://groq.com/) |

---

## 🚀 Getting started

### 1. Clone the repository

```bash
git clone https://github.com/omer1738/rag-pdf-chatbot.git
cd rag-pdf-chatbot
```

### 2. (Optional) Create a virtual environment

```bash
python -m venv myenv

# Windows
.\myenv\Scripts\activate

# macOS / Linux
source myenv/bin/activate
```

### 3. Install the requirements

```bash
pip install -r requirements.txt
```

> The first install is large because `sentence-transformers` installs PyTorch. Python 3.10 to 3.13 is recommended.

### 4. Get a free Groq API key

1. Go to [console.groq.com/keys](https://console.groq.com/keys)
2. Sign up and click **Create API Key**
3. Copy the key (it starts with `gsk_`)

### 5. Run the app

```bash
streamlit run app.py
```

The app opens at `http://localhost:8501`.

---

## 💡 How to use

1. Paste your **Groq API key** in the sidebar.
2. Upload your PDF files.
3. Click **Submit & Process** and wait for "Done!".
4. Type a question in the chat box.

> The first run is slower because the embedding model is downloaded once.

### Try it with the sample file

The repo includes `nova_coffee_test.pdf`, a fictional company handbook. All its facts are invented, so the bot can only answer from the file. Try:

- *Who founded Nova Coffee, and in what year?*
- *How many shops does the company have?*
- *What was the net profit in 2024?*
- *What are the three NovaPoints membership levels?*
- *Who owns the Tokyo branch?* (not in the PDF, so the bot should say it does not know)

---

## 📁 Project structure

```
rag-pdf-chatbot/
├── app.py                # The whole application
├── requirements.txt      # Python dependencies
├── nova_coffee_test.pdf  # Sample PDF for testing
└── .gitignore
```

---

## ⚙️ Configuration

You can tune the app by editing `app.py`:

| Setting | Where | Default | What it does |
|---|---|---|---|
| `chunk_size` | `build_vector_store` | `1000` | Size of each text chunk |
| `chunk_overlap` | `build_vector_store` | `200` | Characters shared between neighbouring chunks |
| `k` | `stream_answer` | `4` | Number of chunks sent to the LLM |
| `temperature` | `stream_answer` | `0.3` | Lower is more factual, higher is more creative |
| Model name | Sidebar box | `openai/gpt-oss-20b` | Which Groq model answers |

---

## 🩺 Troubleshooting

| Problem | Fix |
|---|---|
| `ModuleNotFoundError` | Run `pip install -r requirements.txt` again |
| `404 model_not_found` | The model was retired. Open [Groq's model list](https://console.groq.com/docs/models) and paste a current name in the sidebar model box |
| `429` / rate limit error | You reached the free usage limit. Wait a while or try another model |
| "No text could be extracted" | The PDF is a scan (images). Use an OCR tool first |
| `streamlit` is not recognized | Use `python -m streamlit run app.py` |

---

## ⚠️ Limitations

- Reads **text only**. Scanned PDFs, images and charts are ignored.
- Each question is answered **independently**; the bot does not remember earlier questions when answering.
- Works best for **finding specific facts**, not for summarizing a whole book.
- The vector store lives in memory, so it is lost when you refresh the page.
- Relevant text chunks are sent to Groq's servers, so **do not upload confidential documents**.

---

## 🗺️ Ideas for the future

- [ ] Conversation memory for follow-up questions
- [ ] Show the source page for every answer
- [ ] Support Word and text files
- [ ] OCR for scanned PDFs
- [ ] Save the vector store to disk
- [ ] Deploy online with Streamlit Community Cloud

---

## 👤 Author

**Omar Khaled**
GitHub: [@omer1738](https://github.com/omer1738)

If you found this project useful, please give it a ⭐

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).
