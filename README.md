# Conversational Multi-Doc RAG API

Upload multiple PDFs, ask questions about them in natural conversation (with memory), get answers grounded in your documents with source citations (file + page number).

## Architecture

```
Client (frontend/index.html, curl, or your own app)
   │
   ├── POST /upload   → uploads PDFs, returns a session_id
   ├── POST /query    → asks a question within a session (uses last 5 turns of history)
   ├── POST /reset    → clears conversation memory, keeps documents
   └── GET  /health
```

Each session is isolated: a `RAGSystem` instance holds its own FAISS index and its own bounded conversation history (last 5 Q&A turns — old turns are dropped so the prompt doesn't grow unbounded).

## Stack
- **FastAPI** — API layer
- **LangChain** — document loading, chunking, retrieval orchestration
- **FAISS** — in-memory vector index (per session)
- **HuggingFace `all-MiniLM-L6-v2`** — free, local embeddings (no API key needed for embeddings)
- **Groq (`llama-3.1-8b-instant`)** — free, fast LLM inference for generation

## Local setup

```bash
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env
# edit .env and add your GROQ_API_KEY (free at https://console.groq.com)

uvicorn main:app --reload
```

API will be live at `http://localhost:8000`. Interactive docs at `http://localhost:8000/docs`.

Open `frontend/index.html` directly in your browser (no server needed) to test with a UI instead of curl.

## Example usage (curl)

```bash
# 1. Upload PDFs, get a session_id
curl -X POST http://localhost:8000/upload \
  -F "files=@course_chapter1.pdf" \
  -F "files=@course_chapter2.pdf"

# → {"session_id": "abc-123", "files_processed": [...], "chunks_indexed": 42}

# 2. Ask a question
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"session_id": "abc-123", "question": "What is explained in chapter 1?"}'

# 3. Follow-up question (memory kicks in)
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"session_id": "abc-123", "question": "Can you go deeper on that?"}'

# 4. Reset conversation (keeps documents)
curl -X POST http://localhost:8000/reset \
  -H "Content-Type: application/json" \
  -d '{"session_id": "abc-123"}'
```

## Deploying to Render

1. Push this project to a GitHub repo.
2. On [render.com](https://render.com): **New → Web Service** → connect the repo.
3. Build command: `pip install -r requirements.txt`
4. Start command: `uvicorn main:app --host 0.0.0.0 --port $PORT`
5. In the **Environment** tab, add `GROQ_API_KEY` with your key (never commit it).
6. Deploy. Free tier will cold-start after ~15 min of inactivity — expect the first request after idle to take 20-30s.

Once deployed, update the API base URL in `frontend/index.html` to your Render URL, and host that file on GitHub Pages for a free, clickable demo link.

## Known limitations (by design, for a portfolio project — not hidden bugs)

- **Sessions live in memory only.** A server restart (e.g., Render free tier sleeping) wipes all active sessions. For production this would move to Redis or a database — noted here deliberately rather than solved, to keep the project scoped.
- **No relevance threshold tuning.** The `SCORE_THRESHOLD` in `rag.py` (`similarity_search_with_score`, drops chunks with L2 distance above 1.0) is a reasonable default for MiniLM embeddings, but the right value depends on your documents — tune it against your own data if answers seem to ignore relevant content or include irrelevant chunks.
- **No auth.** Anyone with a `session_id` can query that session. Fine for a demo; add API keys/auth before treating this as more than a portfolio piece.

## Possible extensions
- RAGAS evaluation script comparing chunk sizes or retrieval strategies (see the "RAG + eval report" idea — pairs naturally with this project).
- Corrective retrieval: if the top chunks aren't relevant, rewrite the query once and retry before falling back to "not found."
- Streaming responses (`llm.stream()` instead of `.invoke()`) for better perceived latency.
