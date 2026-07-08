import os
import uuid
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

from rag import RAGSystem

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

app = FastAPI(title="Conversational Multi-Doc RAG API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_methods=["*"],
    allow_headers=["*"],
)


sessions: dict[str, RAGSystem] = {}


class QueryRequest(BaseModel):
    session_id: str
    question: str


class ResetRequest(BaseModel):
    session_id: str


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/upload")
async def upload_documents(files: list[UploadFile] = File(...)):
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded.")

    session_id = str(uuid.uuid4())
    rag = RAGSystem(groq_api_key=GROQ_API_KEY)

    file_data = []
    for f in files:
        if not f.filename.lower().endswith(".pdf"):
            raise HTTPException(status_code=400, detail=f"{f.filename} is not a PDF.")
        raw_bytes = await f.read()
        file_data.append((f.filename, raw_bytes))

    num_chunks = rag.add_documents(file_data)
    if num_chunks == 0:
        raise HTTPException(status_code=400, detail="No extractable text found in the uploaded PDFs.")

    sessions[session_id] = rag

    return {
        "session_id": session_id,
        "files_processed": [f.filename for f in files],
        "chunks_indexed": num_chunks,
    }


@app.post("/query")
def query(req: QueryRequest):
    rag = sessions.get(req.session_id)
    if rag is None:
        raise HTTPException(status_code=404, detail="Session not found. Upload documents first via /upload.")

    result = rag.query(req.question)
    return result


@app.post("/reset")
def reset(req: ResetRequest):
    rag = sessions.get(req.session_id)
    if rag is None:
        raise HTTPException(status_code=404, detail="Session not found.")

    rag.reset()
    return {"status": "conversation history cleared"}


@app.delete("/session/{session_id}")
def delete_session(session_id: str):
    if session_id in sessions:
        del sessions[session_id]
        return {"status": "session deleted"}
    raise HTTPException(status_code=404, detail="Session not found.")
