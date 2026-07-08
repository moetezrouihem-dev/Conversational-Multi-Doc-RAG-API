import io
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.documents import Document

_EMBEDDING_MODEL = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

MAX_HISTORY_TURNS = 5  # bounded memory: only the last N (question, answer) pairs are kept


class RAGSystem:
    def __init__(self, groq_model: str = "llama-3.1-8b-instant", groq_api_key: str | None = None):
        self.embedding_model = _EMBEDDING_MODEL
        self.llm = ChatGroq(model=groq_model, temperature=0.3, api_key=groq_api_key)
        self.db: FAISS | None = None
        self.chat_history: list[tuple[str, str]] = []  # [(question, answer), ...]

    

    def add_documents(self, files: list[tuple[str, bytes]]) -> int:
        """
        files: list of (filename, raw_pdf_bytes)
        Extracts text, chunks it, embeds it, and either creates or extends the FAISS index.
        Returns the number of chunks added.
        """
        docs: list[Document] = []

        for filename, raw_bytes in files:
            reader = PdfReader(io.BytesIO(raw_bytes))
            for page_num, page in enumerate(reader.pages, start=1):
                text = page.extract_text() or ""
                if not text.strip():
                    continue
                # metadata is what makes source citations possible later
                docs.append(Document(page_content=text, metadata={"source": filename, "page": page_num}))

        if not docs:
            return 0

        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = splitter.split_documents(docs)

        if self.db is None:
            self.db = FAISS.from_documents(chunks, self.embedding_model)
        else:
            self.db.add_documents(chunks)  # extend existing index instead of rebuilding from scratch

        return len(chunks)

    

    def query(self, question: str, k: int = 4) -> dict:
        if self.db is None:
            return {
                "answer": "No documents have been uploaded yet for this session. Upload PDFs first.",
                "sources": [],
            }

        
        results = self.db.similarity_search_with_score(question, k=k)

        
        
        SCORE_THRESHOLD = 2.2
        relevant = [(doc, score) for doc, score in results if score <= SCORE_THRESHOLD]

        if not relevant:
            answer = "I couldn't find anything relevant to that in the uploaded documents."
            self._update_history(question, answer)
            return {"answer": answer, "sources": []}

        context = "\n\n".join(doc.page_content for doc, _ in relevant)
        sources = [
            {"file": doc.metadata.get("source"), "page": doc.metadata.get("page")}
            for doc, _ in relevant
        ]

        history_str = "\n".join(
            f"Q: {q}\nA: {a}" for q, a in self.chat_history[-MAX_HISTORY_TURNS:]
        )

        prompt = f"""You are a helpful assistant answering questions strictly based on the provided document excerpts. \
If the excerpts don't contain the answer, say so honestly instead of guessing.

Conversation history (for context on follow-up questions):
{history_str if history_str else "(no previous turns)"}

Document excerpts:
{context}

Current question: {question}

Answer:"""

        response = self.llm.invoke(prompt)
        answer = response.content

        self._update_history(question, answer)

        return {"answer": answer, "sources": sources}

    def _update_history(self, question: str, answer: str):
        self.chat_history.append((question, answer))
        
        
        if len(self.chat_history) > MAX_HISTORY_TURNS:
            self.chat_history = self.chat_history[-MAX_HISTORY_TURNS:]

    def reset(self):
        self.chat_history = []
