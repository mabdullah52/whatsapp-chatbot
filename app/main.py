from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import chromadb
from app.rag import query_rag, get_embedding

app = FastAPI()
app.mount("/static", StaticFiles(directory="app/static"), name="static")

@app.get("/")
async def root():
    return FileResponse("app/static/index.html")
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection("business_docs")

class Message(BaseModel):
    sender: str
    text: str

@app.post("/webhook")
async def receive_message(message: Message):
    reply = query_rag(message.text, collection)
    return {"reply": reply, "sender": message.sender}

@app.post("/upload")
async def upload_doc(file: UploadFile = File(...)):
    content = await file.read()
    
    if file.filename.lower().endswith(".pdf"):
        import fitz
        pdf = fitz.open(stream=content, filetype="pdf")
        text = ""
        for page in pdf:
            text += page.get_text()
    else:
        text = content.decode("utf-8")
    
    if not text.strip():
        return {"error": "Could not extract text from file"}
    
    chunks = [text[i:i+500] for i in range(0, len(text), 500)]
    ids = [f"chunk_{i}" for i in range(len(chunks))]
    embeddings = [get_embedding(chunk) for chunk in chunks]
    collection.add(documents=chunks, ids=ids, embeddings=embeddings)
    return {"message": f"Uploaded {len(chunks)} chunks from {file.filename}"}