from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
import chromadb
from app.rag import query_rag, get_embedding

app = FastAPI()

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
    text = content.decode("utf-8")
    chunks = [text[i:i+500] for i in range(0, len(text), 500)]
    ids = [f"chunk_{i}" for i in range(len(chunks))]
    embeddings = [get_embedding(chunk) for chunk in chunks]
    collection.add(documents=chunks, ids=ids, embeddings=embeddings)
    return {"message": f"Uploaded {len(chunks)} chunks"}