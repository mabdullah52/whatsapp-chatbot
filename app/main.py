import os
import httpx
from fastapi import FastAPI, UploadFile, File, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, PlainTextResponse
import chromadb
from app.rag import query_rag, get_embedding, send_whatsapp_reply

app = FastAPI()

app.mount("/static", StaticFiles(directory="app/static"), name="static")

chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection("business_docs")

VERIFY_TOKEN = "umme_shafiqa_verify_123"

@app.get("/")
async def root():
    return FileResponse("app/static/index.html")

@app.get("/webhook")
async def verify_webhook(request: Request):
    mode = request.query_params.get("hub.mode")
    token = request.query_params.get("hub.verify_token")
    challenge = request.query_params.get("hub.challenge")
    if mode == "subscribe" and token == VERIFY_TOKEN:
        return PlainTextResponse(content=challenge)
    return PlainTextResponse(content="Forbidden", status_code=403)

@app.post("/webhook")
async def receive_message(request: Request):
    body = await request.json()
    print("INCOMING PAYLOAD:", body, flush=True)
    try:
        entry = body["entry"][0]
        changes = entry["changes"][0]
        value = changes["value"]
        if "messages" not in value:
            print("NO MESSAGES KEY - likely a status update, ignoring", flush=True)
            return {"status": "ok"}
        message = value["messages"][0]
        sender = message["from"]
        text = message["text"]["body"]
        phone_number_id = value["metadata"]["phone_number_id"]
        reply = query_rag(text, collection)
        result = send_whatsapp_reply(
            to=sender,
            message=reply,
            phone_number_id=phone_number_id,
            access_token=os.environ.get("WHATSAPP_ACCESS_TOKEN")
        )
        print("SEND RESULT:", result, flush=True)
        return {"status": "ok"}
    except Exception as e:
        import traceback
        print("WEBHOOK ERROR:", e, flush=True)
        traceback.print_exc()
        return {"status": "ok"}

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