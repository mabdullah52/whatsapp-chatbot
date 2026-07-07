from fastapi import Request
from fastapi.responses import PlainTextResponse

VERIFY_TOKEN = "umme_shafiqa_verify_123"

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
    try:
        entry = body["entry"][0]
        changes = entry["changes"][0]
        value = changes["value"]
        message = value["messages"][0]
        sender = message["from"]
        text = message["text"]["body"]
        reply = query_rag(text, collection)
        return {"status": "ok", "reply": reply}
    except Exception as e:
        return {"status": "ok"}