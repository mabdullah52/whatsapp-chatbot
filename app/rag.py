import os
import hashlib
import random
import httpx
from groq import Groq

client_groq = Groq(api_key=os.environ.get("GROQ_API_KEY"))

def get_embedding(text: str) -> list:
    hash_val = int(hashlib.md5(text.encode()).hexdigest(), 16)
    random.seed(hash_val)
    return [random.uniform(-1, 1) for _ in range(384)]

def query_rag(question: str, collection=None) -> str:
    context = ""
    if collection:
        try:
            embedding = get_embedding(question)
            results = collection.query(query_embeddings=[embedding], n_results=3)
            if results['documents'][0]:
                context = "\n".join(results['documents'][0])
        except:
            pass

    messages = [
        {"role": "system", "content": f"You are a helpful business assistant. Use this context to answer: {context}" if context else "You are a helpful business assistant."},
        {"role": "user", "content": question}
    ]

    response = client_groq.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=messages
    )
    return response.choices[0].message.content

def send_whatsapp_reply(to: str, message: str, phone_number_id: str, access_token: str):
    url = f"https://graph.facebook.com/v18.0/{phone_number_id}/messages"
    headers = {"Authorization": f"Bearer {access_token}", "Content-Type": "application/json"}
    payload = {"messaging_product": "whatsapp", "to": to, "type": "text", "text": {"body": message}}

    timeout = httpx.Timeout(connect=30.0, read=30.0, write=30.0, pool=30.0)
    last_error = None
    for attempt in range(3):
        try:
            resp = httpx.post(url, headers=headers, json=payload, timeout=timeout)
            print("WHATSAPP API STATUS:", resp.status_code, resp.text, flush=True)
            return resp.json()
        except httpx.ConnectTimeout as e:
            last_error = e
            print(f"ATTEMPT {attempt+1} TIMED OUT, retrying...", flush=True)
    print("ALL ATTEMPTS FAILED:", last_error, flush=True)
    return {"error": str(last_error)}