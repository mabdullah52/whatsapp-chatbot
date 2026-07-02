import os
from groq import Groq

client_groq = Groq(api_key=os.environ.get("GROQ_API_KEY"))

def get_embedding(text: str) -> list:
    # Use a simple hash-based embedding as fallback
    # since Groq doesn't provide embeddings API
    import hashlib
    hash_val = int(hashlib.md5(text.encode()).hexdigest(), 16)
    import random
    random.seed(hash_val)
    return [random.uniform(-1, 1) for _ in range(384)]

def query_rag(question: str, collection=None) -> str:
    context = ""
    if collection:
        try:
            embedding = get_embedding(question)
            results = collection.query(
                query_embeddings=[embedding],
                n_results=3
            )
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