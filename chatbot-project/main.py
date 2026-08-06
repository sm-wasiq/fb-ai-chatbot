import os
import json
import chromadb
import requests
from fastapi import FastAPI, HTTPException, Request, Response, BackgroundTasks
from pydantic import BaseModel
from dotenv import load_dotenv
from chromadb.utils import embedding_functions
from groq import Groq

load_dotenv()

app = FastAPI(title="Production Ready AI Support Engine")

# Environment Variables
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
FB_PAGE_ACCESS_TOKEN = os.getenv("FB_PAGE_ACCESS_TOKEN", "YOUR_PAGE_TOKEN")
VERIFY_TOKEN = os.getenv("FB_VERIFY_TOKEN", "my_custom_secret_token")

# Initialize Groq Client & Local Embedding
# Initialize Groq Client & Lightweight Embedding
groq_client = Groq(api_key=GROQ_API_KEY)

# heavy sentence-transformers এর বদলে ChromaDB-এর ডিফল্ট হালকা এমবেডিং ব্যবহার
embedding_func = embedding_functions.DefaultEmbeddingFunction()

chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(
    name="business_kb",
    embedding_function=embedding_func
)

# Knowledge Base Synchronization on Server Start
@app.on_event("startup")
def startup_event():
    if os.path.exists("knowledge_base.json"):
        with open("knowledge_base.json", "r", encoding="utf-8") as f:
            data = json.load(f)
        
        documents, metadatas, ids = [], [], []
        for idx, item in enumerate(data):
            doc_text = f"প্রশ্ন: {item['question']} উত্তর: {item['answer']}"
            documents.append(doc_text)
            metadatas.append({"answer": item['answer']})
            ids.append(f"id_{idx}")

        collection.upsert(documents=documents, metadatas=metadatas, ids=ids)
        print("✅ Production Vector Knowledge Base synchronized successfully!")

# Core Intelligence Function with Semantic Cache
def get_ai_response(user_query: str) -> str:
    # 1. High Precision Vector Search
    results = collection.query(query_texts=[user_query], n_results=1)
    
    # Cache hit logic based on distance similarity threshold
    if results['distances'] and len(results['distances'][0]) > 0:
        distance = results['distances'][0][0]
        # Very close query matches return cached answers instantly (0 latency)
        if distance < 0.25:
            print("[CACHE HIT] Direct Answer Served")
            return results['metadatas'][0][0]['answer']

    retrieved_context = ""
    if results['documents'] and len(results['documents'][0]) > 0:
        retrieved_context = results['documents'][0][0]

    # 2. Strict Prompt Instruction to avoid Hallucinations
    system_prompt = f"""
    You are a professional, helpful customer service assistant for a business in Bangladesh.
    Strictly use the provided Context to answer the customer query accurately in natural Bangla or Banglish.
    If the context does not contain the answer, politely ask them to leave a phone number for human support.
    Do not invent fake prices, terms, or contact information.

    Context:
    {retrieved_context}
    """

    response = groq_client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_query}
        ],
        temperature=0.1
    )

    return response.choices[0].message.content

# Outbound Message Handler to Messenger Graph API
def send_fb_message(recipient_id: str, text: str):
    url = f"https://graph.facebook.com/v19.0/me/messages?access_token={FB_PAGE_ACCESS_TOKEN}"
    payload = {
        "recipient": {"id": recipient_id},
        "message": {"text": text}
    }
    headers = {"Content-Type": "application/json"}
    try:
        requests.post(url, json=payload, headers=headers, timeout=5)
    except Exception as e:
        print(f"Failed to deliver message via FB Graph API: {e}")

# Async Background Processor for Messenger Requests
def process_messenger_event(sender_id: str, user_text: str):
    bot_reply = get_ai_response(user_text)
    send_fb_message(sender_id, bot_reply)

# ----------------- ENDPOINTS -----------------

@app.get("/webhook")
async def verify_webhook(request: Request):
    """Facebook Webhook Authentication Verification Endpoint"""
    params = request.query_params
    mode = params.get("hub.mode")
    token = params.get("hub.verify_token")
    challenge = params.get("hub.challenge")

    if mode == "subscribe" and token == VERIFY_TOKEN:
        return Response(content=challenge, status_code=200)
    return Response(status_code=403)

@app.post("/webhook")
async def handle_messenger_payload(request: Request, background_tasks: BackgroundTasks):
    """Event Receiver for Facebook Messenger Events"""
    data = await request.json()

    if data.get("object") == "page":
        for entry in data.get("entry", []):
            for messaging_event in entry.get("messaging", []):
                sender_id = messaging_event.get("sender", {}).get("id")
                
                if messaging_event.get("message") and "text" in messaging_event["message"]:
                    user_text = messaging_event["message"]["text"]
                    # Offload response generation to background task to respond to Facebook instantly (<1s)
                    background_tasks.add_task(process_messenger_event, sender_id, user_text)

    return Response(content="EVENT_RECEIVED", status_code=200)