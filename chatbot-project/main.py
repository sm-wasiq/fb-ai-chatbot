import os
import re
import json
import requests
from fastapi import FastAPI, Request, Response
from groq import Groq, RateLimitError
from dotenv import load_dotenv

# Load environment variables from .env file if present
load_dotenv()

app = FastAPI()

# Groq Client Initialization
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# Load Knowledge Base JSON File
def load_knowledge_base():
    try:
        file_path = "knowledge_base.json"
        if not os.path.exists(file_path):
            file_path = "chatbot-project/knowledge_base.json"
            
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading knowledge base: {e}")
        return {}

KNOWLEDGE_BASE = load_knowledge_base()

# Point-to-Point High Conversion System Prompt Generator
def get_system_prompt(page_id: str):
    shop = KNOWLEDGE_BASE.get(page_id, {
        "shop_name": "Zoré Fashion",
        "location": "মিরপুর ১, ঢাকা (ডিসপ্লে সেন্টার রয়েছে)। সারা বাংলাদেশে হোম ডেলিভারি দেওয়া হয়।",
        "delivery_charge": "ঢাকার ভেতরে ৮০ টাকা (১-২ কার্যদিবস), ঢাকার বাইরে ১২০ টাকা (২-৩ কার্যদিবস)।",
        "payment_method": "ক্যাশ অন ডেলিভারি (পণ্য হাতে পেয়ে পেমেন্ট সুবিধা) এবং বিকাশ/নগদ প্রযোজ্য।",
        "size_guide": "M (বুক ৩৮ ইঞ্চি), L (বুক ৪০ ইঞ্চি), XL (বুক ৪২ ইঞ্চি), XXL (বুক ৪৪ ইঞ্চি) এভেইলেবল রয়েছে।",
        "return_exchange": "ডেলিভারি ম্যানের সামনে চেক করে নেওয়ার সুবিধা আছে। এছাড়া ৩ দিনের মধ্যে সাইজ এক্সচেঞ্জ পলিসি রয়েছে।",
        "order_process": "অর্ডার কনফার্ম করতে প্রোডাক্টের ছবি বা সাইজ, আপনার নাম, পূর্ণাঙ্গ ঠিকানা এবং মোবাইল নম্বর ইনবক্সে পাঠিয়ে দিন।"
    })

    return f"""
You are the official AI sales & customer support representative for "{shop.get('shop_name')}".
Your objective is to answer questions accurately and convert prospects into happy buyers with polite, friendly, and helpful responses.

[KNOWLEDGE BASE]
- Shop Name: {shop.get('shop_name')}
- Location / Outlet: {shop.get('location')}
- Delivery Charge & Delivery Time: {shop.get('delivery_charge')}
- Payment Methods: {shop.get('payment_method')}
- Available Sizes & Measurements: {shop.get('size_guide')}
- Return & Exchange Policy: {shop.get('return_exchange')}
- How to Order / Customer Conversion: {shop.get('order_process')}

[RULES FOR RESPONDING]
1. Language Handling:
   - English input (e.g. "Do you have size L available? What is the chest size?"): Reply in fluent, polite English. Example: "Yes! Size L is available with a 40-inch chest measurement. To place an order, please send us your name, address, and phone number here."
   - Bangla or Banglish input (e.g. "order kivabe korbo?", "size L ache?"): Reply in polite, natural, conversational Bengali (বাংলা).
2. High-Conversion Friendly Tone:
   - Always confirm product/size availability positively according to the size guide.
   - When asked how to order or showing buying intent, clearly guide them to drop their Name, Delivery Address, Phone Number, and desired size/product right here in the chat, or reach out via call/WhatsApp.
3. Keep answers concise, clear, and direct (1-3 sentences maximum).
4. Do NOT prefix sentences with bullet dashes or hyphens (`- `). Write natural conversational text.
5. Never state that size or store information is missing when it is listed in the knowledge base.
"""

# Model Configuration from Environment
PRIMARY_MODEL = os.getenv("GROQ_MODEL", "groq/compound-mini")
FALLBACK_MODEL = os.getenv("GROQ_FALLBACK_MODEL", "qwen/qwen3.8-27b")

def clean_reply_text(text: str) -> str:
    if not text:
        return ""
    # Strip <think>...</think> reasoning blocks from Qwen / reasoning models
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    # Strip any leading hyphens or bullet asterisks from lines if present
    lines = [re.sub(r"^[-*•\s]+\s*", "", line).strip() for line in text.split("\n") if line.strip()]
    return " ".join(lines).strip()

def generate_ai_reply(system_prompt: str, user_message: str) -> str:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message}
    ]
    # 1. Try Primary Model
    try:
        chat_completion = client.chat.completions.create(
            messages=messages,
            model=PRIMARY_MODEL,
        )
        raw_reply = chat_completion.choices[0].message.content
        return clean_reply_text(raw_reply)
    except Exception as primary_err:
        print(f"WARNING: Primary model ({PRIMARY_MODEL}) failed: {primary_err}")
        
        # 2. Try Fallback Backup Model if configured
        if FALLBACK_MODEL and FALLBACK_MODEL != PRIMARY_MODEL:
            try:
                print(f"DEBUG: Retrying with fallback model ({FALLBACK_MODEL})...")
                chat_completion = client.chat.completions.create(
                    messages=messages,
                    model=FALLBACK_MODEL,
                )
                raw_reply = chat_completion.choices[0].message.content
                return clean_reply_text(raw_reply)
            except Exception as fallback_err:
                print(f"ERROR: Fallback model ({FALLBACK_MODEL}) also failed: {fallback_err}")
        
        # Raise original exception if all models fail
        raise primary_err

# Webhook Verification
@app.get("/webhook")
async def verify_webhook(request: Request):
    params = request.query_params
    mode = params.get("hub.mode")
    token = params.get("hub.verify_token")
    challenge = params.get("hub.challenge")

    if mode and token:
        if mode == "subscribe" and token == os.getenv("VERIFY_TOKEN"):
            return Response(content=challenge, status_code=200)
        return Response(content="Verification failed", status_code=403)
    return Response(content="Bad Request", status_code=400)

# Message deduplication cache
PROCESSED_MESSAGES = set()

# Webhook Event Handling
@app.post("/webhook")
async def handle_webhook(request: Request):
    data = await request.json()
    print(f"DEBUG: Webhook payload received: {data}")

    if data.get("object") == "page":
        for entry in data.get("entry", []):
            page_id = str(entry.get("id"))
            for messaging_event in entry.get("messaging", []):
                message = messaging_event.get("message")
                if message:
                    # 1. Ignore echo messages sent by the page/bot itself
                    if message.get("is_echo"):
                        print(f"DEBUG: Ignoring echo message: {message.get('mid')}")
                        continue

                    # 2. Ignore duplicate message retries from Meta
                    mid = message.get("mid")
                    if mid:
                        if mid in PROCESSED_MESSAGES:
                            print(f"DEBUG: Ignoring duplicate mid: {mid}")
                            continue
                        PROCESSED_MESSAGES.add(mid)
                        if len(PROCESSED_MESSAGES) > 1000:
                            PROCESSED_MESSAGES.clear()

                    if "text" in message:
                        sender_id = messaging_event["sender"]["id"]
                        user_message = message["text"]
                        print(f"DEBUG: Received message '{user_message}' from sender {sender_id} for page {page_id}")

                        try:
                            # 1. Generate System Prompt according to Page ID
                            system_prompt = get_system_prompt(page_id)

                            # 2. Generate Reply using Groq (Primary or Backup)
                            bot_reply = generate_ai_reply(system_prompt, user_message)
                            print(f"DEBUG: Groq reply generated: '{bot_reply}'")

                            # 3. Send Reply to Messenger
                            send_message(sender_id, bot_reply, page_id)
                        except RateLimitError as e:
                            print(f"ERROR: Groq Rate limit reached: {e}")
                            fallback_reply = "ধন্যবাদ মেসেজ করার জন্য! বর্তমানে অনেক কাস্টমারের ইনকোয়ারি থাকায় সামান্য বিলম্ব হচ্ছে, অনুগ্রহ করে ১ মিনিট পর আবার চেষ্টা করুন।"
                            send_message(sender_id, fallback_reply, page_id)
                        except Exception as e:
                            print(f"ERROR: Exception while processing message: {e}")
                            emergency_reply = "ধন্যবাদ আপনার বার্তার জন্য! আমাদের একজন প্রতিনিধি খুব শীঘ্রই আপনার সাথে যোগাযোগ করবেন।"
                            send_message(sender_id, emergency_reply, page_id)

        return Response(content="EVENT_RECEIVED", status_code=200)
    return Response(content="Not Found", status_code=404)

# Function to Send Message back to Meta Messenger
def send_message(recipient_id: str, message_text: str, page_id: str):
    # Dynamic Token Selection based on Page ID from .env
    access_token = os.getenv(f"PAGE_TOKEN_{page_id}")
    
    if not access_token:
        access_token = os.getenv("FB_PAGE_ACCESS_TOKEN")

    if not access_token:
        print(f"ERROR: No Page Access Token found for Page ID {page_id} or FB_PAGE_ACCESS_TOKEN!")

    url = f"https://graph.facebook.com/v20.0/me/messages?access_token={access_token}"
    payload = {
        "recipient": {"id": recipient_id},
        "message": {"text": message_text}
    }
    headers = {"Content-Type": "application/json"}

    response = requests.post(url, json=payload, headers=headers)
    print(f"DEBUG: Send message to {recipient_id}, status: {response.status_code}, response: {response.text}")