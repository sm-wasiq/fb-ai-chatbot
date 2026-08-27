import os
import json
import requests
from fastapi import FastAPI, Request, Response
from groq import Groq, RateLimitError

# Load environment variables from .env file if present
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

# Groq Client Initialization
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# Load Knowledge Base JSON File
def load_knowledge_base():
    try:
        # chatbot-project ফোল্ডারের ভেতর ফাইল থাকলে পাথ অনুযায়ী রিড করবে
        file_path = "knowledge_base.json"
        if not os.path.exists(file_path):
            file_path = "chatbot-project/knowledge_base.json"
            
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading knowledge base: {e}")
        return {}

KNOWLEDGE_BASE = load_knowledge_base()

# Point-to-Point System Prompt Generator
def get_system_prompt(page_id: str):
    shop = KNOWLEDGE_BASE.get(page_id, {
        "shop_name": "আমাদের শপ",
        "location": "অনলাইন সার্ভিস",
        "delivery_charge": "ঢাকার ভেতরে ৮০ টাকা, বাইরে ১৫০ টাকা।",
        "payment_method": "ক্যাশ অন ডেলিভারি।",
        "return_policy": "প্রোডাক্টে সমস্যা থাকলে ৩ দিনের মধ্যে জানান।"
    })

    return f"""
তুমি "{shop.get('shop_name')}"-এর একজন প্রফেশনাল এবং পয়েন্ট-টু-পয়েন্ট কাস্টমার সাপোর্ট এজেন্ট।

[KNOWLEDGE BASE]
- শপের নাম: {shop.get('shop_name')}
- লোকেশন: {shop.get('location')}
- ডেলিভারি চার্জ: {shop.get('delivery_charge')}
- পেমেন্ট পদ্ধতি: {shop.get('payment_method')}
- রিটার্ন পলিসি: {shop.get('return_policy')}

[STRICT RULES FOR REPLYING]
১. কাস্টমার ঠিক যতটুকু জানতে চেয়েছে, ঠিক ততটুকুর উত্তর ১ থেকে ২ লাইনে পয়েন্ট আকারে দেবে।
২. অপ্রাসঙ্গিক তথ্য দেওয়া সম্পূর্ণ নিষেধ (যেমন: ডেলিভারি চার্জ জানতে চাইলে লোকেশন বা পেমেন্ট নিয়ে কিছু বলবে না)।
৩. নলেজ বেসে উত্তর থাকলে ভুলেও "হিউম্যান সাপোর্টে যোগাযোগ করুন" জাতীয় কথা বলবে না। সরাসরি সঠিক তথ্য জানিয়ে দেবে।
৪. কোনো অতিরিক্ত ভূমিকা বা ভূমিকা-মূলক বাক্য (যেমন: "আমাদের শপের লোকেশন হলো...", "আমি আপনাকে জানাতে পারি যে...") লেখা যাবে না।
৫. উত্তর সবসময় পয়েন্ট আকারে অথবা খুব সংক্ষেপে সহজ বাংলায় দেবে।
"""

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

                            # 2. Call Groq API
                            chat_completion = client.chat.completions.create(
                                messages=[
                                    {"role": "system", "content": system_prompt},
                                    {"role": "user", "content": user_message}
                                ],
                                model="llama-3.1-8b-instant"
                            )

                            bot_reply = chat_completion.choices[0].message.content
                            print(f"DEBUG: Groq reply generated: '{bot_reply}'")

                            # 3. Send Reply to Messenger
                            send_message(sender_id, bot_reply, page_id)
                        except RateLimitError as e:
                            print(f"ERROR: Groq Rate limit reached: {e}")
                            fallback_reply = "ধন্যবাদ মেসেজ করার জন্য! বর্তমানে অনেক কাস্টমারের ইনকোয়ারি থাকায় সামান্য বিলম্ব হচ্ছে, অনুগ্রহ করে ১ মিনিট পর আবার চেষ্টা করুন।"
                            send_message(sender_id, fallback_reply, page_id)
                        except Exception as e:
                            print(f"ERROR: Exception while processing message: {e}")

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