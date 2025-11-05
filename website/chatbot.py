from flask import Blueprint, request, jsonify, session
import os
from dotenv import load_dotenv
from openai import OpenAI
import httpx

load_dotenv()

chatbot = Blueprint("chatbot", __name__)

HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    raise ValueError("HF_TOKEN not found. Add to your environment variables or .env file.")

httpx_client = httpx.Client(verify=False)

client = OpenAI(
    api_key=HF_TOKEN,
    base_url="https://router.huggingface.co/v1",
    http_client=httpx_client
)

@chatbot.route("/ask", methods=["POST"])
def ask():
    """Handle user question and generate AI response with chat history."""
    user_question = request.form.get("question", "").strip()
    if not user_question:
        return jsonify({"answer": "No question provided."}), 400

    # Retrieve chat history from session
    history = session.get("chat_history", [])

    # Append user message to history
    history.append({"role": "user", "content": user_question})

    try:
        # Send full conversation history to the API
        completion = client.chat.completions.create(
            model="Qwen/Qwen2.5-VL-7B-Instruct:hyperbolic",
            messages=history
        )
        bot_reply = completion.choices[0].message.content
    except Exception as e:
        bot_reply = f"Error: {e}"

    # Append assistant response to history
    history.append({"role": "assistant", "content": bot_reply})

    # Save updated history in session
    session["chat_history"] = history

    return jsonify({"answer": bot_reply})

@chatbot.route("/reset", methods=["POST"])
def reset():
    """Clear chat session memory."""
    session.pop("chat_history", None)
    return jsonify({"status": "reset successful"})
