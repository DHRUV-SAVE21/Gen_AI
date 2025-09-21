# from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain_core.prompts import ChatPromptTemplate
# from langchain_core.chat_history import InMemoryChatMessageHistory
# from langchain_core.runnables.history import RunnableWithMessageHistory
# import os
# from dotenv import load_dotenv

# load_dotenv()
# api_key = os.getenv("GEMINI_API_KEY")

# llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", api_key=api_key)

# prompt = ChatPromptTemplate.from_messages(
#     [
#         ("system", "You are a helpful Medical Assistant Bot developed by MedTrack."),
#         ("human", "{input}"),
#     ]
# )

# chain = prompt | llm

# store = {}


# def get_session_history(session_id: str):
#     if session_id not in store:
#         store[session_id] = InMemoryChatMessageHistory()
#     return store[session_id]

# chat = RunnableWithMessageHistory(chain, get_session_history)

# session_id = "user1"
# while True:
#     user_input = input("You: ").strip()
#     if user_input.lower() in ["exit", "quit"]:
#         print("Bot: 👋 Goodbye!")
#         break

#     response = chat.invoke(
#         {"input": user_input}, config={"configurable": {"session_id": session_id}}
#     )
#     print("Bot:", response.content)


from flask import Blueprint, render_template, request, jsonify
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
import os
from dotenv import load_dotenv

load_dotenv()

# Load API key
api_key = os.getenv("GEMINI_API_KEY")

# Init model
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", api_key=api_key)

# Prompt
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a professional, intelligent, and highly reliable Career Assistant Bot developed by Team Data Pirates. Your goal is to guide users with accurate career insights, skill-building paths, and job recommendations in the best possible manner."),
        ("human", "{input}"),
    ]
)

chain = prompt | llm

# Memory store
store = {}


def get_session_history(session_id: str):
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]


chat = RunnableWithMessageHistory(chain, get_session_history)

# Blueprint
aibot_bp = Blueprint("aibot_bp", __name__)


@aibot_bp.route("/bot")
def bot():
    return render_template("bot_ui/aibot_ui.html")


@aibot_bp.route("/bot/chat", methods=["POST"])
def chat_route():
    data = request.get_json()
    user_input = data.get("message", "").strip()
    if not user_input:
        return jsonify({"response": "Please enter a valid message."}), 400

    session_id = "user1"  # can make dynamic later
    response = chat.invoke(
        {"input": user_input}, config={"configurable": {"session_id": session_id}}
    )

    return jsonify({"response": response.content})
