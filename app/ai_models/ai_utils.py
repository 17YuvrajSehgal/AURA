from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()
open_ai_chat_model = ChatOpenAI(
    model="gpt-4o",
    temperature=0,
)
