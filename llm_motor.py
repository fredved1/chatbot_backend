from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, AIMessage
from langchain.memory import ConversationBufferMemory
from typing import List, Dict
from templates.botlease_agent import create_botlease_agent

def create_chat_model(api_key: str, model: str = "gpt-4o", temperature: float = 0.7):
    return ChatOpenAI(
        openai_api_key=api_key,
        model=model,
        temperature=temperature
    )

class LLMMotor:
    def __init__(self, api_key: str, model: str = "gpt-4o", temperature: float = 0.7):
        self.chat_model = create_chat_model(api_key, model, temperature)
        self.botlease_agent = create_botlease_agent(self.chat_model)
        self.memory = ConversationBufferMemory(return_messages=True)

    def generate_response(self, prompt: str) -> str:
        self.memory.chat_memory.add_user_message(prompt)
        response = self.botlease_agent.invoke({"messages": self.memory.chat_memory.messages})
        self.memory.chat_memory.add_ai_message(response.content)
        return response.content

    def get_chat_history(self) -> List[Dict[str, str]]:
        return [
            {"role": "user" if isinstance(msg, HumanMessage) else "assistant", "content": msg.content}
            for msg in self.memory.chat_memory.messages
        ]

    def clear_memory(self):
        self.memory.clear()

    def start_new_conversation(self) -> str:
        self.clear_memory()
<<<<<<< HEAD
        opening_message = "Hallo! Hoe kan ik u vandaag helpen met vragen over Bot Lease?"
=======
        opening_message = "Hallo en welkom bij Bot Lease! Heb je vragen stel ze gerust."
>>>>>>> 988d5dcfb7b51161e728f46a3751c54ae40b2453
        self.memory.chat_memory.add_ai_message(opening_message)
        return opening_message

def get_available_models(api_key: str) -> List[str]:
    from openai import OpenAI
    client = OpenAI(api_key=api_key)
    models = client.models.list()
    return [model.id for model in models.data if model.id.startswith("gpt")]
