from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, AIMessage
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from typing import List, Dict
from templates.uwv_agent import create_uwv_agent

# Creëert een ChatOpenAI model instantie
# Parameters:
#   api_key: OpenAI API sleutel
#   model: Te gebruiken model (standaard: "gpt-3.5-turbo")
#   temperature: Bepaalt willekeurigheid (0.0-2.0, standaard: 0.7)
# Retourneert: ChatOpenAI instantie
def create_chat_model(api_key: str, model: str = "gpt-4o", temperature: float = 0.7):
    return ChatOpenAI(
        openai_api_key=api_key,
        model=model,
        temperature=temperature
    )

class LLMMotor:
    def __init__(self, api_key: str, model: str = "gpt-4o", temperature: float = 0.7):
        self.chat_model = create_chat_model(api_key, model, temperature)
        self.uwv_agent = create_uwv_agent(self.chat_model)
        self.memory = ConversationBufferMemory(return_messages=True)

    def generate_response(self, input_message: str) -> str:
        # Voeg het nieuwe bericht toe aan het geheugen
        self.memory.chat_memory.add_user_message(input_message)

        # Haal de volledige geschiedenis op, inclusief het nieuwe bericht
        history = self.memory.chat_memory.messages

        # Gebruik de UWV-agent met de volledige geschiedenis
        response = self.uwv_agent.invoke({"messages": history})
        
        # Voeg het antwoord toe aan het geheugen
        self.memory.chat_memory.add_ai_message(response.content)

        return response.content



    # Opmerking: Als je de chatgeschiedenis nodig hebt, 
    # kun je direct toegang krijgen tot:
    # self.memory.chat_memory.messages
    
    def reset_memory(self):
        self.memory.clear()

# Haalt een lijst op van beschikbare GPT-modellen van OpenAI
# Parameters:
#   api_key: OpenAI API sleutel
# Retourneert: Lijst van beschikbare modelnamen
def get_available_models(api_key: str) -> List[str]:
    from openai import OpenAI
    client = OpenAI(api_key=api_key)
    models = client.models.list()
    return [model.id for model in models.data if model.id.startswith("gpt")]
