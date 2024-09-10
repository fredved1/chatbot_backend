from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, AIMessage
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

# Genereert een antwoord met behulp van het ChatOpenAI model en de UWV-agent
# Parameters:
#   chat_model: ChatOpenAI model instantie
#   messages: Lijst van berichtwoordenboeken
# Retourneert: Gegenereerd antwoord als een string
def generate_response(chat_model: ChatOpenAI, messages: List[Dict[str, str]]) -> str:
    uwv_agent = create_uwv_agent(chat_model)
    # Zet berichten om naar het juiste formaat voor LangChain
    langchain_messages = [
        HumanMessage(content=msg["content"]) if msg["role"] == "user" else AIMessage(content=msg["content"])
        for msg in messages[1:]  # Sla het eerste assistent-bericht over
    ]
    response = uwv_agent.invoke({"messages": langchain_messages})
    return response.content

# Haalt een lijst op van beschikbare GPT-modellen van OpenAI
# Parameters:
#   api_key: OpenAI API sleutel
# Retourneert: Lijst van beschikbare modelnamen
def get_available_models(api_key: str) -> List[str]:
    from openai import OpenAI
    client = OpenAI(api_key=api_key)
    models = client.models.list()
    return [model.id for model in models.data if model.id.startswith("gpt")]