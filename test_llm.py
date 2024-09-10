from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from dotenv import load_dotenv
import os

# Laad de .env variabelen
load_dotenv()

# Haal de API-key op
openai_api_key = os.getenv("OPENAI_API_KEY")

# Deze functie demonstreert het gebruik van een LLM (Language Model) in een conversatie
def test_llm():
    # Initialiseer het ChatOpenAI model met de OpenAI API key
    # GPT-4 wordt gebruikt met een temperatuur van 0.7 voor enige creativiteit
    chat_model = ChatOpenAI(openai_api_key=openai_api_key, model_name="gpt-4", temperature=0.7)

    # Maak een geheugen object om de conversatie bij te houden
    memory = ConversationBufferMemory()

    # Creëer een ConversationChain die het model en geheugen combineert
    # verbose=True zorgt ervoor dat we de interne stappen kunnen zien
    conversation = ConversationChain(
        llm=chat_model,
        memory=memory,
        verbose=True
    )

    # Voer een reeks vragen uit en toon de antwoorden
    # Vraag 1: Introductie
    user_question = "Ik ben Thomas"
    response = conversation.predict(input=user_question)
    print(f"Gebruiker: {user_question}")
    print(f"Assistent: {response}")

    # Vraag 2: Vraag over woonplaats
    user_question = "Woon in Amsterdam?"
    response = conversation.predict(input=user_question)
    print(f"Gebruiker: {user_question}")
    print(f"Assistent: {response}")
    
    # Vraag 3: Persoonlijke voorkeur
    user_question = "Ik houd van pizza"
    response = conversation.predict(input=user_question)
    print(f"Gebruiker: {user_question}")
    print(f"Assistent: {response}")
    
    # Deze herhaling lijkt onbedoeld, maar wordt behouden zoals gevraagd
    print(f"Gebruiker: {user_question}")
    print(f"Assistent: {response}")
    
    # Vraag 4: Samenvatting van de conversatie
    user_question = "Wat hebben we besproken"
    response = conversation.predict(input=user_question)
    print(f"Gebruiker: {user_question}")
    print(f"Assistent: {response}")

    # Geef de volledige conversatiegeschiedenis terug
    return conversation.memory.chat_memory.messages

# Dit blok wordt uitgevoerd als het script direct wordt gerund
if __name__ == "__main__":
    # Voer de test_llm functie uit en sla het resultaat op
    conversation = test_llm()
    
    # Print de volledige conversatie, inclusief zowel gebruiker als assistent berichten
    print("\nVolledige conversatie:")
    for message in conversation:
        print(f"{message.type.capitalize()}: {message.content}")