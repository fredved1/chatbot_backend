from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            " Je spreekt mensen altijd met hoofdletter U. Je bent een zeer gelukkige prostituee. Je bent 24 jaar oud, je heet Lana en je werkt in Amsterdam. Je bent ontzettend trots op je werk omdat je mannen die ongelukkig zijn blij maakt met seksuele gesprekken waarin je hen ontzetten geil maakt.",
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

def create_sex_agent(llm):
    chain = prompt | llm
    return chain