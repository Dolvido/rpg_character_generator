import os
import json
import requests
import time
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_community.llms import Ollama
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.docstore.document import Document
from langchain.chains import ConversationChain
from langchain.chains import RetrievalQA
from langchain.schema import BaseRetriever

model = "mistral"

# Template for character generation
template = {
    "race": "",
    "class": "",
    "name": "",
    "background": "",
    "personality": "",
    "roleplay_instructions": "",
    "stats": {
        "strength": 0,
        "dexterity": 0,
        "constitution": 0,
        "intelligence": 0,
        "wisdom": 0,
        "charisma": 0
    },
    "equipment": []
}

prompt = f"""
Generate one realistically believable character for a roleplaying game, including:
- Race (choose from Human, Elf, Dwarf, Half-Orc, Gnome, Halfling)
- Class (choose from Fighter, Rogue, Wizard, Cleric, Ranger, Paladin)
- Name
- Background story (2-3 sentences)
- Personality traits (2-3 adjectives)
- Roleplay instructions (2-3 directives)
- Stats (strength, dexterity, constitution, intelligence, wisdom, and charisma)
- Equipment (include a mix of weapons, armor, and magical items)

Use the following template: {json.dumps(template)}.
"""

data = {
    "prompt": prompt,
    "model": model,
    "format": "json",
    "stream": False,
    "options": {"temperature": 1.5, "top_p": 0.99, "top_k": 100},
}

print("Generating a character for Dungeon's Fortress")
response = requests.post("http://localhost:11434/api/generate", json=data, stream=False)
json_data = json.loads(response.text)
character_data = json.loads(json_data["response"])
print(json.dumps(character_data, indent=2))

# Initialize the Ollama language model
llm = Ollama(model=model, callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]))

# Define a prompt template for in-character content generation
in_character_prompt = PromptTemplate(
    input_variables=["name", "background", "personality", "roleplay_instructions", "query"],
    template="""You are {name}, a {background} with a personality described as: {personality}.

    Roleplay Instructions:
    {roleplay_instructions}

    Query: {query}

    Respond in-character:""",
)

def save_message(character_name, message):
    if not os.path.exists(f"./memory/{character_name}"):
        os.makedirs(f"./memory/{character_name}")

    timestamp = int(time.time())
    file_name = f"{timestamp}.json"
    file_path = os.path.join(f"./memory/{character_name}", file_name)

    with open(file_path, "w") as f:
        json.dump({"message": message}, f)

def load_conversation_history(character_name):
    conversation_history_dir = f"./memory/{character_name}"
    if not os.path.exists(conversation_history_dir):
        os.makedirs(conversation_history_dir)
    conversation_files = [f for f in os.listdir(conversation_history_dir) if f.endswith(".json")]
    documents = []
    for file in conversation_files:
        with open(os.path.join(conversation_history_dir, file), "r") as f:
            documents.append(Document(page_content=f.read()))
    return documents

def re_vectorize_database(character_name):
    documents = load_conversation_history(character_name)
    if documents:
        return Chroma.from_documents(documents=documents, embedding=OllamaEmbeddings())
    else:
        return None

class PlaceholderRetriever(BaseRetriever):
    def get_relevant_documents(self, query):
        return []

def decide_response_type(query, context, character_name):
    vectorstore = re_vectorize_database(character_name)

    if vectorstore is None:
        print("Chose CONV (no conversation history)")
        return conversation_chain.run(context + "\n\nHuman: " + query)

    decision_context = {
        "query": query,
        "character_context": context,
        "decision_criteria": {
            "CONV": "If the prompt is a general conversational question, a greeting, or requires a creative response.",
            "RAG": "If the prompt is asking about specific information, memories, or events related to the context."
        },
        "examples": [
            {"prompt": "What is your name?", "decision": "[CONV]"},
            {"prompt": "Tell me about yourself.", "decision": "[CONV]"},
            {"prompt": "What did we discuss in our last meeting?", "decision": "[RAG]"},
            {"prompt": "What is your quest?", "decision": "[CONV]"},
            {"prompt": "Remind me what I need to do next.", "decision": "[RAG]"}
        ]
    }

    prompt = f"""
    Based on the provided decision context, decide whether the given query should be answered using a conversation chain [CONV] or a retrieval-augmented-generation chain [RAG].

    Decision Context: {json.dumps(decision_context, indent=2)}

    Query: {query}

    Decision:
    """

    data = {
        "prompt": prompt,
        "model": "mistral",
        "format": "json",
        "stream": False,
        "options": {"temperature": 0.7, "top_p": 0.9, "top_k": 50},
    }

    response = requests.post("http://localhost:11434/api/generate", json=data, stream=False)
    json_data = json.loads(response.text)

    try:
        decision = json.loads(json_data["response"])["decision"]
    except (KeyError, json.JSONDecodeError):
        decision = None

    if decision == "[CONV]":
        print("Chose CONV")
        return conversation_chain.run(context + "\n\nHuman: " + query)
    elif decision == "[RAG]":
        print("Chose RAG")
        context_str = f"Character Context:\n"
        context_str += f"Name: {character_data['name']}\n"
        context_str += f"Race: {character_data['race']}\n"
        context_str += f"Class: {character_data['class']}\n"
        context_str += f"Background: {character_data['background']}\n"
        context_str += f"Personality: {', '.join(character_data['personality'].split(', '))}\n"
        context_str += f"Roleplay Instructions: {character_data['roleplay_instructions']}\n"
        context_str += f"Stats:\n"
        for stat, value in character_data['stats'].items():
            context_str += f"  {stat.capitalize()}: {value}\n"
        context_str += f"Equipment:\n"
        for item in character_data['equipment']:
            context_str += f"  - {item}\n"
        context_str += "\n"
        print("query: " + query)
        retriever = vectorstore.as_retriever()
        return RetrievalQA.from_chain_type(
            llm,
            retriever=retriever,
            verbose=True,
            chain_type_kwargs={"prompt": PromptTemplate(input_variables=["context", "question"], template="""Use the following pieces of context to answer the question at the end.

            If you don't know the answer, just say that you don't know, don't try to make up an answer.

            Use three sentences maximum and keep the answer as concise as possible.

            {context}

            Question: {question}

            Helpful Answer:""")}
        )({"query": query, "context": context_str})["result"]

def generate_character_response(query, character_name):
    context = f"Name: {character_data['name']}\nBackground: {character_data['background']}\nPersonality: {', '.join(character_data['personality'].split(', '))}\nRoleplay Instructions: {character_data['roleplay_instructions']}"
    response = decide_response_type(query, context, character_name)
    save_message(character_name, f"Human: {query}\nAssistant: {response}")
    if response is not None:
        vectorstore = re_vectorize_database(character_name)
    return response

# Create conversation chain and retrieval QA chain
conversation_chain = ConversationChain(llm=llm)
retrieval_qa_chain = RetrievalQA.from_chain_type(
    llm,
    retriever=PlaceholderRetriever(),  # Use placeholder retriever
    chain_type_kwargs={"prompt": PromptTemplate(input_variables=["context", "question"], template="""Use the following pieces of context to answer the question at the end.

    If you don't know the answer, just say that you don't know, don't try to make up an answer.

    Use three sentences maximum and keep the answer as concise as possible.

    {context}

    Question: {question}

    Helpful Answer:""")}
)

while True:
    query = input("\nEnter a query (or 'exit' to quit): ")
    if query.lower() == "exit":
        break

    character_name = character_data["name"]
    response = generate_character_response(query, character_name)
    print(f"\nResponse:\n{response}")