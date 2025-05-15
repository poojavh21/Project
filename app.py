import streamlit as st
import pandas as pd
from datetime import datetime
import os ,io
import json
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import CohereEmbeddings
from langchain.llms import Cohere
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.agents import initialize_agent, AgentType
from langchain.tools import Tool
from sentence_transformers import SentenceTransformer # Changed 'sentance_transformers' to 'sentence_transformers' and 'SentanceTransformer' to 'SentenceTransformer'

from langchain_community.embeddings import HuggingFaceEmbeddings

# Load API key from .env
load_dotenv('.env')
co = os.getenv('cohere_api_key')

# Initialize Cohere LLM
llm = Cohere(cohere_api_key=co, truncate="END")     # truncate for token limit errors(4081 token)

# Initialize Cohere
# Embeddingsembedding_model = CohereEmbeddings(model="embed-english-v3.0", cohere_api_key=co, user_agent="my-app")

# Load CSV data into a buffer
buffer = io.StringIO()           # allowing for treat string as file
with open('Flipkart_laptops.csv', 'r') as file:
    buffer.write(file.read())

buffer.seek(0)  # Reset buffer position

# Read laptop data from buffer
laptops = pd.read_csv(buffer)

# Convert DataFrame rows into formatted text for FAISS embedding
laptop_texts = laptops.apply(lambda row: " ".join(map(str, row)), axis=1).tolist()



embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Create FAISS vector store
vectorstore = FAISS.from_texts(laptop_texts, embedding_model)

# Create retriever (limit results for efficiency)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})  # Max 3 results

# Define custom prompt template
template = """You are a laptop expert. Use the following context to answer the question.
              Context:{context},   Question: {question}"""
prompt = PromptTemplate(input_variables=["context", "question"], template=template)

# Initialize conversation memory
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# Define retrieval tool
retrieval_tool = Tool(
    name="LaptopRecommendationTool",
    func=lambda query: retriever.get_relevant_documents(query),
    description="Provides relevant laptops based on specifications."
)

# Initialize AI agent with conversational memory
agent = initialize_agent(
    tools=[retrieval_tool],
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    memory=memory,
)

# Streamlit UI
st.title("🛍️ AI-Powered Laptop Assistant ")
st.write("Ask about laptop specifications, battery life, or performance!")

# User query input
user_query = st.text_input("Enter your laptop-related question:", "")

if st.button("Get Recommendation"):
    if user_query:
        # --- Tabs ---
        tab1, tab2 = st.tabs(["Recommendation", " History"])
        response = agent.run(user_query)
        with tab1:
            st.write("# AI Recommendation")
            st.write(response)

        # Show conversation history
        chat_history = memory.load_memory_variables({})
        with tab2:
            st.write("# Conversation History")

            try:
                with open("Searching_history.json", "r", encoding="utf-8") as f:
                    full_history = json.load(f)
            except FileNotFoundError:
                full_history = []

            if full_history:
                # Show most recent first
                full_history = sorted(full_history, key=lambda x: x["timestamp"], reverse=True)

                for item in full_history:
                    st.markdown(f"**🕓 {item['timestamp']}**")
                    st.markdown(f"- **Query:** {item['query']}")
                    st.markdown(f"- **Response:** {item['response']}")
                    st.markdown("---")
            else:
                st.info("No conversation history found.")

        # Save conversation history
        def save_conversation(history, filename="Searching_history.json"):
            try:
                with open(filename, "r", encoding="utf-8") as file:
                    existing_data = json.load(file)
            except FileNotFoundError:
                existing_data = []

            existing_data.extend(history)

            with open(filename, "w", encoding="utf-8") as file:
                json.dump(existing_data, file, indent=4)

        conversation_history = chat_history["chat_history"]
        history_list = [
            {
                "query": message.content,
                "response": conversation_history[idx + 1].content,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            for idx, message in enumerate(conversation_history[:-1]) if idx % 2 == 0
        ]

        save_conversation(history_list)
        st.write("Conversation saved to conversation_history.json!")
    else:
        st.warning("Please enter a question first!")