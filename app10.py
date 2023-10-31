import re
import time
from io import BytesIO
from typing import Any, Dict, List
import openai
import streamlit as st
from langchain import LLMChain, OpenAI
from langchain.agents import AgentExecutor, Tool, ZeroShotAgent
from langchain.chains import RetrievalQA
from langchain.chains.question_answering import load_qa_chain
from langchain.docstore.document import Document
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.memory import ConversationBufferMemory
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import VectorStore
from langchain.vectorstores.faiss import FAISS
from pypdf import PdfReader
import os



api = st.secrets['OPENAI_API_KEY']



# Define a function to parse a PDF file and extract its text content
@st.cache_data
def parse_pdf(file: BytesIO) -> List[str]:
    pdf = PdfReader(file)
    output = []
    for page in pdf.pages:
        text = page.extract_text()
        output.append(text)
    return output





# Define a function to convert text content to a list of documents
@st.cache_data
def text_to_docs(text):
    # Ensure input is a list of strings
    if isinstance(text, str):
        text = [text]

    doc_chunks = []

    # Iterate over pages and text content
    for i, page_text in enumerate(text, start=1):
        chunk_size = 2000
        separators = ["\n\n", "\n", ".", "!", "?", ",", " "]

        # Split page text by newline characters and process each chunk
        for j, chunk_text in enumerate(page_text.split("\n"), start=1):
            # Replace specified separators with spaces for consistent chunking
            for separator in separators:
                chunk_text = chunk_text.replace(separator, " ")

            # Limit each chunk to a maximum size of 2000 characters
            chunk_text = chunk_text[:chunk_size]

            # Create a Document for each chunk with appropriate metadata
            doc = Document(
                page_content=chunk_text,
                metadata={
                    "page": i,
                    "chunk": j,
                    "source": f"{i}-{j}"
                }
            )
            doc_chunks.append(doc)

    return doc_chunks






# Define a function for the embeddings
@st.cache_data
def create_embeddings():
    embeddings = OpenAIEmbeddings(openai_api_key=api)
    # Indexing
    # Save in a Vector DB
    with st.spinner("It's indexing..."):
        index = FAISS.from_documents(pages, embeddings)
    st.success("Embeddings done.", icon="✔")
    return index




# Define a function to clear the conversation history
def clear_history():
    st.session_state.memory.clear()




# Sidebar contents
with st.sidebar:
    st.title('🗨️ LLM Chat App 🗨️')
    st.write("Welcome to the LLM Chat App powered by LangChain and OpenAI!")
    st.write("About this App:")
    st.markdown(
        "This app allows you to chat with a language model powered by LLM. "
        "Ask questions and get answers from a large corpus of text data. "
        "Try it out and explore the capabilities of modern language models."
    )
    st.markdown("[Streamlit](https://streamlit.io/)")
    st.markdown("[LangChain](https://python.langchain.com/)")
    st.markdown("[OpenAI LLM](https://platform.openai.com/docs/models)")
    st.write("Made by [Sampik Kumar Gupta](https://www.linkedin.com/in/sampik-gupta-41544bb7/)")





# Set up the Streamlit app
st.title("Ask any questions from your PDF by uploading it Here")




# Allow the user to upload a PDF file
uploaded_file = st.file_uploader("**Upload Your PDF File**", type=["pdf"])



if uploaded_file:
    name_of_file = uploaded_file.name
    doc = parse_pdf(uploaded_file)
    pages = text_to_docs(doc)

    if pages:
        # Allow the user to select a page and view its content
        with st.expander("Show Page Content", expanded=False):
            page_sel = st.number_input(label="Select Page", min_value=1, max_value=len(pages), step=1)
            st.write(pages[page_sel - 1])



        if api:
            # Test the embeddings and save the index in a vector database
            index = create_embeddings()
            # Set up the question-answering system
            qa = RetrievalQA.from_chain_type(llm=OpenAI(openai_api_key=api), chain_type="stuff", retriever=index.as_retriever())



            # Set up the conversational agent
            tools = [
            Tool(name="Document Q&A Tool", func=qa.run,
                       description="This tool allows you to ask questions about the document you've uploaded. You can ask about any topic or content within the document.",
                 )]
            prefix = """Engage in a conversation with the AI, answering questions about the uploaded document. You have access to a single tool:"""
            suffix = """Begin the conversation!"

            {chat_history}
            Question: {input}
            {agent_scratchpad}"""



            prompt = ZeroShotAgent.create_prompt(tools,
                prefix=prefix,
                suffix=suffix,
                input_variables=["input", "chat_history", "agent_scratchpad"]
            )


            if "memory" not in st.session_state:
                st.session_state.memory = ConversationBufferMemory(memory_key="chat_history")


            llm_chain = LLMChain(llm=OpenAI(temperature=0, openai_api_key=api, model_name="gpt-3.5-turbo"), prompt=prompt)


            agent = ZeroShotAgent(llm_chain=llm_chain, tools=tools, verbose=True)
            agent_chain = AgentExecutor.from_agent_and_tools(agent=agent, tools=tools, verbose=True, memory=st.session_state.memory)


            # Allow the user to enter a query and generate a response
            query = st.text_input("Start a Conversation with the Bot!", placeholder="Ask the bot anything from {}".format(name_of_file))


            if query:
                with st.spinner("Generating Answer to your Query: `{}`".format(query)):
                    res = agent_chain.run(query)
                    st.info(res, icon="📝")


            # Display conversation history in parallel
            st.title("Conversation History")
            st.session_state.memory


            # Add a "New Chat" button to clear the conversation history
            if st.button("New Chat"):
                clear_history()



