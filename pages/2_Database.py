from ossaudiodev import openmixer
import streamlit as st
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import MongoDBAtlasVectorSearch
from langchain.document_loaders import TextLoader
#from dotenv import load_dotenv
import os
import getpass
from pymongo import MongoClient
from langchain.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain
#####################################################################################
st.title("Database browser")

st.write('this part will help the client browse his data on the DB and query it via prompting')
############################################################################################
################################## prompt template ##################################
prompt_template = """You are a personal medical Bot assistant for answering any questions about documents.
Use medical technical language.
You are given a question and a set of documents.
divide the answer in the corresponding sections.
If the user's question requires you to provide specific information from the documents, give your answer based only on the examples provided below.
If you don't find the answer to the user's question with the examples provided to you below, answer that you didn't find the answer in the documentation and propose him to rephrase his query with more details.
Use bullet points if appropriate.

QUESTION: {question}

DOCUMENTS:
=========
{context}
=========
Finish by proposing your help for anything else.
"""
k = 4  # number of chunks to consider when generating answer
################################## loading the .env variables #######################
#load_dotenv()



st.header('Clinical Report Chat')
#####################################################################################
#################################### sidebar uploader ##############################
with st.sidebar:
    model = st.selectbox('OpenAI models',
                         options=['gpt-3.5-turbo', 'text-curie-001'])
    # Connect to MongoDB and retrieve data from the collection

    # Load environment variables
    #load_dotenv()

    # Replace the connection string and database name with your MongoDB details
    mongo_uri = st.secrets('MONGO_URI')
    db_name = st.secrets('MONGO_DB_NAME')
    collection_name = st.secrets('MONGO_COLLECTION_NAME')

    # Connect to MongoDB
    client = MongoClient(mongo_uri)
    db = client[db_name]
    collection = db[collection_name]
    myclient = MongoClient(mongo_uri)
    # Fetch data from the collection (assuming the data contains a field named 'text')
    text =  client.find()




    st.write(text)

    #split chunks

    text_splitter = CharacterTextSplitter(separator="\n",
                                            chunk_size=1000,
                                            chunk_overlap=200,
                                            length_function=len)

    #create the chunks

    chunks = text_splitter.split_text(text)

    #st.write(chunks)

    #creating embeddings

    embeddings = OpenAIEmbeddings()

    knowledge_base = FAISS.from_texts(chunks, embeddings)

###################################################################################
############################### chat functionality ###########################################
try:
    # Check if "messages" key exists in session state, if not, initialize an empty list
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Iterate through each message in the saved messages and display them in the chat
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message['content'])

    # Get the user input question
    if user_question := st.chat_input('Query the document'):
        # Save the user's question in the session state
        st.session_state.messages.append({
            'role': 'user',
            'content': user_question
        })
        with st.chat_message('user'):
            st.markdown(user_question)

    if user_question:
        docs = knowledge_base.similarity_search(user_question)
        prompt_template = PromptTemplate(
            template=prompt_template, input_variables=['context', 'question'])

        llm = openmixer(temperature=0.7, model_name=model)

        chain = load_qa_chain(llm, chain_type='stuff', prompt=prompt_template)
        response = chain.run(input_documents=docs, question=user_question)

        # Save the assistant's response in the session state
        st.session_state.messages.append({
            'role': 'assistant',
            'content': response
        })

        with st.chat_message('assistant'):
            message_placeholder = st.empty()
            st.markdown(response)

except Exception as e:
    st.warning("A PDF file hasn't been uploaded correctly")
