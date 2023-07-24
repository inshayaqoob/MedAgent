############################### importing relevant libraries #########################
import streamlit as st
#from pyperclip import copy
#from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
import openai
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
OPENAI_API_KEY = st.secrets['API_KEY']
openai.api_key = st.secrets["API_KEY"]


st.header('Clinical Report Chat')
#####################################################################################
#################################### sidebar uploader ##############################
with st.sidebar:
    model = st.selectbox('OpenAI models', options=['gpt-3.5-turbo', 'text-curie-001'])
    # file upload
    pdf = st.file_uploader('Upload your pdf here', type='pdf')

    # text extract
    if pdf is not None:
        reader = PdfReader(pdf)
        text = ''
        for page in reader.pages:
            text += page.extract_text()

        #st.write(text)

        #split chunks

        text_splitter = CharacterTextSplitter(
            separator = "\n",
            chunk_size = 1000,
            chunk_overlap=200,
            length_function = len
        )

        #create the chunks

        chunks = text_splitter.split_text(text)

        #st.write(chunks)

        #creating embeddings

        embeddings = OpenAIEmbeddings(openai_api_key=st.secrets['API_KEY'])

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
        prompt_template = PromptTemplate(template=prompt_template,
                                        input_variables=['context', 'question'])

        llm = OpenAI(
            st.secrets['API_KEY'],
            temperature=0.7,
            model_name=model)

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
    st.write("A PDF file hasn't been uploaded correctly", e)
