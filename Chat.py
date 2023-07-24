from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain import OpenAI, VectorDBQA
from langchain.document_loaders import DirectoryLoader
from langchain.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain
import os
import streamlit as st
import logging

#config

PERSIST_DIR = '/raw_data/vector'
LOGS_FILE = str(os.environ.get('LOGS_FILE'))
prompt_template = """You are a personal medical Bot assistant for answering any questions about documents.
Use medical technical language.
You are given a question and a set of documents.
If the user's question requires you to provide specific information from the documents, give your answer based only on the examples provided below. DON'T generate an answer that is NOT written in the provided examples.
If you don't find the answer to the user's question with the examples provided to you below, answer that you didn't find the answer in the documentation and propose him to rephrase his query with more details.
Use bullet points if you have to make a list, only if necessary.

QUESTION: {question}

DOCUMENTS:
=========
{context}
=========
Finish by proposing your help for anything else.
"""
k = 4  # number of chunks to consider when generating answer

# Initialize logging with the specified configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(LOGS_FILE),
        logging.StreamHandler(),
    ],
)
LOGGER = logging.getLogger(__name__)
file_dir = "./raw_data"
loader = DirectoryLoader(path=file_dir, glob='*.pdf')
documents = loader.load()


#split the text

text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)

#split the documents into chunks of size 1000 using the splitter

texts = text_splitter.split_documents(documents)

#create a vector store from the chunks using the OpenAI embeddings and a chroma object
api = st.secrets['API_KEY']
embeddings = OpenAIEmbeddings(openai_api_key= api)
docsearch = Chroma.from_documents(texts, embeddings)

###############################################################################################
def answer(prompt: str, presist_directory: str = PERSIST_DIR) -> str:

    #api = os.environ.get('OPENAI_API')
    LOGGER.info(f"Start answerin based on prompt: {prompt}")
    #api = os.environ.get('OPENAI_API')

    #promt template

    prompt_template_2 = PromptTemplate(template=prompt_template,
                                       input_variables=['context', 'question'])

    #load qa chain
    doc_chain = load_qa_chain(llm=OpenAI(openai_api_key=api,
                                         model_name="text-davinci-003",
                                         temperature=0.7,
                                         max_tokens=300),
                              prompt=prompt_template_2)
    #log a message indicating the number of chunks to be considered
    LOGGER.info(
        f"The top {k} chunks are considered to answer the user's query")

    #create the vectorDBQA object
    qa = VectorDBQA(vectorstore=docsearch,
                    combine_documents_chain=doc_chain,
                    k=k)

    #call the vectorDBQA object to generate an answer to the prompt
    result = qa({"query": prompt})
    answer = result['result']

    #log a message indicating the answer
    LOGGER.info(f"The returned answer is {answer}")

    #log a message indicating the function has finished and returned an answer
    LOGGER.info(f"Answering module over.")

    return answer
