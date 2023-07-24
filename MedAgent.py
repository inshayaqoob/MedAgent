import streamlit as st
import os
#from dotenv import load_dotenv

#load_dotenv()
#st.set_page_config(page_title='MedAgentapp')
st.title("Let's set up the project")

st.markdown("""
         ### This is the MedAgent web app.
         This app is a prototype for the querying of **medical** data from pdf clinical reports, or Databases
         It uses the OpenAI models and langchain libraries to power this tool.

         You can **chat** with your documents and with your db.
         """)
st.subheader('Open AI API key')

st.text_input('Write your api key here', type='password' )
