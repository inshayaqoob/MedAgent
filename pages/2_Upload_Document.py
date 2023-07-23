import streamlit as st

st.title('Document Uploader')
st.write('this part will let the user upload a document (pdf) so it can be read by the AI Agent and extract information')

uploaded_file = st.file_uploader('Pick a file')
