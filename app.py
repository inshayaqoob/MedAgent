import streamlit as st
import os
import Chat
import logging
from streamlit_chat import message




st.set_page_config(
    page_title= 'Homepage'
)
st.title('Hello')
st.write('This is the home page')

with st.sidebar:
    File = st.file_uploader('Pick a file')
#create the chatbot interface

if 'generated' not in st.session_state:
    st.session_state['generated'] = []
if 'past' not in st.session_state:
    st.session_state['past'] = []

#define a function to clear text

def clear_input_text():
    global input_text
    input_text = ''

# get the users input

def get_text():
    global input_text
    input_text = st.chat_input('Ask something', key='input')
    return input_text


def main():
    user_input = get_text()

    if user_input:
        output = Chat.answer(user_input)
        #store the output
        st.session_state.past.append(user_input)
        st.session_state.generated.append(output)
    if st.session_state['generated']:
        for i in range(len(st.session_state['generated'])):

            message(st.session_state['past'][i], is_user=True, key=str(i) + '_user')
            message(st.session_state['generated'][i], key=str(i))

main()
st.write(st.session_state)
