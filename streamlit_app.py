import streamlit as st
from htmlTemplates import css, bot_template, user_template
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
#from llama_index.embeddings.openai import OpenAIEmbedding
#from llama_index.llms import openai
from llama_index.core import StorageContext, load_index_from_storage
#from langchain_community.chat_message_histories import ChatMessageHistory
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.groq import Groq
from dotenv import load_dotenv
from streamlit_extras.buy_me_a_coffee import button
#from get_questions import get_mcq, get_owq
from txt_to_template import generate_output
import os

load_dotenv() 
#os.environ["OPENAI_API_KEY"] = os.getenv('openai_key')
groq_api = os.getenv('GROQ_API_KEY')

llm = Groq(model="llama3-70b-8192", api_key=groq_api)
embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2", trust_remote_code=True) # "BAAI/bge-small-en-v1.5", "sentence-transformers/all-MiniLM-L6-v2", nomic-ai/nomic-embed-text-v1.5

Settings.embed_model = embed_model
Settings.text_splitter = SentenceSplitter(chunk_size=1024, chunk_overlap=20)
Settings.llm = llm





def main():
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    if len(st.session_state.chat_history) > 20:
        st.session_state.chat_history = st.session_state.chat_history[-20:]
    st.set_page_config(page_title="Create exam from multiple PDFs",page_icon=":books:", layout="wide")
    st.write(css, unsafe_allow_html=True)
    if "conversation" not in st.session_state:
        st.session_state.conversation=None

    
    #st.header("Create exam from your PDFs :books:", divider="rainbow")
    #st.markdown("# :orange[Create exam from your PDFs] :books:", unsafe_allow_html=True)
    st.markdown("<h1 class='centered-header'>Create exam from your PDFs 📚</h1>", unsafe_allow_html=True)
    st.markdown("--------")




    def save_uploaded_files(file, folder='data'):
        with open(os.path.join(folder, file.name), 'wb') as f:
            f.write(file.getbuffer())
        return None

    def get_respond(user_query):
        storage_context = StorageContext.from_defaults(persist_dir="storage")
        index = load_index_from_storage(storage_context)
        query_engine = index.as_query_engine()
        return query_engine.query(user_query)
    
    def show_chat():
        st.session_state.show_chat = True

    # function to open a file and return its contents as a string
    def open_file(filepath):
        with open(filepath, 'r', encoding='utf-8') as infile:
            return infile.read()



    def get_mcq():
        user_input = "Create 20 multiple choice questions with 4 answer options for each, having only 1 correct answer. Indicate correct answer under each question"
        system_message = open_file('system_message_mcq.txt')
        user_input = user_input + system_message

        with open("chat_mcq.txt", 'w', encoding="utf-8") as f:
            print(get_respond(user_input), file=f)


    def get_owq():
        user_input = "Create exactly 8 open-written questions. For each question add a correct answer with 2-3 sentences."
        system_message = open_file('system_message_owq.txt')
        user_input = user_input + system_message
        with open("chat_owq.txt", 'w', encoding="utf-8") as f:
            print(get_respond(user_input), file=f)
    



    with st.sidebar:
        st.header("Your documents")
        docs = st.file_uploader("Upload PDFs and click 'Process'",accept_multiple_files=True)
        st.session_state.uploaded_files = docs
        if st.button("Process", on_click=show_chat):
            os.mkdir('data')
            with st.spinner("Processing"):
                for doc in docs:
                    save_uploaded_files(doc)
                st.success('Files uploaded!')

            with st.spinner("Creating embeddings and vectors store..."):
                documents = SimpleDirectoryReader("./data").load_data()
                index = VectorStoreIndex.from_documents(documents, embed_model=embed_model)
                index.storage_context.persist()
                st.write('Completed!')
            
            st.session_state.show_chat = True


    if "show_chat" in st.session_state and st.session_state.show_chat:

        #
        with st.container(border=True):
            st.header('Generate exam parts')
            st.markdown('____________')


            col1, col2 = st.columns(2)
            with col1:
                st.markdown('Multiple Choice Questions')
                if st.button(label="Create MCQs"):
                    get_mcq()
                    with open("chat_mcq.txt", 'r', encoding="utf-8") as f: 
                        data = f.read()
                        st.session_state["new_mcq"] = data

                if "new_mcq" in st.session_state:
                    st.text_area("Check and edit if needed: ", key="new_mcq", height=420)
                    if st.button("Confirm MCQs"):
                        with open("chat_mcq.txt", 'w') as out: # rewrite the file with updated questions
                            print(st.session_state["new_mcq"], file=out)
                        st.write('Confirmed!')
            with col2:
                st.markdown('Open-Written Questions')
                if st.button(label="Create OWQs"):
                    get_owq()
                    with open("chat_owq.txt", 'r', encoding="utf-8") as f: 
                        data = f.read()
                        st.session_state["new_owq"] = data

                if "new_owq" in st.session_state:
                    st.text_area("Check and edit if needed: ", key="new_owq", height=420)
                    if st.button("Confirm OWQs"):
                        with open("chat_owq.txt", 'w') as out: # rewrite the file with updated questions
                            print(st.session_state["new_owq"], file=out)
                        st.write('Confirmed!')

            if 'new_mcq' in st.session_state or 'new_owq' in st.session_state:
                if st.button("Create exam!", use_container_width = True):
                    generate_output()
                    if st.session_state.get("new_mcq"): del st.session_state["new_mcq"] 
                    if st.session_state.get("new_owq"): del st.session_state["new_owq"]





            # download exam 
            if os.path.exists("output.docx"):
                with open("output.docx", 'rb') as file:
                    st.download_button(label="Download Exam", data=file, file_name="output.docx")




            # delete the output.docx
            if os.path.exists("output.docx"):
                os.remove("output.docx")
                # clean the user_topics file
                with open('user_topics.txt', 'w') as file:
                    pass
                with open('chat_mcq.txt', 'w') as cmcq:
                    pass
                with open('chat_owq.txt', 'w') as cowq:
                    pass


        # ------------------
        # Chat section
        # ------------------


        for message in st.session_state.chat_history:
            if message["role"] == "assistant":
                with st.chat_message("Cybria", avatar="👩‍🎤"):
                    st.write(message["content"])
            elif message["role"] == "user":
                with st.chat_message("Human"):
                    st.write(message["content"])


        user_query = st.chat_input("now chat with me...")
        if user_query is not None and user_query != '':
            st.session_state.chat_history.append({"role": "user", "content": user_query})
            with st.chat_message("human"):
                st.markdown(user_query)
            with st.chat_message("Cybria", avatar="👩‍🎤"):
                response = get_respond(user_query)
                st.write(response.response)
            st.session_state.chat_history.append({"role": "assistant", "content": response.response})

        # -------------------------

    else:
        st.markdown('<div class="centered-subheader"><h3> <-- First upload the course materials</h3></div>', unsafe_allow_html=True)

    import shutil
    if os.path.exists('data'):
        shutil.rmtree('data')
    
if __name__ == '__main__':
    main()
