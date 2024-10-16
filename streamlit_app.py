import streamlit as st
from htmlTemplates import css, bot_template, user_template
from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import StorageContext, load_index_from_storage, DocumentSummaryIndex, get_response_synthesizer
from llama_index.core import SimpleDirectoryReader
from llama_index.core import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.groq import Groq
from txt_to_template import generate_output
import fitz
import os
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.indices.document_summary import DocumentSummaryIndexEmbeddingRetriever
from pathlib import Path

groq_api = st.secrets['groq_api']

llm = Groq(model="gemma2-9b-it", api_key=groq_api) # llama-3.1-70b-versatile, "llama-3.1-8b-instant", llama3-8b-8192
embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-mpnet-base-v2", trust_remote_code=True) # "BAAI/bge-small-en-v1.5", "sentence-transformers/all-mpnet-base-v2", "sentence-transformers/all-MiniLM-L6-v2", nomic-ai/nomic-embed-text-v1.5

Settings.embed_model = embed_model
Settings.llm = llm



def main():
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    if len(st.session_state.chat_history) > 20:
        st.session_state.chat_history = st.session_state.chat_history[-20:]
    st.set_page_config(page_title="Create exam from multiple PDFs", page_icon=":books:", layout="wide")
    st.write(css, unsafe_allow_html=True)
    if "conversation" not in st.session_state:
        st.session_state.conversation=None


    st.markdown("<h1 class='centered-header'>Create exam from your PDFs 📚</h1>", unsafe_allow_html=True)
    st.markdown("--------")


    def save_uploaded_files(file, folder='data'):
        with open(os.path.join(folder, file.name), 'wb') as f:
            f.write(file.getbuffer())
        return None

    def get_respond(user_query):
        storage_context = StorageContext.from_defaults(persist_dir="index")
        doc_summary_index = load_index_from_storage(storage_context)
        response_synthesizer = get_response_synthesizer(response_mode="tree_summarize")
        retriever = DocumentSummaryIndexEmbeddingRetriever(
            doc_summary_index,
        )
        # assemble query engine
        query_engine = RetrieverQueryEngine(
            retriever=retriever,
            response_synthesizer=response_synthesizer
            )
        return query_engine.query(user_query)
    
    def show_chat():
        st.session_state.show_chat = True

    # function to open a file and return its contents as a string
    def open_file(filepath):
        with open(filepath, 'r', encoding='utf-8') as infile:
            return infile.read()



    def get_mcq():
        
        user_input = "Generate varied 20 multiple choice questions covering all topics from the 01 materials. Indicate correct answer under each question. Make one question per topic."
        system_message = open_file('system_message_mcq.txt')
        user_input = system_message + user_input
        with open("chat_mcq.txt", 'w', encoding="utf-8") as f:
            print(get_respond(user_input), file=f)


    def get_owq():
        user_input = "Create exactly 5 open-written questions covering all topics from the 01 materials. For each question add a correct answer with 3-4 sentences."
        system_message = open_file('system_message_owq.txt')
        user_input = system_message + user_input
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

            with st.spinner("Creating embeddings and summary..."):
                all_text = ""
                # Loop through all files in the folder
                for filename in os.listdir('./data'):
                    if filename.endswith(".pdf"):
                        file_path = os.path.join('./data', filename)
                        # Open the PDF
                        pdf_document = fitz.open(file_path)
                        for page_num in range(pdf_document.page_count):
                            # Extract text from each page
                            page = pdf_document.load_page(page_num)
                            page_text = page.get_text()
                            page_text = page_text.replace('COPYRIGHT – DO NOT DISTRIBUTE WITHOUT WRITTEN PERMISSION\nV2.0', '\n')
                            all_text += page_text
                        pdf_document.close()
                

                data_path = Path('./data')
                with open(data_path / 'all_text.txt', 'w') as f:
                    f.write(all_text)

                # load txt file with SimpleDirectoryReader
                docs = SimpleDirectoryReader(input_files=['data/all_text.txt']).load_data()
                docs[0].doc_id = '01'

                splitter = SentenceSplitter(chunk_size=1024) 

                # default mode of building the index
                response_synthesizer = get_response_synthesizer(
                    response_mode="tree_summarize",
                )
                doc_summary_index = DocumentSummaryIndex.from_documents(
                    docs,
                    llm=llm,
                    transformations=[splitter],
                    response_synthesizer=response_synthesizer,
                    show_progress=True,
                )

                doc_summary_index.storage_context.persist("index")
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
                    with st.spinner('Generating...'):
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
                    with st.spinner('Generating...'):
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
                    if st.session_state.uploaded_files: del st.session_state.uploaded_files





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
                import shutil
                if os.path.exists('data'):
                    shutil.rmtree('data')



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
    
if __name__ == '__main__':
    main()
