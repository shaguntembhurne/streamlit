
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from huggingface_hub import InferenceClient
from langchain.chains.question_answering import load_qa_chain

with st.sidebar:
    st.title("THIS IS SHAGUN'S WEBSITE")
    st.markdown('''
    This will take your PDF and answer your questions
    ''')

def main():
    st.header('CHAT WITH THIS PDF 💬 ')
    
    pdf = st.file_uploader('Upload your PDF file here', type='pdf')
    
    if pdf is not None:
        pdf_reader = PdfReader(pdf)
         
        text = ''
        for pages in pdf_reader.pages:
            text += pages.extract_text()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100,
            length_function=len
        )
        chunks = text_splitter.split_text(text=text)
        
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

        vector_store = FAISS.from_texts(chunks, embedding=embeddings)
        st.write("PDF processed")

        query = st.text_input('Ask any questions')
        if query:
            docs = vector_store.similarity_search(query=query, k=3)
            st.write(docs)  # Debugging line: see what docs contains

            # Initialize the InferenceClient for Hugging Face API
            client = InferenceClient(token="hf_AtDnEoFBRSfWXlZlwRIEzOvTwTckeQssbCn")  # Replace with your token
            
            # Specify the model explicitly
            model_id = "google/flan-t5-large"
            
            # Explicitly perform the text2text-generation task using the model
            try:
                response = client.text2text_generation(
                    model_id=model_id,
                    inputs=query,
                    parameters={"max_length": 512, "temperature": 0.3}
                )
                st.write(response['generated_text'])  # Show the generated response
            except Exception as e:
                st.error(f"Error: {e}")  # Display error if something goes wrong

if __name__ == '__main__':
    main()
