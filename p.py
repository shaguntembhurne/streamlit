import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
with st.sidebar:
    st.title("THIS IS SHAGUN'S WEBSITE")
    st.markdown('''
    This will take you pdf and will answer your questions
    ''')






def main():
    st.header('CHAT WITH THIS PDF 💬 ')
    
    pdf = st.file_uploader('upload your pdf file here' ,type='pdf')

    
    if pdf is not None:
        pdf_reader = PdfReader(pdf)
         
        text = ''
        for pages in pdf_reader.pages:
            text += pages.extract_text()



        text_spit = RecursiveCharacterTextSplitter(
            chunk_size = 1000,
            chunk_overlap = 100,
            length_function = len
        )
        chunks = text_spit.split_text(text=text)
        
        embeddings = HuggingFaceEmbeddings(model_name= "all-MiniLM-L6-v2")
        vector_store = FAISS.from_texts(chunks, embedding= embeddings)
        st.write("PDF processed and vector store created!")







if __name__ == '__main__':
    main()