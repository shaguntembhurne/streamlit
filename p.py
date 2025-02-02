import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import HuggingFaceHub
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

            # Initialize HuggingFaceHub LLM with a better model
            llm = HuggingFaceHub(
                repo_id="google/flan-t5-large",  # Correct model ID
                task="text2text-generation",  # Specify the task type
                model_kwargs={"temperature": 0.3, "max_length": 512},
                huggingfacehub_api_token="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"  # Replace with your actual Hugging Face token
            )
            chain = load_qa_chain(llm=llm, chain_type='stuff')
            
            try:
                responses = chain.run(input_documents=docs, question=query)
                st.write(responses)
            except Exception as e:
                st.error(f"Error: {e}")  # Display error if something goes wrong

if __name__ == '__main__':
    main()
