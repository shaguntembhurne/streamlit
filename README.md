
# **PDF Reader Chatbot**

## Overview

This project is a **PDF Reader Chatbot** that allows users to upload a PDF file, processes its content, and answers questions based on the extracted text. The chatbot leverages **LangChain**, **FAISS**, and **HuggingFace** embeddings for advanced question-answering capabilities. It utilizes **Streamlit** for a simple web interface, allowing users to interact with the PDF content by asking specific questions.

## Features

- **PDF Upload**: Users can upload a PDF document for analysis.
- **Text Extraction**: Extracts text from the uploaded PDF.
- **Text Splitting**: Splits the text into smaller chunks to enhance the accuracy of the question-answering system.
- **Embedding Generation**: Converts the text chunks into vector embeddings using **HuggingFaceEmbeddings**.
- **Similarity Search**: Searches the most relevant chunks based on user queries.
- **Question Answering**: Uses **Google FLAN-T5** via **HuggingFaceHub** for generating answers to user queries.

## Tech Stack

- **Streamlit**: For creating the web-based interface.
- **PyPDF2**: For extracting text from PDF files.
- **LangChain**: For splitting text, creating embeddings, and running the question-answering chain.
- **FAISS**: For efficient similarity search on text embeddings.
- **HuggingFaceEmbeddings**: For converting text into vector representations.
- **HuggingFaceHub**: For using the **Google FLAN-T5** language model for question-answering.

## Installation

### Prerequisites:
- **Python 3.x**
- **pip** (Python package manager)

### Installation Steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/pdf-reader-chatbot.git
   ```
2. Navigate to the project directory:
   ```bash
   cd pdf-reader-chatbot
   ```
3. Install the required Python libraries:
   ```bash
   pip install -r requirements.txt
   ```
4. Start the Streamlit application:
   ```bash
   streamlit run app.py
   ```
5. Open your browser and visit `http://127.0.0.1:8501` to access the PDF Reader Chatbot.

## Usage

1. **Upload a PDF**: Click on the "Upload your PDF file here" button to upload a PDF file.
2. **Ask Questions**: Once the PDF is processed, you can ask questions about the contents. The chatbot will analyze the document and provide relevant answers.
3. **View Responses**: The chatbot will return answers based on the most relevant sections of the PDF.

## Code Explanation

- **PDF Processing**: The `PdfReader` class from PyPDF2 is used to read the contents of the uploaded PDF and extract the text from all pages.
- **Text Splitting**: The text is split into chunks of size 1000 characters with 100-character overlap using the `RecursiveCharacterTextSplitter` from LangChain to make the question-answering system more efficient.
- **Embeddings**: **HuggingFaceEmbeddings** are used to convert the text chunks into vector embeddings, which are then stored in a **FAISS** vector store for similarity search.
- **Question-Answering**: User queries are answered by searching for the most relevant document chunks and passing them through the **FLAN-T5** model via the **HuggingFaceHub** API.

## Example

1. **Upload a PDF**: Upload any PDF document.
2. **Ask a Question**: For example, you can ask, "What is the main topic of this PDF?" The chatbot will process the PDF and return the most relevant information from the document.

## Contribution

Feel free to fork the repository and submit pull requests for:
- Bug fixes
- Feature additions
- UI/UX improvements

### How to Contribute:
1. Fork the repository
2. Create a new branch (`git checkout -b feature/your-feature-name`)
3. Make your changes
4. Commit your changes (`git commit -am 'Added feature'`)
5. Push to the branch (`git push origin feature/your-feature-name`)
6. Open a pull request
