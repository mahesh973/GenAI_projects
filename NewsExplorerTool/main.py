import os
import streamlit as st
import pickle
import time
import validators
from langchain.document_loaders import UnstructuredURLLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from validation import validate_and_check_url
from langchain.chat_models import ChatOpenAI

from dotenv import load_dotenv
load_dotenv()

st.title("News Explorer Tool.....")

st.sidebar.title("Enter the article links")

links = []

for i in range(2):
    link = st.sidebar.text_input(f"Enter link {i + 1}").strip()
    if link:
        if validate_and_check_url(link) == "URL is valid and accessible, with status code 200.":
            links.append(link)
        else:
            st.sidebar.write(f"{validate_and_check_url(link)}")
    st.write("\n")

process_button = st.sidebar.button("Process")

file_path = "faiss_vectordb.pkl"

main_placeholder = st.empty()

llm = ChatOpenAI(model = "gpt-3.5-turbo",temperature=0.5, openai_api_key = os.environ["OPENAI_API_KEY"])

if process_button:
    ## Loading the articles
    links_loader = UnstructuredURLLoader(urls=links)
    main_placeholder.text("Data Loading...Started...✅✅✅")
    data = links_loader.load()

    # Splitting the data
    text_splitter = RecursiveCharacterTextSplitter(
        separators=['\n\n', '\n', '.', ','],
        chunk_size=1000
    )

    main_placeholder.text("Text Splitter...Started...✅✅✅")
    documents = text_splitter.split_documents(data)

    # Creating the embeddings and saving it to FAISS index
    embeddings = OpenAIEmbeddings()
    vectordb = FAISS.from_documents(documents, embeddings)
    main_placeholder.text("Embedding Vector Started Building...✅✅✅")

    # Saving the FAISS index to pickle file
    with open(file_path, "wb") as f:
        pickle.dump(vectordb, f)

    time.sleep(2)


query = st.text_area("Enter your query:")

retrieve_answer = st.button("Generate")

if retrieve_answer:
    # if process_button:
    if os.path.exists(file_path):
        with open(file_path,"rb") as f:
            vectordb = pickle.load(f)
            response_chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=vectordb.as_retriever())
            response = response_chain({"question": query}, return_only_outputs=True)
            st.write("\n")
            st.header("Answer")
            st.write("\n")
            st.write(response['answer'])

            # Displaying the relevant source, from which the the answer is extracted
            sources = response.get("sources", "")
            if sources:
                st.subheader("Sources:")
                sources_list = sources.split("\n")
                for source in sources_list:
                    st.write(source)
    # else:
    #     st.write("Please enter the links and process them first")



