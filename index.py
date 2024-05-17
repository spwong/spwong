"""
This script creates a database of information gathered from local text files.
"""

from langchain_community.document_loaders import DirectoryLoader, CSVLoader, BSHTMLLoader, PyPDFDirectoryLoader


from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# define what documents to load
#loader = DirectoryLoader("./", glob="*.x*", loader_cls=BSHTMLLoader)
loader = PyPDFDirectoryLoader("./")


# interpret information in the documents
documents = loader.load()
splitter = RecursiveCharacterTextSplitter(chunk_size=500,
                                          chunk_overlap=50)
texts = splitter.split_documents(documents)
embeddings = HuggingFaceEmbeddings(
    #model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_name="sentence-transformers/all-mpnet-base-v2",
    model_kwargs={'device': 'cpu'})

# create and save the local database
db = FAISS.from_documents(texts, embeddings)
db.save_local("csd")

#Based on Table 7.17, it appears that the age group with the highest rate of unemployment is 15-24 years old, with a rate of 16.9% for males and 20.8% for females.
