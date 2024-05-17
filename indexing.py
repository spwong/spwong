from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain.chains import create_history_aware_retriever
from langchain_core.prompts import MessagesPlaceholder
from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import OllamaEmbeddings
model="llama3"
llm = ChatOllama(model=model, temperature=0)
loader = PyPDFLoader('./test.pdf')
docs = loader.load()
embeddings = OllamaEmbeddings(model=model)
text_splitter = RecursiveCharacterTextSplitter()
documents = text_splitter.split_documents(docs)[:10]#解析前10页
#vector = FAISS.from_documents(documents, embeddings)
vector = FAISS.from_texts(["小明是一位科学家", "小明在balala地区工作"],embeddings)
retriever = vector.as_retriever()
prompt1 = ChatPromptTemplate.from_messages([MessagesPlaceholder(variable_name="chat_history"),("user", "{input}"),("user", "Given the above dialog, generate a search query to be found for information related to the dialog")])
retriever_chain = create_history_aware_retriever(llm, retriever, prompt1)
prompt2 = ChatPromptTemplate.from_messages([("system", "Answer the question based on the article.{context}"),MessagesPlaceholder(variable_name="chat_history"),("user", "{input}"),])
document_chain = create_stuff_documents_chain(llm, prompt2)
retrieval_chain = create_retrieval_chain(retriever_chain, document_chain)
chat_history = []
while True:
    question = input('User：')
    response = retrieval_chain.invoke({
        "chat_history": chat_history,
        "input": question
    })
    answer = response["answer"]
    chat_history.extend([question, answer])
    print('AI：', answer)
