"""
This script reads the database of information from local text files
and uses a large language model to answer questions about their content.
"""

from langchain_community.llms import Ollama
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA


# prepare the template we will use when prompting the AI
template = """Use the following pieces of information to answer the user's question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
Context: {context}
Question: {question}
Only return the helpful answer below and nothing else.
Helpful answer:
"""


# load the language model
# llm = CTransformers(model='D:\\dev\\llama3\\Meta-Llama-3-8B-Instruct.Q4_K_M.gguf',
#                     model_type='llama',
#                     config={'max_new_tokens': 4096, 'temperature': 0.02})

ollama = Ollama(model="llama3")

# load the interpreted information from the local database
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2",
    model_kwargs={'device': 'cpu'})
db = FAISS.load_local("csd", embeddings=embeddings, allow_dangerous_deserialization=True )

# prepare a version of the llm pre-loaded with the local content
retriever = db.as_retriever(search_kwargs={'k': 4})
prompt = PromptTemplate(template=template,input_variables=['context', 'question'])
qa_llm = RetrievalQA.from_chain_type(llm=ollama,
                                     chain_type='stuff',
                                     retriever=retriever,
                                     return_source_documents=True,
                                     chain_type_kwargs={'prompt': prompt})

# ask the AI chat about information in our local files
prompt = "which age group has the most serious unemployment?"
output = qa_llm({'query': prompt})
print(output["result"])
