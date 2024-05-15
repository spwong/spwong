from llama_index.core import SimpleDirectoryReader
from llama_index.readers.web import SimpleWebPageReader
from llama_index.core import Settings
from llama_index.core import VectorStoreIndex
from llama_index.llms.ollama import Ollama
from llama_index.core import PromptTemplate
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
# load data
docs = SimpleWebPageReader(html_to_text=True).load_data(["https://en.wikipedia.org/wiki/Hong_Kong"])


# ====== Create vector store and upload indexed data ======
llm = Ollama(model="llama3", request_timeout=120.0)
embed_model = HuggingFaceEmbedding( model_name="BAAI/bge-large-en-v1.5", trust_remote_code=True)

Settings.llm = llm
Settings.embed_model = embed_model
index = VectorStoreIndex.from_documents(docs)

Settings.llm = llm
query_engine = index.as_query_engine(streaming=True, similarity_top_k=4)
qa_prompt_tmpl_str = (
            "Context information is below.\n"
            "---------------------\n"
            "{context_str}\n"
            "---------------------\n"
            "Given the context information above I want you to think step by step to answer the query in a crisp manner, incase case you don't know the answer say 'I don't know!'.\n"
            "Query: {query_str}\n"
            "Answer: "
            )
#qa_prompt_tmpl = PromptTemplate(qa_prompt_tmpl_str)
#query_engine.update_prompts({"response_synthesizer:text_qa_template": qa_prompt_tmpl})

streaming_response = query_engine.query("How many Chinese in Hong Kong in 2021")
streaming_response.print_response_stream()

