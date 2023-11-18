from langchain.document_loaders import PyPDFLoader, OnlinePDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Pinecone
from sentence_transformers import SentenceTransformer
from langchain.chains.question_answering import load_qa_chain
import os
from langchain.llms import LlamaCpp
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from huggingface_hub import hf_hub_download
import pinecone

loader = PyPDFLoader("yolov7paper.pdf")
data = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
docs = text_splitter.split_documents(data)
# print(len(docs))
# print(docs[0])
key = <your pinecone api key>
env = <your pinecone environment>
PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY', key)
PINECONE_API_ENV = os.environ.get('PINECONE_API_ENV', env)
embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-V2')

pinecone.init(
    api_key=PINECONE_API_KEY,
    environment=PINECONE_API_ENV
)
index_name = <your index_name>

docsearch = Pinecone.from_texts([t.page_content for t in docs], embeddings, index_name=index_name)



callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

model_name_or_path = "TheBloke/Llama-2-7B-chat-GGML"
model_basename = "llama-2-7b-chat.ggmlv3.q5_1.bin"
model_path = hf_hub_download(repo_id=model_name_or_path, filename=model_basename)

n_gpu_layers = 40
n_batch = 256

llm  = LlamaCpp(
    model_path=model_path,
    max_tokens=256,
    n_gpu_layers=n_gpu_layers,
    n_batch=n_batch,
    callback_manager=callback_manager,
    n_ctx=1024,
    verbose=False 
)

chain = load_qa_chain(llm, chain_type="stuff")
print(chain)

query = "YOLOv7 outperform which models"
docs = docsearch.similarity_search(query)
print(docs)
print(query)
result = chain({"input_documents": docs, "question": query}, return_only_outputs=True)
print(result)
# chain.run({query})
