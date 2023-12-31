import os
import sys

import openai
import qdrant_client
import qdrant_client.http
from dotenv import load_dotenv
from langchain.chains import ConversationalRetrievalChain, RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import DirectoryLoader, TextLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.indexes import VectorstoreIndexCreator
from langchain.chains import RetrievalQA
from langchain.indexes.vectorstore import VectorStoreIndexWrapper
from langchain.llms import OpenAI
from langchain.schema import retriever
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma, __all__
import constants
import json
import dotenv
from langchain.vectorstores.qdrant import Qdrant
from langchain.document_loaders import PyPDFLoader
from PyPDF2 import PdfReader
from bs4 import BeautifulSoup



#os.environ["OPENAI_API_KEY"] = constants.APIKEY

def load_documents_name(file_path: str) -> dict:
  with open(file_path, 'r') as file:
    data: dict = json.load(file)
  return data

def save_documents_name(file_path: str, data: dict):
  with open(file_path, 'w') as file:
    json.dump(data, file, indent=2)

def get_document_name(file_path):
  file_path_components = file_path.split('/')
  file_name_and_extension = file_path_components[-1].rsplit('.', 1)
  return file_name_and_extension[0]

def list_documents_name(dir_path):
  res = []
  # Iterate directory
  for file_path in os.listdir(dir_path):
    # check if current file_path is a file
    if os.path.isfile(os.path.join(dir_path, file_path)):
      # add filename to list
      res.append(file_path)
  return res

def get_chunks(content):
  text_splitter = CharacterTextSplitter(
    separator="\n",
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len)
  chunks = text_splitter.split_text(content)
  return chunks

def read_files_in_folder(folder_path):
  """
  Read the content of text, PDF, and HTML files in a folder and store in a dictionary.
  Parameters:
  - folder_path (str): The path to the folder containing the files.
  Returns:
  - dict: A dictionary where keys are file names and values are the content of the files.
  """
  file_contents = {}

  for file_name in os.listdir(folder_path):
    file_path = os.path.join(folder_path, file_name)
    if os.path.isfile(file_path):
      if file_path.endswith('.txt'):
        with open(file_path, 'r', encoding='utf-8') as file:
          file_contents[file_name] = file.read()
      elif file_path.endswith('.pdf'):
        with open(file_path, 'rb') as file:
          pdf_reader = PdfReader(file)
          text = ''
          for page_num in range(len(pdf_reader.pages)):
            text += pdf_reader.pages[page_num].extract_text()
          file_contents[file_name] = text
    elif file_path.endswith('.html'):
        with open(file_path, 'r', encoding='utf-8') as file:
          soup = BeautifulSoup(file, 'html.parser')
          text = ' '.join(soup.stripped_strings)
          file_contents[file_name] = text
  return file_contents

load_dotenv()
client = qdrant_client.QdrantClient(
  os.getenv('QDRANT_HOST'),
  api_key=os.getenv('QDRANT_API_KEY')
)
# vectors_config = qdrant_client.http.models.VectorParams(
#   size=1536,
#   distance=qdrant_client.http.models.Distance.COSINE
# )
# client.recreate_collection(
#   collection_name=os.getenv('QDRANT_COLLECTION_NAME'),
#   vectors_config=vectors_config
# )

embeddings = OpenAIEmbeddings()
vector_store = Qdrant(
  client=client,
  collection_name=os.getenv('QDRANT_COLLECTION_NAME'),
  embeddings=embeddings
)
# Enable to save to disk & reuse the model (for repeated queries on the same data)
# PERSIST = False
#
query = None
if len(sys.argv) > 1:
  query = sys.argv[1]


# if PERSIST and os.path.exists("persist"):
#   print("Reusing index...\n")
  # vector van repo
  # vectorstore = Chroma(persist_directory="persist", embedding_function=OpenAIEmbeddings())
  # index = VectorStoreIndexWrapper(vectorstore=vectorstore)
# else:
  # loader = TextLoader("data/data.txt") # Use this line if you only need data.txt
  # documents = get_chunks(loader)
  # vector_store.add_documents(documents)
#   if PERSIST:
#     index = VectorstoreIndexCreator(vectorstore_kwargs={"persist_directory":"persist"}).from_loaders([loader])
#   else:
#     index = VectorstoreIndexCreator().from_loaders([loader])
#

# directory = "AFO_1699527039/"
#
# raw_text_data = read_files_in_folder(directory)
#
# for file_name, content in raw_text_data.items():
#   chunks = get_chunks(content)
#   # print(f"File: {file_name}, Number of chunks: {len(chunks)}")
#   vector_store.add_texts(chunks)


# for file_name, content in raw_text_data.items():
#   print(f"File: {file_name}, Content: {content}")

chain = ConversationalRetrievalChain.from_llm(
  llm=ChatOpenAI(model="gpt-3.5-turbo", temperature="0", streaming="True", max_retries=2),
  # retriever from repo
  # retriever=index.vector_store.as_retriever(search_kwargs={"k": 1}),
  retriever=vector_store.as_retriever(search_kwargs={"k": 1})
)

qa = RetrievalQA.from_chain_type(
  llm=ChatOpenAI(model="gpt-4-32k-0613"),
  chain_type="stuff",
  retriever=vector_store.as_retriever(search_kwargs={"k": 1})
)

chat_history = []
while True:
  if not query:
    query = input("Prompt: ")
  if query in ['quit', 'q', 'exit']:
    sys.exit()
  print("loading chain")
  result = chain({"question": query, "chat_history": chat_history})
  # res = qa.run(query)
  print("loaded chain")
  # print(res)
  print(result['answer'])
  chat_history.append((query, result['answer']))
  # chat_history.append((query, res))
  query = None
