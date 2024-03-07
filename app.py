from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.vectorstores import Chroma
import os
from fastapi import FastAPI, UploadFile, File
from typing import List
from langchain.llms import Ollama
import shutil
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Allow all origins, adjust as needed
    allow_credentials=True,
    allow_methods=["*"], # Allow all methods
    allow_headers=["*"], # Allow all headers
)
class QueryData(BaseModel):
    query: str

class QueryEmbedData(BaseModel):
    files: List[str]
    # collection_name: str

load_dotenv()

embeddings_model_name = os.environ.get("EMBEDDINGS_MODEL_NAME")
persist_directory = os.environ.get('PERSIST_DIRECTORY')
model = os.environ.get("MODEL", "llama2")
target_source_chunks = int(os.environ.get('TARGET_SOURCE_CHUNKS',4))
source_directory = os.environ.get('SOURCE_DIRECTORY', 'source_documents')

from constants import CHROMA_SETTINGS

# Example route
@app.get("/")
async def root():
    return {"message": "Hello, the APIs are now ready for your embeds and queries!"}

@app.get("/clear")
async def clear():
     if os.path.exists(persist_directory):
        print(f"Clearing existing vectorstore at {persist_directory}")
        shutil.rmtree(persist_directory)
     return {"message": "cleaning existing embeddings"}

def save_paths_to_file(json_data):
    with open('source_documents/paths.txt', 'w') as file:
        # Iterate through the list of file paths
        for path in json_data:
            # Write each path to the file
            file.write(path + '\n')

# @app.post("/test")
# async def test(files: QueryEmbedData):
#     print(files.files)
#     with open('source_documents/paths.txt', 'w') as file:
#         # Iterate through the list of file paths
#         for path in files.files:
#             # Write each path to the file
#             file.write(path + '\n')
    
    
@app.post("/embed")
async def embed(files: QueryEmbedData):
    # Delete the embeddings folder
    if os.path.exists(persist_directory):
        print(f"Clearing existing vectorstore at {persist_directory}")
        shutil.rmtree(persist_directory)
        # return {"message": "cleaning existing embeddings"}

    # Prepare a list to store the file paths
    saved_files = files.files

    # Write the paths of the uploaded files to paths.txt
    save_paths_to_file(files.files)
    
    os.system(f'python3 ingest.py ')
    
    # Delete the contents of the folder
    # [os.remove(os.path.join(source_directory, file.filename)) or os.path.join(source_directory, file.filename) for file in files]
    
    return {"message": "Files embedded successfully", "saved_files": saved_files}

@app.post("/retrieve")
async def query(data: QueryData):
    question=data.query
    embeddings = HuggingFaceEmbeddings(model_name=embeddings_model_name)

    db = Chroma(persist_directory=persist_directory, embedding_function=embeddings)

    retriever = db.as_retriever(search_kwargs={"k": target_source_chunks})
    callbacks = [StreamingStdOutCallbackHandler()]

    llm = Ollama(model=model, callbacks=callbacks)

    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, return_source_documents= True)
    res = qa(question)
    print(res)   
    answer, docs = res['result'], res['source_documents']


    return {"results": answer, "docs":docs}


# embed(["/home/ubroger/Documents/GitHub/filegpt-filedime/1.txt",
# "/home/ubroger/Documents/GitHub/filegpt-filedime/requirements.txt"])

# curl -X POST http://localhost:8080/embed -H "Content-Type: application/json" -d '{"files": ["/home/ubroger/Documents/GitHub/filegpt-filedime/1.txt","/home/ubroger/Documents/GitHub/filegpt-filedime/requirements.txt"]}'
