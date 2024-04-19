import asyncio
from contextlib import asynccontextmanager
import signal
from dotenv import load_dotenv
from fastapi.responses import StreamingResponse
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.vectorstores import Chroma
import os
from fastapi import FastAPI, UploadFile, File
from typing import List,Optional
from langchain.llms import Ollama
import shutil
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from langchain.callbacks.base import BaseCallbackHandler


import sys
from typing import Any, Dict, List, Union

from langchain.callbacks.base import BaseCallbackHandler
from langchain.schema import AgentAction, AgentFinish, LLMResult
from langchain.schema.messages import BaseMessage
from fastapi import FastAPI,Depends
from fastapi.responses import Response
from time import time,sleep,localtime
import multiprocessing
import random
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the ML model
    # ml_models["answer_to_everything"] = fake_answer_to_everything_ml_model
    yield
    # Clean up the ML models and release the resources
    print("closing")
    model_tts.closing()
    os.kill(os.getpid(), signal.SIGTERM)
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
    where:str

class QueryEmbedData(BaseModel):
    files: List[str]
    # collection_name: str
    
from langchain.prompts import PromptTemplate
load_dotenv()

embeddings_model_name = os.environ.get("EMBEDDINGS_MODEL_NAME")
persist_directory = os.environ.get('PERSIST_DIRECTORY')
model = os.environ.get("MODEL", "llama3")
target_source_chunks = int(os.environ.get('TARGET_SOURCE_CHUNKS',1))
source_directory = os.environ.get('SOURCE_DIRECTORY', 'source_documents')
base_url = os.environ.get('OLLAMA_URL', 'http://localhost:11434')

template = """You're a helpful AI assistant. Use the following pieces of context to answer the question at the end. 
If you don't know the answer, just say that you don't know, don't try to make up an answer. Always start your reply with "thanks".
{context}
Question: {question}
Helpful Answer:"""

QA_CHAIN_PROMPT = PromptTemplate(
    template=template, input_variables=["context", "question"]
)
# class MItem(BaseModel):
#     def __init__(self, from_: str, message: str, time: str, timestamp: float):
#         self.from_ = from_
#         self.message = message
#         self.time = time
#         self.timestamp = timestamp

# my_array = [MItem(from_, message, time, timestamp) for from_, message, time, timestamp in [("user1", "Hello", "12:00", 123.45), ("user2", "World", "13:00", 678.90)]]

quit_stream = False
from constants import CHROMA_SETTINGS
@app.get("/updates")
async def updates(message: str = "", delay: float = 1.0):
    """
    Returns a Server-Sent Events (SSE) stream of updates.
    """
    async def generate():
        while True:
            
            yield f"data: {time()} {message}\n\n"
            await asyncio.sleep(1)

    return StreamingResponse(generate(), media_type="text/event-stream")


# @app.post("/save_mitem/")
# async def save_mitem(mitem: MItem):
#     with open("mitems.json", "a") as f:
#         f.write(json.dumps(mitem.dict()) + "\n")
#     return {"message": "MItem saved successfully"}

class QueryD(BaseModel):
    text: str
    comments:str
from melo.api import TTS

# Speed is adjustable
speed = 0.8

# CPU is sufficient for real-time inference.
# You can set it manually to 'cpu' or 'cuda' or 'cuda:0' or 'mps'
device = 'auto' # Will automatically use GPU if available

# English 
model_tts = TTS(language='EN_V2', device=device)
# print(model)
speaker_ids = model_tts.hps.data.spk2id

@app.post("/tts")
async def tts(query: QueryD):
    print(query.text)

    # American accent
    # output_path = 'en-us.wav'
    # model.tts_to_file(text, speaker_ids['EN-US'], output_path, speed=speed)

    # British accent
    model_tts.tts_to_sound(query.text, speaker_ids['EN-BR'], speed=speed)
    
    return {"response":"text recieved"}

from fastapi.responses import JSONResponse
# Example route
@app.get("/", response_class=JSONResponse)
async def root():
    print("hello")
    return {"message": "Hello, the FiledimeGPT APIs are now ready for your embeds and queries!"}

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


@app.post("/embedfromremote")
async def embedfromremote(files: List[UploadFile], collection_name: Optional[str] = None):
    file_paths=read_file_paths_from_txt("source_documents/paths.txt")
    already_files = [file_path for file_path in file_paths]
    saved_files = [os.path.join(source_directory, file.filename) for file in files]
    all_in_b = all(x in already_files for x in saved_files)
    if(all_in_b):
        return {"message": "Files already analyzed(embedded)", "saved_files": saved_files}

    if os.path.exists(persist_directory):
        print(f"Clearing existing vectorstore at {persist_directory}")
        shutil.rmtree(persist_directory)


    saved_files = []
    # Save the files to the specified folder
    for file in files:
        file_path = os.path.join(source_directory, file.filename)
        saved_files.append(file_path) 
        
        with open(file_path, "wb") as f:
            f.write(await file.read())
        
        # if collection_name is None:
        #     # Handle the case when the collection_name is not defined
        #     collection_name = file.filename

    save_paths_to_file(saved_files)
    
    exit_code=os.system(f'python3 ingest.py --collection test')
    # main("test")
    
    
    # Delete the contents of the folder
    [os.remove(os.path.join(source_directory, file.filename)) or os.path.join(source_directory, file.filename) for file in files]
    if exit_code == 0:
        return {"message": "Files embedded successfully", "saved_files": saved_files}
    else:
        save_paths_to_file([])
        return {"message": "Files embeddeing failed", "saved_files": []}

    

def read_file_paths_from_txt(file_path: str) -> List[str]:
    with open(file_path, 'r') as file:
        file_paths = [line.strip() for line in file.readlines()]
    return file_paths

from ingest import main
@app.post("/embed")
async def embed(files: QueryEmbedData):
    file_paths=read_file_paths_from_txt("source_documents/paths.txt")
    already_files = [file_path for file_path in file_paths]
    saved_files = files.files
    all_in_b = all(x in already_files for x in saved_files)
    if(all_in_b):
        return {"message": "Files already analyzed(embedded)", "saved_files": saved_files}

    # return {"message": "Files embedded successfully", "saved_files": [saved_files]}
    # Delete the embeddings folder
    if os.path.exists(persist_directory):
        print(f"Clearing existing vectorstore at {persist_directory}")
        shutil.rmtree(persist_directory)
        # return {"message": "cleaning existing embeddings"}

    # Prepare a list to store the file paths
    saved_files = files.files

    # Write the paths of the uploaded files to paths.txt
    save_paths_to_file(files.files)
    
    # os.system(f'python3 ingest.py --collection test')
    main("test")
    
    # Delete the contents of the folder
    # [os.remove(os.path.join(source_directory, file.filename)) or os.path.join(source_directory, file.filename) for file in files]
    
    return {"message": "Files embedded successfully", "saved_files": saved_files}

# def queryhandle(query:QueryData):
#     # return {"hello":"test"}
    
    
from time import time,sleep,localtime,perf_counter
from threading import Thread
from langchain.callbacks import AsyncIteratorCallbackHandler
start_time=perf_counter()
from torch import cuda

def findres(query):
    start_time = perf_counter()
    embeddings = HuggingFaceEmbeddings(model_name=embeddings_model_name,model_kwargs={"device":"cuda"} if cuda.is_available() else {})
    
    db = Chroma(persist_directory=persist_directory, embedding_function=embeddings,collection_name="test")

    retriever = db.as_retriever(search_kwargs={"k": target_source_chunks})

    llm = Ollama(model="llama2",base_url=base_url)
    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, return_source_documents= True)
    res = qa(query) 
    print(perf_counter()-start_time)
    print(res)   
    answer, docs = res['result'], res['source_documents']
    print(answer)
    return answer,docs



@app.post("/retrieve")
async def retrieve(query: QueryData):
    
    answer,docs=findres(query.query)
    return {"results": answer, "docs":docs}
    

    # embed(["/home/ubroger/Documents/GitHub/filegpt-filedime/1.txt",
# "/home/ubroger/Documents/GitHub/filegpt-filedime/requirements.txt"])

# curl -X POST http://localhost:8080/embed -H "Content-Type: application/json" -d '{"files": ["/home/ubroger/Documents/GitHub/filegpt-filedime/1.txt","/home/ubroger/Documents/GitHub/filegpt-filedime/requirements.txt"]}'
from collections.abc import Generator
class QueueCallback(BaseCallbackHandler):
    """Callback handler for streaming LLM responses to a queue."""

    def __init__(self, q):
        self.q = q

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.q.put(token)

    def on_llm_end(self, *args, **kwargs) -> None:
        return self.q.empty()

from queue import Queue, Empty

def stream(cb, q) -> Generator:
    job_done = object()

    def task():
        x = cb()
        # print(x['source_documents'])
        q.put(job_done)

    t = Thread(target=task)
    t.start()

    content = ""

    # Get each new token from the queue and yield for our generator
    while True:
        try:
            next_token = q.get(True, timeout=1)
            if next_token is job_done:
                break
            content += next_token
            yield next_token, content
        except Empty:
            continue
import json
from langchain.chat_models import ChatOllama
from langchain.schema.document import Document
from pydantic.json import pydantic_encoder

@app.post("/query-stream")
def qstream(query:QueryData ):
    print(query)
    start_time = perf_counter()

    global quit_stream
    quit_stream=False
    # print(query.query)
    embeddings = HuggingFaceEmbeddings(model_name=embeddings_model_name,model_kwargs={"device":"cuda"} if cuda.is_available() else {})
    
    db = Chroma(persist_directory=persist_directory, embedding_function=embeddings,collection_name="test")

    retriever = db.as_retriever(search_kwargs={"k": target_source_chunks})
    q = Queue()
    if query.where=="ollama":
        llm = Ollama(model=model,callbacks=[QueueCallback(q)])
    else:
        llm = Ollama(model="llama2",callbacks=[QueueCallback(q)])
    
    output_function = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, return_source_documents= True)
    
    def cb():
        if query.where=="ollama":
            llm.generate(prompts=[query.query],stop=["<|eot_id|>"]) #to query in ollama
        else:
            try:
                output_function(query.query) #to query in context
            except Exception as e:
                print("error" + e)

    def generate():
        # yield json.dumps({"init": True, "model": llm_name})
        try:
            for token, _ in stream(cb, q):
            # print()
            # print( f"data: {perf_counter()-start_time} {token} \n\n")
                yield json.dumps({"token":token})
            # yield json.dumps({"token": token})
        # source_documents = output_function(query.query)['source_documents']
        # print(source_documents)
        # yield json.dumps({"token": "Source--"+"\n".join(json.dumps(doc.__dict__, default=str) for doc in source_documents)})
        except Exception as e:
            yield json.dumps({"token":"error"+e})
        yield json.dumps({"token":"[DONESTREAM]"})
    return EventSourceResponse(generate(), media_type="text/event-stream")

from sse_starlette.sse import EventSourceResponse


import uvicorn
if __name__ == "__main__":
    # findres("what are the contents")
    # findres("how far is the sun from earth")
    uvicorn.run(app, host="0.0.0.0", port=8694, timeout_keep_alive=1500)
