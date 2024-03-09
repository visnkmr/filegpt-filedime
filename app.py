from dotenv import load_dotenv
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

class StreamHandler(BaseCallbackHandler):
    def on_llm_start(
        self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any
    ) -> None:
        """Run when LLM starts running."""

    def on_chat_model_start(
        self,
        serialized: Dict[str, Any],
        messages: List[List[BaseMessage]],
        **kwargs: Any
    ) -> None:
        """Run when LLM starts running."""

    def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        """Run on new LLM token. Only available when streaming is enabled."""
        print(token)

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        """Run when LLM ends running."""

    def on_llm_error(
        self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any
    ) -> None:
        """Run when LLM errors."""

    def on_chain_start(
        self, serialized: Dict[str, Any], inputs: Dict[str, Any], **kwargs: Any
    ) -> None:
        """Run when chain starts running."""

    def on_chain_end(self, outputs: Dict[str, Any], **kwargs: Any) -> None:
        """Run when chain ends running."""

    def on_chain_error(
        self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any
    ) -> None:
        """Run when chain errors."""

    def on_tool_start(
        self, serialized: Dict[str, Any], input_str: str, **kwargs: Any
    ) -> None:
        """Run when tool starts running."""

    def on_agent_action(self, action: AgentAction, **kwargs: Any) -> Any:
        """Run on agent action."""
        pass

    def on_tool_end(self, output: str, **kwargs: Any) -> None:
        """Run when tool ends running."""

    def on_tool_error(
        self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any
    ) -> None:
        """Run when tool errors."""

    def on_text(self, text: str, **kwargs: Any) -> None:
        """Run on arbitrary text."""

    def on_agent_finish(self, finish: AgentFinish, **kwargs: Any) -> None:
        """Run on agent end."""
 

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
target_source_chunks = int(os.environ.get('TARGET_SOURCE_CHUNKS',1))
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
@app.post("/embedfromremote")
async def embed(files: List[UploadFile], collection_name: Optional[str] = None):
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
        
        if collection_name is None:
            # Handle the case when the collection_name is not defined
            collection_name = file.filename

    save_paths_to_file(saved_files)
    
    os.system(f'python3 ingest.py {collection_name}')
    
    # Delete the contents of the folder
    [os.remove(os.path.join(source_directory, file.filename)) or os.path.join(source_directory, file.filename) for file in files]
    
    return {"message": "Files embedded successfully", "saved_files": saved_files}

@app.post("/embed")
async def embed(files: QueryEmbedData):
    saved_files = files.files
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
    
    os.system(f'python3 ingest.py ')
    
    # Delete the contents of the folder
    # [os.remove(os.path.join(source_directory, file.filename)) or os.path.join(source_directory, file.filename) for file in files]
    
    return {"message": "Files embedded successfully", "saved_files": saved_files}

@app.post("/retrieve")
async def query(data: QueryData):
    question=data.query
    # return {"results": question, "docs":data}
    embeddings = HuggingFaceEmbeddings(model_name=embeddings_model_name,model_kwargs={"device":"cuda"})

    db = Chroma(persist_directory=persist_directory, embedding_function=embeddings)

    retriever = db.as_retriever(search_kwargs={"k": target_source_chunks})
    # message_placeholder = st.empty()
    stream_handler = StreamHandler()  
    callbacks = [stream_handler]

    llm = Ollama(model=model, callbacks=callbacks)

    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, return_source_documents= True)
    res = qa(question)
    # print(res)   
    answer, docs = res['result'], res['source_documents']
    print(answer)
    return {"results": answer, "docs":docs}


# embed(["/home/ubroger/Documents/GitHub/filegpt-filedime/1.txt",
# "/home/ubroger/Documents/GitHub/filegpt-filedime/requirements.txt"])

# curl -X POST http://localhost:8080/embed -H "Content-Type: application/json" -d '{"files": ["/home/ubroger/Documents/GitHub/filegpt-filedime/1.txt","/home/ubroger/Documents/GitHub/filegpt-filedime/requirements.txt"]}'
