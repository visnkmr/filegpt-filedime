
#!/usr/bin/env python3
import os
import glob
from typing import List
from multiprocessing import Pool
from tqdm import tqdm

from langchain.document_loaders.unstructured import (UnstructuredFileLoader)
from langchain.document_loaders import (
    CSVLoader,
    UnstructuredExcelLoader,
    EverNoteLoader,
    PyMuPDFLoader,
    TextLoader,
    PythonLoader,
    UnstructuredODTLoader,
    UnstructuredEmailLoader,
    UnstructuredEPubLoader,
    UnstructuredHTMLLoader,
    UnstructuredMarkdownLoader,
    UnstructuredODTLoader,
    UnstructuredPowerPointLoader,
    JSONLoader,
    UnstructuredWordDocumentLoader
)

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.docstore.document import Document
from constants import CHROMA_SETTINGS


# Load environment variables
persist_directory = os.environ.get('PERSIST_DIRECTORY', 'db')
source_directory = os.environ.get('SOURCE_DIRECTORY', 'source_documents')
# embeddings_model_name = os.environ.get('EMBEDDINGS_MODEL_NAME', 'mixedbread-ai/mxbai-embed-large-v1')
embeddings_model_name = os.environ.get('EMBEDDINGS_MODEL_NAME', 'all-MiniLM-L6-v2')
chunk_size = 500
chunk_overlap = 50

# Custom document loaders
class MyElmLoader(UnstructuredEmailLoader):
    """Wrapper to fallback to text/plain when default does not work"""

    def load(self) -> List[Document]:
        """Wrapper adding fallback for elm without html"""
        try:
            try:
                doc = UnstructuredEmailLoader.load(self)
            except ValueError as e:
                if 'text/html content not found in email' in str(e):
                    # Try plain text
                    self.unstructured_kwargs["content_source"]="text/plain"
                    doc = UnstructuredEmailLoader.load(self)
                else:
                    raise
        except Exception as e:
            # Add file_path to exception message
            raise type(e)(f"{self.file_path}: {e}") from e

        return doc


# Map file extensions to document loaders and their arguments
LOADER_MAPPING = {
    ".csv": (CSVLoader, {}),
    ".xlsx": (UnstructuredExcelLoader, {}),
    ".xls": (UnstructuredExcelLoader, {}),
    ".odt": (UnstructuredODTLoader, {}),
    ".py": (PythonLoader, {}),
    # ".json": (JSONLoader, {"jq_schema":"."}),
    # ".docx": (Docx2txtLoader, {}),
    ".doc": (UnstructuredWordDocumentLoader, {}),
    ".docx": (UnstructuredWordDocumentLoader, {}),
    ".enex": (EverNoteLoader, {}),
    ".eml": (MyElmLoader, {}),
    ".epub": (UnstructuredEPubLoader, {}),
    ".html": (UnstructuredHTMLLoader, {}),
    ".md": (UnstructuredMarkdownLoader, {}),
    ".odt": (UnstructuredODTLoader, {}),
    ".pdf": (PyMuPDFLoader, {}),
    ".ppt": (UnstructuredPowerPointLoader, {}),
    ".pptx": (UnstructuredPowerPointLoader, {}),
    ".txt": (TextLoader, {"encoding": "utf8"}),
    # Add more mappings for other file extensions and loaders as needed
}

import chardet
def load_single_document(file_path: str) -> List[Document]:
    ext = "." + file_path.rsplit(".", 1)[-1]
    if ext in LOADER_MAPPING:
        loader_class, loader_args = LOADER_MAPPING[ext]
        loader = loader_class(file_path, **loader_args)
        return loader.load()
    # else:
    #     with open(file_path, 'rb') as f:
    #         result = chardet.detect(f.read())["encoding"]
    #         print("-------------->"+result)

    #     metadata = {"source": file_path}
    #     text=""
    #     with open(file_path, 'r',encoding=result) as file:
    #         text=file.read()
    #     return [Document(page_content=text, metadata=metadata)]
        # loader = UnstructuredFileLoader(file_path)
        # return loader.load()
    

    raise ValueError(f"Unsupported file extension '{ext}'")

def load_documents(file_paths: List[str], ignored_files: List[str] = []) -> List[Document]:
    """
    Loads all documents from the specified file paths, ignoring specified files
    """
    # Filter out ignored files
    filtered_files = [file_path for file_path in file_paths if file_path not in ignored_files]

    with Pool(processes=os.cpu_count()) as pool:
        results = []
        with tqdm(total=len(filtered_files), desc='Loading new documents', ncols=80) as pbar:
            for i, docs in enumerate(pool.imap_unordered(load_single_document, filtered_files)):
                results.extend(docs)
                pbar.update()

    return results

# Function to read file paths from paths.txt
def read_file_paths_from_txt(file_path: str) -> List[str]:
    with open(file_path, 'r') as file:
        file_paths = [line.strip() for line in file.readlines()]
    return file_paths

def process_documents(ignored_files: List[str] = []) -> List[Document]:
    """
    Load documents and split in chunks
    """
    print(f"Loading documents from {source_directory}")
    file_paths = read_file_paths_from_txt('source_documents/paths.txt')
    # documents = load_documents(file_paths)
    documents = load_documents(file_paths, ignored_files)
    if not documents:
        print("No new documents to load")
        exit(0)
    print(f"Loaded {len(documents)} new documents from {source_directory}")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    texts = text_splitter.split_documents(documents)
    print(f"Split into {len(texts)} chunks of text (max. {chunk_size} tokens each)")
    return texts

def does_vectorstore_exist(persist_directory: str) -> bool:
    """
    Checks if vectorstore exists
    """
    if os.path.exists(os.path.join(persist_directory, 'index')):
        if os.path.exists(os.path.join(persist_directory, 'chroma-collections.parquet')) and os.path.exists(os.path.join(persist_directory, 'chroma-embeddings.parquet')):
            list_index_files = glob.glob(os.path.join(persist_directory, 'index/*.bin'))
            list_index_files += glob.glob(os.path.join(persist_directory, 'index/*.pkl'))
            # At least 3 documents are needed in a working vectorstore
            if len(list_index_files) > 3:
                return True
    return False
from torch import cuda
from langchain.llms import Ollama
from langchain.chains import RetrievalQA

def main():
    

    # Create embeddings
    embeddings = HuggingFaceEmbeddings(model_name=embeddings_model_name,model_kwargs={"device":"cuda"}  if cuda.is_available() else {})

    # if does_vectorstore_exist(persist_directory):
    #     # Update and store locally vectorstore
    #     print(f"Appending to existing vectorstore at {persist_directory}")
    #     db = Chroma(persist_directory=persist_directory, embedding_function=embeddings,collection_name=collection, client_settings=CHROMA_SETTINGS)
    #     collection = db.get()
    #     print(collection)
    #     texts = process_documents([metadata['source'] for metadata in collection['metadatas']])
    #     print(f"Creating embeddings. May take some minutes...")
    #     db.add_documents(texts)
    # else:
        # Create and store locally vectorstore
    print("Creating new vectorstore")
    texts = process_documents()
    print(f"Creating embeddings. May take some minutes...")
    db = Chroma.from_documents(texts, embeddings, persist_directory=persist_directory)
    db.persist()
    # db = None

    print(f"Ingestion complete! You can now run privateGPT.py/use the /retrieve route to query your documents")
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from vlite.utils import process_pdf,process_file
# from vlite import VLite
from vlite.vlite import VLite
from langchain_community.chat_models import ChatOpenAI
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain import hub
from langchain_core.prompts import ChatPromptTemplate
import argparse
if __name__ == "__main__":
    from langchain_core.pydantic_v1 import Field
    metadata: dict = Field(default_factory=dict)
    # metadata={'source': '/home/ubroger/Documents/GitHub/filegpt-filedime/requirements.txt'}
    print(metadata)
    embeddings = HuggingFaceEmbeddings(model_name=embeddings_model_name,
                                       model_kwargs={"device": "cuda"} if cuda.is_available() else {})
    vlite=VLite(embedding_function=embeddings)
    # vlite=VLite(model_name=embeddings_model_name)
 # Create the argument parser
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--collection", help="Saves the embedding in a collection name as specified")

    # # Parse the command-line arguments
    # args = parser.parse_args()
    print("Creating new vectorstore")
    print(f"Loading documents from {source_directory}")
    texts=process_documents()
    db = vlite.from_documents(documents=texts,embedding=embeddings)
    retriever = db.as_retriever(search_kwargs={"k": 1})
    # from langchain_core.vectorstores import VectorStoreRetriever
    # i=VectorStoreRetriever(vectorstore=db)
    # print(i)
    # res=db.vlite.retrieve(
    #     text="is langchain installed",
    #     top_k=1,
    #     return_scores=True,
    #     embedding=None,
    # )
    # db.similarity_search_with_score(query="is langchain installed",k=1)
    # print(retriever)
    # results=db.vlite.retrieve(
    #     text="is langchain installed",
    #     top_k=1,
    #     return_scores=True,
    #     embedding=embeddings,
    # )
    # import json
    # parsed_data = json.loads(results)
    # print(parsed_data[0])
    # print(type(results))
    # retriever = vlite.retrieve()
    # print(retriever)
    # prompt = ChatPromptTemplate.from_template("""Answer the following question based only on the provided context:
    #
    # <context>
    # {context}
    # </context>
    #
    # Question: {input}""")
    # llm = Ollama(model="llama3",base_url='http://localhost:11434')
    # document_chain = create_stuff_documents_chain(llm, prompt)  # chain the LLM to the prompt
    # retrieval_chain = create_retrieval_chain(retriever, document_chain)
    # # qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, return_source_documents= True)
    # response = retrieval_chain.invoke({"input": "is langchain installed"})
    # print(response["answer"])

    # embeddings = HuggingFaceEmbeddings(model_name=embeddings_model_name,model_kwargs={"device":"cuda"}  if cuda.is_available() else {})
    # db = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
    # db.persist()
    # retriever = db.as_retriever(search_kwargs={"k": 1})

    llm = Ollama(model="llama3",base_url='http://localhost:11434')
    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, return_source_documents= True)
    res = qa("is langchain installed")
    # res = qa("What does this convey about the author")
    print(res)


    # lite = VLite(model_name="sentence-transformers/all-MiniLM-L6-v2")
    # file_paths = read_file_paths_from_txt('source_documents/paths.txt')
    # results = []
    # llm = Ollama(model="llama3", base_url='http://localhost:11434')
    # with Pool(processes=os.cpu_count()) as pool:
    #     with tqdm(total=len(file_paths), desc='Loading new documents', ncols=80) as pbar:
    #         for i, docs in enumerate(pool.imap_unordered(process_file, file_paths)):
    #             results.extend(docs)
    #             pbar.update()
    # print(results)
    # db = vlite.add(results)
    # retriever = query_constructor.load_query_constructor_runnable(llm=llm, document_contents=vlite.retrieve(
    #     text="is langchain installed", embedding=embeddings, top_k=1))

    # lite=VLite(model_name="sentence-transformers/all-MiniLM-L6-v2")
    # print(f"Loading documents from {source_directory}")
    # file_paths = read_file_paths_from_txt('source_documents/paths.txt')
    # results = []
    # with Pool(processes=os.cpu_count()) as pool:
    #     with tqdm(total=len(file_paths), desc='Loading new documents', ncols=80) as pbar:
    #         for i, docs in enumerate(pool.imap_unordered(process_file, file_paths)):
    #             results.extend(docs)
    #             pbar.update()
    # # print(texts)
    # db = vlite.add(results)
    # print(vlite.retrieve(text="is langchain installed",top_k=1))