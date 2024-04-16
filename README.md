FiledimeGPT is made to work out of the box with Filedime, remotely or locally.
#### Thanks to (https://github.com/imartinez/privateGPT), (https://github.com/jmorganca/ollama), (https://github.com/myshell-ai/MeloTTS) and (https://github.com/menloparklab/privateGPT-app)

#### Make sure to have Ollama running on your system from https://ollama.ai and by default it expects ollama to be running on port 11434
#### Step: Download/pull models (if you already have models loaded in Ollama, then not required)
```
ollama pull llama2
```
#### Step: Clone the github repo and navigate into it
```
git clone https://github.com/visnkmr/filegpt-filedime && cd filegpt-filedime
```
#### Step: Setup a Python Virtual Environment
```
python3 -m venv filedimegpt
```

#### Step: Activate the Python Virtual Environment
```
source ./filedimegpt/bin/activate
```

#### Step: Install the Requirements
```
pip install -r requirements.txt
```


#### Modify the env file as neccesary by default it has the following values:
```
PERSIST_DIRECTORY=db
MODEL_TYPE=llama2
EMBEDDINGS_MODEL_NAME=all-MiniLM-L6-v2
MODEL_N_CTX=1000
SOURCE_DIRECTORY=source_documents
OLLAMA_URL=http://localhost:11434
```

#### Step: Run this command
```
python app.py  

```

The supported extensions are:

- `.csv`: CSV,
- `.docx`: Word Document,
- `.doc`: Word Document,
- `.enex`: EverNote,
- `.eml`: Email,
- `.epub`: EPub,
- `.html`: HTML File,
- `.md`: Markdown,
- `.msg`: Outlook Message,
- `.odt`: Open Document Text,
- `.pdf`: Portable Document Format (PDF),
- `.pptx` : PowerPoint Document,
- `.ppt` : PowerPoint Document,
- `.txt`: Text file (UTF-8),
