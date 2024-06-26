FiledimeGPT is made to work out of the box with Filedime, remotely or locally.

#### Step: Clone the github repo and navigate into it
```
git clone https://github.com/visnkmr/filegpt-filedime && cd filegpt-filedime
```

#### Step: Setup a Python Virtual Environment
Linux
```
python3 -m venv filedimegpt
```
Windows
```
py -m venv .venv 
```

#### Step: Activate the Python Virtual Environment
Linux
```
source ./filedimegpt/bin/activate
```
Windows
```
.\.venv\Scripts\activate   
```

#### Step: Install the Requirements
Linux
```
pip install -r requirements.txt
```
Windows
```
py -m pip install -r .\requirements.txt 
```

#### Step: Run this command
Linux
```
python app.py  
```
Windows
```
py .\app.py
```

If you only need filedimespeech then no need to install ollama which is described below,

#### Thanks to (https://github.com/imartinez/privateGPT), (https://github.com/jmorganca/ollama), (https://github.com/myshell-ai/MeloTTS) and (https://github.com/menloparklab/privateGPT-app)

#### Make sure to have Ollama running on your system from https://ollama.ai and by default it expects ollama to be running on port 11434
#### Step: Download/pull models (if you already have models loaded in Ollama, then not required)
```
ollama pull llama3
```

#### Modify the env file as neccesary by default it has the following values:
```
PERSIST_DIRECTORY=db
MODEL_TYPE=llama3
EMBEDDINGS_MODEL_NAME=all-MiniLM-L6-v2
MODEL_N_CTX=1000
SOURCE_DIRECTORY=source_documents
OLLAMA_URL=http://localhost:11434
```

The supported file extension that FileGPT can embed are:

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
