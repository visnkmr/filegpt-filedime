#### Inspired from (https://github.com/imartinez/privateGPT) and (https://github.com/jmorganca/ollama) and (https://github.com/menloparklab/privateGPT-app)

#### Step 1: Step a Virtual Environment

#### Step 2: Install the Requirements and add a dot in front of env file '.env'
```
pip install -r requirements.txt
```

#### Step 3: Pull the models (if you already have models loaded in Ollama, then not required)
#### Make sure to have Ollama running on your system from https://ollama.ai
```
ollama pull llama2
```
#### Step 4: Run this command (use python3 if on mac)
```
gunicorn app:app -k uvicorn.workers.UvicornWorker --timeout 1500 -b 0.0.0.0:8080

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
