# document-chatbot
Query any PDF document using Langchain, Huggingface and Openai.

## Initial setup
Start by cloning this repo to your directory,
```
git clone https://github.com/jsluo413/document-chatbot.git
cd document-chatbot
```
Then, assuming you have Anaconda installed in your computer, create a new environment and install the requirements.
```
conda create -n chatbot python=3.9
conda activate chatbot
pip install -r  requirements.txt
```

## Create index 

You can use our example documents in `docs/cs229` or Web URLs

example: 

```
python index.py "docs/cs229"
```

or,

```
python index.py "https://lilianweng.github.io/posts/2023-06-23-agent/"
```

## Bot UI

You're all set to start interacting with your personal chatbot in Gradio UI. Just type in your queries or instructions, and watch as it provides responses or performs tasks for you.

```
python bot.py
```