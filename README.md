# Microservice Q&A: Semantic Search Engine

A small Streamlit app that answers questions about microservice architecture. It retrieves the most relevant passage from a ChromaDB collection and passes it, along with the question, to a small Hugging Face LLM that writes the answer.

## How it works

1. `preprocessing.py` reads `text/input.txt`, splits it into chunks, and loads them into a Chroma collection (embedded by Chroma's default model).
2. `app.py` takes the user's question and queries the collection for the best-matching chunk.
3. The question and chunk go to `HuggingFaceTB/SmolLM2-360M-Instruct`, which generates the response shown on the page.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install streamlit transformers torch chromadb langchain-text-splitters
```

## Run

```bash
streamlit run app.py
```

The first run downloads the embedding and language models, so it may take a few minutes.

## Files

| File | Purpose |
|------|---------|
| `app.py` | Streamlit UI and question-answering flow |
| `preprocessing.py` | Text loading, chunking, and Chroma setup |
| `text/input.txt` | Source text about microservices |

## Notes

- The Chroma collection is in-memory and rebuilt each time the app starts.
- `load_chroma` currently assumes exactly 46 chunks.
