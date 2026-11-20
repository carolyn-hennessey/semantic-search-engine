EMBEDDING_MODEL_NAME = "thenlper/gte-small"

# load dataset
def load_text():
    text = ""
    with open("./text/input.txt", "r", encoding="utf-8") as f:
        text = f.read()

    return text

def generate_chunks(text):
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,      # Max characters per chunk
        chunk_overlap=200,    # Overlap to preserve context between chunks
        separators=["\n\n", "\n", " ", ""]
    )

    chunks = splitter.split_text(text)

    return chunks


# Not required by Chromadb by default
def generate_embeddings(chunks):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL_NAME)
    embeddings = [tokenizer.encode(chunk) for chunk in chunks]

    return embeddings

def load_chroma(embeddings):
    import chromadb

    ids = []

    for num, _ in enumerate(range(0, 46)):
        ids.append("doc" + str(num))

    # 1. Initialize an in-memory client (use PersistentClient(path="./db") to save to disk)
    client = chromadb.Client()

    # 2. Create or load a collection
    collection = client.get_or_create_collection(name="my_docs")

    # 3. Add documents (automatically embedded by default)
    collection.add(
        documents=embeddings,
        ids=ids
    )

    results = collection.query(
        query_texts=["What are microservices?"],
        n_results=3
    )

    print(f"RESULT: {results}")

    return collection


# if __name__ == "__main__":
#     "loading text "
#     text = load_text()
#     "generating chunks "
#     chunks = generate_chunks(text)
#     "generating embeddings"
#     embeddings = generate_embeddings(chunks)
#     load_chroma(chunks)

