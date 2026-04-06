from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma

# 1. Load documents from the ./docs folder
loader = DirectoryLoader(
    "./docs",
    glob="**/*.pdf",        # change to "**/*.txt" for plain text files
    loader_cls=PyPDFLoader
)
documents = loader.load()
print(f"Loaded {len(documents)} document(s)")

# 2. Split into chunks
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,       # characters per chunk
    chunk_overlap=50      # overlap to preserve context at boundaries
)
chunks = splitter.split_documents(documents)
print(f"Created {len(chunks)} chunks")

# 3. Embed and store in ChromaDB
embeddings = OllamaEmbeddings(model="nomic-embed-text")

vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./chroma_db"   # saves to disk
)

print("Documents ingested and stored in ChromaDB.")