from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# 1. Load the persisted vector store
embeddings = OllamaEmbeddings(model="nomic-embed-text")
vectorstore = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embeddings
)

# 2. Set up the retriever (fetch top 3 most relevant chunks)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# 3. Define a custom RAG prompt
prompt_template = PromptTemplate(
    input_variables=["context", "question"],
    template="""You are a helpful assistant. Use ONLY the context below to answer.
If the answer isn't in the context, say "I don't know based on the provided documents."

Context:
{context}

Question: {question}

Answer:"""
)

# 4. Connect to Llama 3.2 via Ollama
llm = OllamaLLM(model="llama3.2")

# 5. Build the RAG chain
rag_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",          # "stuff" = put all chunks into one prompt
    retriever=retriever,
    chain_type_kwargs={"prompt": prompt_template},
    return_source_documents=True
)

# 6. Chat loop
print("RAG Chatbot ready. Type 'quit' to exit.\n")
while True:
    query = input("You: ").strip()
    if query.lower() in ("quit", "exit"):
        break
    if not query:
        continue

    result = rag_chain.invoke({"query": query})
    print(f"\nBot: {result['result']}")

    # Optionally show which chunks were used
    print("\n[Sources]")
    for doc in result["source_documents"]:
        src = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page", "?")
        print(f"  - {src}, page {page}")
    print()