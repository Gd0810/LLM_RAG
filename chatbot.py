from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# 1. Load the persisted vector store
embeddings = OllamaEmbeddings(model="nomic-embed-text")
vectorstore = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embeddings
)

# 2. Set up the retriever
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# 3. Prompt template
prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""You are a helpful assistant. Use ONLY the context below to answer.
If the answer isn't in the context, say "I don't know based on the provided documents."

Context:
{context}

Question: {question}

Answer:"""
)

# 4. LLM
llm = OllamaLLM(model="llama3.2")

# 5. Build chain using modern LCEL syntax
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# 6. Chat loop
print("RAG Chatbot ready. Type 'quit' to exit.\n")
while True:
    query = input("You: ").strip()
    if query.lower() in ("quit", "exit"):
        break
    if not query:
        continue

    answer = rag_chain.invoke(query)
    print(f"\nBot: {answer}\n")