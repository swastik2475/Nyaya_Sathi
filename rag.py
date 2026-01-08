from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

db = FAISS.load_local(
    "faiss_index",
    embeddings,
    allow_dangerous_deserialization=True
)


retriever = db.as_retriever(search_kwargs={"k": 3})


llm = ChatOllama(
    model="gemma2:2b",
    temperature=0.2
)

def ask_llm(question: str):
    docs = retriever.invoke(question)
    context = "\n\n".join(d.page_content for d in docs)

    prompt = f"""
You are NyayaSathi, an AI legal assistant for Indian law.

Use ONLY the context below.
If the answer is not present, say you don't have enough information.

Context:
{context}

Question:
{question}
"""

    return llm.invoke(prompt).content
