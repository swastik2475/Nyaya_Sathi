from fastapi import FastAPI
from pydantic import BaseModel
from rag import ask_llm

app = FastAPI(title="NyayaSathi – Legal AI")

class Query(BaseModel):
    question: str

@app.post("/query")
def query_legal_ai(q: Query):
    answer = ask_llm(q.question)
    return {"answer": answer}
