from fastapi import FastAPI
from model import get_embeddings
from pydantic import BaseModel

app = FastAPI()

class TextChunk(BaseModel):
    text: str

@app.post("/get_embeddings/")
def get_embeddings_route(text_chunk: TextChunk):
    embeddings = get_embeddings(text_chunk.text)
    return embeddings
