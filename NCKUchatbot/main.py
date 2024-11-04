from fastapi import FastAPI
from pydantic import BaseModel
from ncku_adaptive_RAG_chatbot import run

app = FastAPI()

# Request model
class QuestionRequest(BaseModel):
    question: str
    session_id: str

# Response model
class AnswerResponse(BaseModel):
    answer: str

@app.post("/ask", response_model=AnswerResponse)
async def ask_question(request: QuestionRequest):
    inputs = {"question": request.question, "session_id": request.session_id}

    # Stream output
    output = run(inputs['question'], inputs["session_id"])

    return {"answer": output}

# Run this with Uvicorn if this file is named `main.py`
# uvicorn main:app --reload
