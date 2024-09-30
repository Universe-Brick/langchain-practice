from fastapi import FastAPI, Request
from llm_chat import ask_question

app = FastAPI()

@app.post("/ask")
async def ask_chatbot(request: Request):
    data = await request.json()
    question = data.get("question")
    chat_history = data.get("chat_history", [])
    response = ask_question(question, chat_history)
    return {"response": response}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
