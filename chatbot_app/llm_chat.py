from langchain_openai import OpenAIEmbeddings 
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI 
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain.chains import create_history_aware_retriever
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts import MessagesPlaceholder


# get api key
import os
from dotenv import load_dotenv
load_dotenv()

OPENAI_API_KEY = os.getenv('CHATGPT_API_KEY')

# init model
embeddings_model = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
llm_model = ChatOpenAI(openai_api_key=OPENAI_API_KEY, temperature=0, model="gpt-3.5-turbo") 

new_db = FAISS.load_local("faiss_index", embeddings_model, allow_dangerous_deserialization=True)
retriever = new_db.as_retriever()



prompt_search_query = ChatPromptTemplate.from_messages([
    MessagesPlaceholder(variable_name="chat_history"),
    ("user", "{input}"),
    ("user", "Given the above conversation, generate a search query to look up in order to get information relevant to the conversation")
])

retriever_chain = create_history_aware_retriever(llm_model, retriever, prompt_search_query)

prompt_get_answer = ChatPromptTemplate.from_messages([
    ('system', 'Answer the user\'s questions based on the below context:\n\n{context}'),
    MessagesPlaceholder(variable_name="chat_history"),
    ('user', '{input}'),
])

document_chain = create_stuff_documents_chain(llm_model, prompt_get_answer)

retrieval_chain_combine = create_retrieval_chain(retriever_chain, document_chain)



chat_history = []

def ask_question(query: str, chat_history: list) -> str:
    """
    向 LLM 提問並返回回應。

    :param query: 要提問的問題
    :param chat_history: 之前的聊天記錄
    :return: LLM 的回答
    """
    response = retrieval_chain_combine.invoke({"input": query, "chat_history": chat_history})
    chat_history.append(HumanMessage(content=query))
    chat_history.append(AIMessage(content=response['answer']))
    return response['answer']


query = "請列出資料結構這堂課的分數百分比"
response = retrieval_chain_combine.invoke({"input": query, "chat_history": chat_history}) 
print(response['answer'])
chat_history.append(HumanMessage(content=query))
chat_history.append(AIMessage(content=response['answer']))

query2 = "那這堂課的課程描述是什麼"
response2 = retrieval_chain_combine.invoke({
    "input": query2,
    "chat_history": chat_history
}) 
print(response2['answer']) 