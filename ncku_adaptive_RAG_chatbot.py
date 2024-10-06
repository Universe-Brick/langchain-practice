from langchain_openai import OpenAIEmbeddings 
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI 
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain.chains import create_history_aware_retriever
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts import MessagesPlaceholder

from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.document_loaders import CSVLoader
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.output_parsers import StrOutputParser
from typing_extensions import TypedDict
from typing import List
from langchain.schema import Document
from langgraph.graph import END, StateGraph


from langsmith import Client
import os
import getpass
from dotenv import load_dotenv
load_dotenv()


# Langsmith setup
client = Client()

# API_key
def _set_env(var: str):
    if not os.getenv(var):
        os.environ[var] = getpass.getpass(f"{var}: ")


_set_env("CHATGPT_API_KEY")
_set_env("TAVILY_API_KEY")
OPENAI_API_KEY = os.getenv('CHATGPT_API_KEY')
TAVILY_API_KEY = os.getenv('TAVILY_API_KEY')

### Web search tool
web_search_tool = TavilySearchResults()

### Question Router
# 定義兩個工具的 DataModel
class web_search(BaseModel):
    """
    網路搜尋工具。若問題與課程評價"有關"，請使用web_search工具搜尋解答。
    """
    query: str = Field(description="使用網路搜尋時輸入的問題")

class vectorstore(BaseModel):
    """
    跟課程詳細資料有關的向量資料庫工具。若問題與課程詳細資料，例如 grading, description 等，請使用此工具搜尋解答。
    """
    query: str = Field(description="搜尋向量資料庫時輸入的問題")


# Prompt Template
instruction = """
你是將使用者問題導向向量資料庫或網路搜尋的專家。
向量資料庫包含有關成大選修課程的詳細相關資訊。對於這些主題的問題，請使用向量資料庫工具。其他情況則使用網路搜尋工具。
"""
route_prompt = ChatPromptTemplate.from_messages(
    [
        ("system",instruction),
        ("human", "{question}"),
    ]
)

# Route LLM with tools use
llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY,model="gpt-3.5-turbo", temperature=0)

structured_llm_router = llm.bind_tools(tools=[web_search, vectorstore])

# question router chain
question_router = route_prompt | structured_llm_router


### Retriever
embeddings_model = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
new_db = FAISS.load_local("faiss_index", embeddings_model,allow_dangerous_deserialization=True)

retriever = new_db.as_retriever()

### RAG responder
llm_model = ChatOpenAI(openai_api_key=OPENAI_API_KEY,temperature=0, model="gpt-3.5-turbo") 


### Contextualize question ###
prompt_search_query = ChatPromptTemplate.from_messages([
    MessagesPlaceholder(variable_name="chat_history"),
    ("user", "{input}"),
    ("user", "Given the above conversation, generate a search query to look up in order to get information relevant to the conversation")
])

retriever_chain = create_history_aware_retriever(llm_model, retriever, prompt_search_query)



### Answer question ###
prompt_get_answer = ChatPromptTemplate.from_messages([
    ('system', 'Answer the user\'s questions based on the below context:\n\n{context}'),
    MessagesPlaceholder(variable_name="chat_history"),
    ('user', '{input}'),
])

document_chain = create_stuff_documents_chain(llm_model, prompt_get_answer)

retrieval_chain_combine = create_retrieval_chain(retriever_chain, document_chain)


### Retrieval Grader
class GradeDocuments(BaseModel):
    """
    確認提取文章與問題是否有關(yes/no)
    """

    binary_score: str = Field(description="請問文章與問題是否相關。('yes' or 'no')")

# Prompt Template
instruction = """
你是一個評分的人員，負責評估文件與使用者問題的關聯性。
如果文件包含與使用者問題相關的關鍵字或語意，則將其評為相關。
輸出 'yes' or 'no' 代表文件與問題的相關與否。
"""
grade_prompt = ChatPromptTemplate.from_messages(
    [
        ("system",instruction),
        ("human", "文件: \n\n {document} \n\n 使用者問題: {question}"),
    ]
)

# Grader LLM
llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY ,model="gpt-3.5-turbo", temperature=0)
structured_llm_grader = llm.with_structured_output(GradeDocuments)

# 使用 LCEL 語法建立 chain
retrieval_grader = grade_prompt | structured_llm_grader


### Hallucination Grader
class GradeHallucinations(BaseModel):
    """
    確認答案是否為虛構(yes/no)
    """

    binary_score: str = Field(description="答案是否由為虛構。('yes' or 'no')")

# Prompt Template
instruction = """
你是一個評分的人員，負責確認LLM的回應是否為虛構的。
以下會給你一個文件與相對應的LLM回應，請輸出 'yes' or 'no'做為判斷結果。
'Yes' 代表LLM的回答是虛構的，未基於文件內容 'No' 則代表LLM的回答並未虛構，而是基於文件內容得出。
"""
hallucination_prompt = ChatPromptTemplate.from_messages(
    [
        ("system",instruction),
        ("human", "文件: \n\n {documents} \n\n LLM 回應: {generation}"),
    ]
)


# Grader LLM
llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY, model="gpt-3.5-turbo", temperature=0)
structured_llm_grader = llm.with_structured_output(GradeHallucinations)

# 使用 LCEL 語法建立 chain
hallucination_grader = hallucination_prompt | structured_llm_grader


### Answer Grader
class GradeAnswer(BaseModel):
    """
    確認答案是否可回應問題
    """

    binary_score: str = Field(description="答案是否回應問題。('yes' or 'no')")

# Prompt Template
instruction = """
你是一個評分的人員，負責確認答案是否回應了問題。
輸出 'yes' or 'no'。 'Yes' 代表答案確實回應了問題， 'No' 則代表答案並未回應問題。
"""
# Prompt
answer_prompt = ChatPromptTemplate.from_messages(
    [
        ("system",instruction),
        ("human", "使用者問題: \n\n {question} \n\n 答案: {generation}"),
    ]
)

# LLM with function call
llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY, model="gpt-3.5-turbo", temperature=0)
structured_llm_grader = llm.with_structured_output(GradeAnswer)

# 使用 LCEL 語法建立 chain
answer_grader = answer_prompt | structured_llm_grader



### Statefully manage chat history
store = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]


conversational_rag_chain = RunnableWithMessageHistory(
    retrieval_chain_combine,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="answer",
)


# LangGraph
# graphstate
class GraphState(TypedDict):
    """
    State of graph.

    Attributes:
        question: question
        generation: LLM generation
        documents: list of documents
    """
    question : str
    generation : str
    documents : List[str]
    session_id : int

#Node / Conditional Edges
def retrieve(state):
    """
    Retrieve documents related to the question.

    Args:
        state (dict):  The current state graph

    Returns:
        state (dict): New key added to state, documents, that contains list of related documents.
    """

    print("---RETRIEVE---")
    question = state["question"]
    session_id = state["session_id"]


    # Retrieval
    user_chat_history = get_session_history(session_id)
    documents = retriever.invoke(question)

    return {"documents":documents, "question":question}

def web_search(state):
    """
    Web search based on the re-phrased question.

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Updates documents key with appended web results
    """

    print("---WEB SEARCH---")
    question = state["question"]

    # Web search
    docs = web_search_tool.invoke({"query": question})
    web_results = [Document(page_content=d["content"]) for d in docs]

    documents = web_results

    return {"documents": documents, "question": question}

def retrieval_grade(state):
    """
    filter retrieved documents based on question.

    Args:
        state (dict):  The current state graph

    Returns:
        state (dict): New key added to state, documents, that contains list of related documents.
    """

    # Grade documents
    print("---CHECK DOCUMENT RELEVANCE TO QUESTION---")

    documents = state["documents"]
    question = state["question"]
    session_id = state["session_id"]


    # Score each doc
    filtered_docs = []
    for d in documents:
        score = retrieval_grader.invoke({"question": question, "document": d.page_content})
        grade = score.binary_score
        if grade == "yes":
            print("  -GRADE: DOCUMENT RELEVANT-")
            filtered_docs.append(d)
        else:
            print("  -GRADE: DOCUMENT NOT RELEVANT-")
            continue
    return {"documents": filtered_docs, "question": question,}

def rag_generate(state):
    """
    Generate answer using  vectorstore / web search

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, generation, that contains LLM generation
    """

    print("---GENERATE IN RAG MODE---")
    question = state["question"]
    documents = state["documents"]
    session_id = state["session_id"]


    # RAG generation
    generation = conversational_rag_chain.invoke(
        {"input": question},
        config={
            "configurable": {"session_id": session_id}
        },
    )
    return {"documents": documents, "question": question, "generation": generation}

def plain_answer(state):
    """
    Generate answer using the LLM without vectorstore.

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, generation, that contains LLM generation
    """

    print("---GENERATE PLAIN ANSWER---")
    question = state["question"]
    session_id = state["session_id"]
    generation = conversational_rag_chain.invoke(
        {"input": question},
        config={
            "configurable": {"session_id": session_id}
        },)
    return {"question": question, "generation": generation}


### Edges ###
def route_question(state):
    """
    Route question to web search or RAG.

    Args:
        state (dict): The current graph state

    Returns:
        str: Next node to call
    """

    print("---ROUTE QUESTION---")
    question = state["question"]
    source = question_router.invoke({"question": question})

    # Fallback to plain LLM or raise error if no decision
    if "tool_calls" not in source.additional_kwargs:
        print("  -ROUTE TO PLAIN LLM-")
        return "plain_answer"
    if len(source.additional_kwargs["tool_calls"]) == 0:
      raise "Router could not decide source"

    # Choose datasource
    datasource = source.additional_kwargs["tool_calls"][0]["function"]["name"]
    if datasource == 'web_search':
        print("  -ROUTE TO WEB SEARCH-")
        return "web_search"
    elif datasource == 'vectorstore':
        print("  -ROUTETO VECTORSTORE-")
        return "vectorstore"

def route_retrieval(state):
    """
    Determines whether to generate an answer, or use websearch.

    Args:
        state (dict): The current graph state

    Returns:
        str: Binary decision for next node to call
    """

    print("---ROUTE RETRIEVAL---")
    filtered_documents = state["documents"]

    if not filtered_documents:
        # All documents have been filtered check_relevance
        print("  -DECISION: ALL DOCUMENTS ARE NOT RELEVANT TO QUESTION, ROUTE TO WEB SEARCH-")
        return "web_search"
    else:
        # We have relevant documents, so generate answer
        print("  -DECISION: GENERATE WITH RAG LLM-")
        return "rag_generate"

def grade_rag_generation(state):
    """
    Determines whether the generation is grounded in the document and answers question.

    Args:
        state (dict): The current graph state

    Returns:
        str: Decision for next node to call
    """

    print("---CHECK HALLUCINATIONS---")
    question = state["question"]
    documents = state["documents"]
    generation = state["generation"]

    score = hallucination_grader.invoke({"documents": documents, "generation": generation})
    grade = score.binary_score

    # Check hallucination
    if grade == "no":
        print("  -DECISION: GENERATION IS GROUNDED IN DOCUMENTS-")
        # Check question-answering
        print("---GRADE GENERATION vs QUESTION---")
        score = answer_grader.invoke({"question": question,"generation": generation})
        grade = score.binary_score
        if grade == "yes":
            print("  -DECISION: GENERATION ADDRESSES QUESTION-")
            return "useful"
        else:
            print("  -DECISION: GENERATION DOES NOT ADDRESS QUESTION-")
            return "not useful"
    else:
        print("  -DECISION: GENERATION IS NOT GROUNDED IN DOCUMENTS, RE-TRY-")
        return "not supported"
    

workflow = StateGraph(GraphState)

# Define the nodes
workflow.add_node("web_search", web_search) # web search
workflow.add_node("retrieve", retrieve) # retrieve
workflow.add_node("retrieval_grade", retrieval_grade) # retrieval grade
workflow.add_node("rag_generate", rag_generate) # rag
workflow.add_node("plain_answer", plain_answer) # llm

# Build graph
workflow.set_conditional_entry_point(
    route_question,
    {
        "web_search": "web_search",
        "vectorstore": "retrieve",
        "plain_answer": "plain_answer",
    },
)
workflow.add_edge("retrieve", "retrieval_grade")
workflow.add_edge("web_search", "retrieval_grade")
workflow.add_conditional_edges(
    "retrieval_grade",
    route_retrieval,
    {
        "web_search": "web_search",
        "rag_generate": "rag_generate",
    },
)
workflow.add_conditional_edges(
    "rag_generate",
    grade_rag_generation,
    {
        "not supported": "rag_generate", # Hallucinations: re-generate
        "not useful": "web_search", # Fails to answer question: fall-back to web-search
        "useful": END,
    },
)
workflow.add_edge("plain_answer", END)

# Compile
app = workflow.compile()

def run(question, session_id):
    inputs = {"question": question,
              "session_id": session_id}
    for output in app.stream(inputs):
        print("\n")

    # Final generation
    if 'rag_generate' in output.keys():
        print(output['rag_generate']['generation']['answer'])
    elif 'plain_answer' in output.keys():
        print(output['plain_answer']['generation'])


run("初級會計學的評分標準?", "陳柏儒")
