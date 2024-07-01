from langchain_community.document_loaders import CSVLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings 
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI 
from langchain.chains import ConversationalRetrievalChain


file_paths = [
    "/Users/lucifiel/Downloads/courseScore.csv",
    "/Users/lucifiel/Downloads/courses_data.csv",
    "/Users/lucifiel/Downloads/syllabus_data.csv"
]

all_documents = []

for file_path in file_paths:
    loader = CSVLoader(file_path=file_path)
    documents = loader.load()
    all_documents.extend(documents)  

text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)

docs = text_splitter.split_documents(all_documents)

embeddings_model = OpenAIEmbeddings(openai_api_key='your_api_key')

vectorstore = FAISS.from_documents(all_documents, embeddings_model)

vectorstore.save_local("faiss_index")



new_db = FAISS.load_local("faiss_index", embeddings_model, allow_dangerous_deserialization=True)


llm_model = ChatOpenAI(openai_api_key='your_api_key') 

#chain = ConversationalRetrievalChain.from_llm(llm=llm_model, retriever=new_db.as_retriever(search_kwargs={"k": 3, "fetch_k": 10}, search_type="mmr"))
chain = ConversationalRetrievalChain.from_llm(llm=llm_model, retriever=new_db.as_retriever())

query = "請根據資料,列出甜度為6的課程" 
response = chain.invoke({"question": query, "chat_history": []}) 
print(response) 



