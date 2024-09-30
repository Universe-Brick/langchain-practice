from langchain_community.document_loaders import CSVLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings 
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI 
from langchain.chains import ConversationalRetrievalChain


# file_paths = [
#     "/Users/boruchen/Documents/langchain-practice/ncku_CoM_data.csv",
#     "/Users/boruchen/Documents/langchain-practice/ncku_hub_data.csv"
# ]

# all_documents = []

# for file_path in file_paths:
#     loader = CSVLoader(file_path=file_path)
#     documents = loader.load()
#     all_documents.extend(documents)  

# text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)

# docs = text_splitter.split_documents(all_documents)

embeddings_model = OpenAIEmbeddings(openai_api_key='your api-key')

# vectorstore = FAISS.from_documents(all_documents, embeddings_model)

# vectorstore.save_local("faiss_index")



new_db = FAISS.load_local("faiss_index", embeddings_model, allow_dangerous_deserialization=True)


llm_model = ChatOpenAI(openai_api_key='your api-key') 

# chain = ConversationalRetrievalChain.from_llm(llm=llm_model, retriever=new_db.as_retriever(search_kwargs={"k": 3, "fetch_k": 10}, search_type="mmr"))
chain = ConversationalRetrievalChain.from_llm(llm=llm_model, retriever=new_db.as_retriever())

# query = "請根據資料,列出甜度為6的課程" 
query = "請列出會計系 : 初級會計學（一）的 grading"
response = chain.invoke({"question": query, "chat_history": []}) 
print(response) 
print()

query2 = "請列出會計系 : 初級會計學（一）的課程敘述"
response2 = chain.invoke({"question": query2, "chat_history": []}) 
print(response2) 
print()

query3 = "會計系有什麼課程"
response3 = chain.invoke({"question": query3, "chat_history": []}) 
print(response3)
print()
