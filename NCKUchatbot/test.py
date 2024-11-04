import requests

url = "http://127.0.0.1:8000/ask"
data = {
    "question": "",
    "session_id": "陳柏儒"
}

# result = requests.post(url, json=data)
# print(result.text) 



# data['question'] = '列出三個分數很甜的通識課'
# result = requests.post(url, json=data)
# print(result.text) 

def interactive_chat():
    session_id = "陳柏儒"  # 固定 session_id
    
    while True:
        question = input("請輸入問題 (輸入 'exit' 或 'quit' 結束): ")
        
        if question.lower() in ["exit", "quit"]:
            print("結束對話")
            break

        # 呼叫 run 函數
        data['question'] = question
        result = requests.post(url, json=data)
        print(result.text) 

# 開始互動模式
interactive_chat()
