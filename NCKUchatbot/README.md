# Project Name

NCKUChatbot

## 目錄

<!-- - [專案簡介](#專案簡介)
- [功能特點](#功能特點)
- [安裝指南](#安裝指南)
- [使用說明](#使用說明)
- [技術棧](#技術棧)
- [貢獻指南](#貢獻指南)
- [授權](#授權) -->

## 專案簡介


> 這個專案是一個基於 FastAPI 和 Langchain 的問答系統，，實現自動化處理和回應使用者提問。
> 你可以根據你的需求向聊天機器人詢問管理學院以及通識課相關的課程資訊


## 安裝指南


```bash
# To set up the environments
pipenv install //make sure Pipfile.lock is in the file.
```

## 檔案說明

- main.py 中包含了 fastapi 的 function 可以呼叫
- test.py 用於測試 main.py 的輸入輸出的
- 你必須新增一個 .env 檔，把 api key 放進去，並且將此檔案加入 .gitignore 避免 push 到 api key
