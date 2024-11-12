import requests
#用main:app --reload

state = {
    "course_info": {
        "id":"H3456-1",
        "title": "工資系計算機概論",
        "start": "2024-10-24T10:00",
        "end": "2024-10-24T11:00",
        "day": "Friday",
        "eventGroupId": "H3456",
        "credit":3,
    }
}

def add_to_schedule(state):
    print("---ADD TO SCHEDULE---")

    course_info = state.get("course_info", {})
    api_url = "http://127.0.0.1:8000/addNewEvents"  

    data = {
        "id":course_info.get("id"),
        "title":course_info.get("title"),
        "start":course_info.get("start"),
        "end": course_info.get("end"),
        "day": course_info.get("day"),
        "eventGroupId": course_info.get("eventGroupId"),
        "credit":course_info.get("credit"),
    }

    # 發送 API 請求
    response = requests.post(api_url, json=data)
    if response.status_code == 200:
        print("課程已成功加入課表")
        status = "課程已成功加入課表"
    else:
        print("課表更新失敗")
        status = "課表更新失敗"
    
    return {"status": status, "course_info": course_info}

# 執行 add_to_schedule 函數並顯示結果
result = add_to_schedule(state)
print(result)

