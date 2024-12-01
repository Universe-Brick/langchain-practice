import csv
from datetime import datetime, timedelta
import re

def generate_course_info_from_list(course_data, base_date="2024-10-20"):
    """
    根據輸入的課程資料生成 course_info。
    """
    # 星期對應表
    day_mapping = {
        "Sunday": 0,
        "Monday": 1,
        "Tuesday": 2,
        "Wednesday": 3,
        "Thursday": 4,
        "Friday": 5,
        "Saturday": 6,
    }

    # 時間對應表
    time_mapping = {
        "1": ["08:00", "09:00"],
        "2": ["09:00", "10:00"],
        "3": ["10:00", "11:00"],
        "4": ["11:00", "12:00"],
        "5": ["12:00", "13:00"],
        "6": ["13:00", "14:00"],
        "7": ["14:00", "15:00"],
        "8": ["15:00", "16:00"],
        "9": ["16:00", "17:00"],
        "A": ["17:00", "18:00"],
        "N": ["00:00", "00:00"],  # 'N'代表無時間
        "定": ["00:00", "00:00"],  # '定'代表無時間
    }

    # 設定基準日期
    base_date = datetime.strptime(base_date, "%Y-%m-%d")

    # 從 course_data 中提取課程資訊
    course_code, title, credit, time_string = course_data
    if credit == "未知":
        credit = -1
    else:
        credit = int(credit)  # 確保學分是整數
    sessions = time_string.split()  # 分割每個時段

    course_info = []

    for i, session in enumerate(sessions):
        # 處理每個時段的 day_code 和 time_code
        try:
            day_code = session[1]  # [2] -> "2" (星期幾)
            time_code = session[3:].strip()  # [2]2 -> "2" (時間代碼)，並去除多餘的空格

            if not time_code:  # 如果 time_code 是空的，跳過該時段
                continue

            # 處理時間範圍（如 '7~8'）
            if '~' in time_code:
                start_slot, end_slot = time_code.split('~')
                start_time, _ = time_mapping.get(start_slot, ["00:00", "00:00"])  # 如果無對應時間，使用預設
                _, end_time = time_mapping.get(end_slot, ["00:00", "00:00"])
            else:
                # 如果不是範圍，則使用單一時間
                start_time, end_time = time_mapping.get(time_code, ["00:00", "00:00"])

            # 計算星期幾
            day_offset = int(day_code) - 1  # 星期從 1 開始，所以減 1
            course_date = base_date + timedelta(days=day_offset)
            weekday = list(day_mapping.keys())[day_offset]  # 將數字轉換為星期名稱

            # 添加課程資訊
            course_info.append({
                "id": f"{course_code}-{i + 1}",
                "title": title,
                "start": f"{course_date.strftime('%Y-%m-%d')}T{start_time}",
                "end": f"{course_date.strftime('%Y-%m-%d')}T{end_time}",
                "day": weekday,
                "eventGroupId": course_code,
                "credit": credit,  # 填入讀入的學分數
            })
        except IndexError:
            print(f"錯誤: 資料格式錯誤，無法處理時段 {session}，跳過該時段")

    return course_info


# 從 output-1.csv 讀取資料
data = []
with open("langchain-practice\\output-1.csv", mode="r", encoding="utf-8") as file:
    reader = csv.reader(file)
    for row in reader:
        if len(row) >= 4:  # 確保每行有四個欄位（課程代碼，課程名稱，學分數，課程時間）
            data.append(row)  # 假設每行已經是適當的格式

print(f"讀取了 {len(data)} 條課程資料")
print(data[1])
print(data[10])

# 打開 CSV 檔案以寫入資料到 output-2.csv
with open("output-2.csv", mode="w", newline="", encoding="utf-8") as file:
    writer = csv.DictWriter(file, fieldnames=["id", "title", "start", "end", "day", "eventGroupId", "credit"])
    writer.writeheader()

    # 逐行處理並寫入 CSV
    for course_data in data[1:]:
        course_info_list = generate_course_info_from_list(course_data)

        for course in course_info_list:
            writer.writerow(course)

print("CSV 檔案已經成功生成：output-2.csv")
