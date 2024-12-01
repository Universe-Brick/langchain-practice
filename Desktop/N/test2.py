import csv

def test_course_string_split(input_file="output-1.csv"):
    # 從 output-1.csv 讀取並測試字串分割
    with open(input_file, mode="r", encoding="utf-8") as file:
        reader = csv.reader(file)
        for row in reader:
            input_string = row[0]  # 假設每行的第一個欄位是課程字串
            try:
                # 分割字串，這裡只測試分割部分
                course_code, title, credit, time_string = input_string.split(",", 3)
                print(f"課程代碼: {course_code}")
                print(f"課程名稱: {title}")
                print(f"學分數: {credit}")
                print(f"時間字串: {time_string}")
                print("-" * 40)
            except ValueError:
                print(f"錯誤: 無法分割這一行資料: {input_string}")

# 執行測試
test_course_string_split()
