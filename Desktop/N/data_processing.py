import csv

def transform_csv(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as infile, open(output_file, 'w', encoding='utf-8', newline='') as outfile:
        reader = csv.reader(infile)
        writer = csv.writer(outfile)

        for row in reader:
            # 跳過空行
            if not row or len(row) < 7:
                print(f"跳過無效行（列數不足）: {row}")
                continue

            try:
                # 若第一列為空，嘗試補充
                if not row[0].strip():
                    row[0] = "未知系所"
                
                # 解析主要欄位
                department, room_info = row[0].strip(), row[1].strip()
                course_name = row[3].strip() or "未知課程"
                credits = row[4].strip().split()[0] if row[4].strip() else "未知"
                time_info = row[-1].strip()
                
                # 確保教室名稱存在
                room = room_info.split()[0] if room_info else "未知教室"
                department_course = f"{department} {course_name}"

                # 組裝目標行
                result_row = [room, department_course, credits, time_info]
                
                # 寫入目標檔案
                writer.writerow(result_row)

            except Exception as e:
                print(f"跳過無效行（解析失敗）: {row}. 錯誤: {e}")

# 使用範例
input_file = r"langchain-practice\course_info.csv"  # 替換為你的輸入檔案路徑
output_file = r"langchain-practice\output.csv"  # 輸出的檔案路徑
transform_csv(input_file, output_file)
