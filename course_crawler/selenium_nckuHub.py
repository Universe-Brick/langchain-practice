# import time
import pandas as pd
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.action_chains import ActionChains


service = Service('/Users/lucifiel/chromedriver-mac-x64/chromedriver')
driver = webdriver.Chrome(service=service)
driver.get("https://nckuhub.com/")

input_box = WebDriverWait(driver, 30).until(
    EC.presence_of_element_located((By.CLASS_NAME, "quick_search_input"))
)

input_box.send_keys("A9") 
# print("Search input sent")


element2 = WebDriverWait(driver, 10).until(
    EC.element_to_be_clickable((By.CSS_SELECTOR, '.search_result--course p'))
)
# print("Search result element located")
driver.execute_script("arguments[0].click();", element2)
# print("Search result clicked")


element1 = WebDriverWait(driver, 10).until(
    EC.presence_of_element_located((By.XPATH, "//span[text()='僅顯示有心得之課程']"))
)
driver.execute_script("arguments[0].click();", element1)
# print("Checkbox '僅顯示有心得之課程' clicked")


WebDriverWait(driver, 10).until(
    EC.presence_of_all_elements_located((By.CLASS_NAME, 'list_course_item_title'))
)                                    
# print("Course list loaded")

titles = driver.find_elements(By.CLASS_NAME, 'list_course_item_title')
print(f"Number of course titles found: {len(titles)}")

df = pd.DataFrame(columns=['課程名稱', '收穫', '甜度', '涼度'])

for i, title in enumerate(titles):
    driver.execute_script("arguments[0].click();", title)

    try:
        score_btn_div = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, '.flex-container.score-btn--all'))
        )
        if score_btn_div:
            try:
                print(title.text)
                row_data = {'課程名稱': title.text}
                p_div = WebDriverWait(driver, 10).until(
                    EC.presence_of_element_located((By.CSS_SELECTOR, '.flex-container.score-btn--all'))
                )
                c_divs = p_div.find_elements(By.CLASS_NAME, 'score-btn')
                for div in c_divs:
                    span1_text = div.find_element(By.XPATH, './span[1]').text
                    span2_text = div.find_element(By.CLASS_NAME, 'score_span').text
                    print(f"{span1_text} {span2_text}")  
                    if span1_text in ['收穫', '甜度', '涼度']:
                        row_data[span1_text] = span2_text
                df = pd.concat([df, pd.DataFrame([row_data])], ignore_index=True)                           
        
            except Exception as e:
                print(f"An error occurred: {e}")
                continue
    except Exception as e:
        continue
    actions = ActionChains(driver)
    actions.move_by_offset(0, 0).click().perform()
    
driver.close()
df.to_csv('courseScore.csv', index=False, encoding='utf-8')
