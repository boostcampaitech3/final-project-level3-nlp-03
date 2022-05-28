from selenium import webdriver
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By
import time

def eng_to_kor_func(word):
    webdriver_options = webdriver.ChromeOptions()
    webdriver_options.headless = True
    webdriver_options.add_argument("--no-sandbox")
    webdriver_options.add_argument("--disable-dev-shm-usage")

    # ChromeDriverManager().install()
    driver = webdriver.Chrome(
        "/opt/ml/.wdm/drivers/chromedriver/linux64/102.0.5005.61/chromedriver",
        options=webdriver_options,
    )

    driver.implicitly_wait(3)
    driver.get(f"https://en.dict.naver.com/#/search?range=meaning&query={word}")

    # print(driver.title)
    kor_word = driver.find_element(
        by=By.CSS_SELECTOR,
        value="#searchPage_mean > div.component_keyword.has-saving-function > div:nth-child(1) > div",
    ).text

    return kor_word

