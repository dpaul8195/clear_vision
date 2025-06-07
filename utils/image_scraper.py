from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
import time
import os
import urllib.request

def image_scraper(dir):
    service = Service()
    options = webdriver.ChromeOptions()
    driver = webdriver.Chrome(service=service, options=options)
    driver.maximize_window()
    driver.get("https://images.google.com/")

    name = "real pics"
    diag_box = driver.find_element(By.XPATH, "//textarea[@id='APjFqb']")

    time.sleep(2)
    diag_box.send_keys(name)
    diag_box.send_keys(Keys.ENTER)
    time.sleep(2)
    scroll_pause_time = 2
    last_height = driver.execute_script("return document.body.scrollHeight")

    while True:
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(scroll_pause_time)
        new_height = driver.execute_script("return document.body.scrollHeight")
        if new_height == last_height:
            break
        last_height = new_height

    all_img = driver.find_elements(By.TAG_NAME, "img")
    seen_urls = set()


    for i, img in enumerate(all_img):
        try:
            class_attr = img.get_attribute("class")
            if class_attr and "XNo5Ab lWlVCe" in class_attr:
                continue
            if class_attr and "YQ4gaf zr758c" in class_attr:
                continue

            src = img.get_attribute("src")
            if src and src.startswith("http") and src not in seen_urls:
                seen_urls.add(src)  # mark as seen
                file_path = os.path.join(dir, f"{name} ({i}).png")
                urllib.request.urlretrieve(src, file_path)
        except:
            continue


    driver.quit()

    return dir