import requests
import json
import csv
import urllib.request
import pandas as pd

from tqdm import tqdm
from bs4 import BeautifulSoup
from hanspell import spell_checker
from konlpy.tag import Okt
from hangul_utils import join_jamos, split_syllables

from eng_to_kor import eng_to_kor_func


def lang_check(word):
    lang_client_id = "id"
    lang_client_secret = "secret"
    # id, password 대신 파파고 언어감지 api를 발급받으면 제공되는 client id와 secret key를 대신 입력해주세요.
    # https://developers.naver.com/docs/papago/papago-detectlangs-overview.md 에서 자세한 내용을 확인할 수 있습니다.

    data = f"query={word}"
    url = "https://openapi.naver.com/v1/papago/detectLangs"
    request = urllib.request.Request(url)
    request.add_header("X-Naver-Client-Id", lang_client_id)
    request.add_header("X-Naver-Client-Secret", lang_client_secret)
    response = urllib.request.urlopen(request, data=data.encode("utf-8"))
    rescode = response.getcode()
    if rescode == 200:
        response_body = response.read()
        source = json.loads(response_body.decode("utf-8"))["langCode"]
        return source
    else:
        print("Error Code:" + rescode)


def sentence_to_word(word):
    word = word.replace(".", "").replace("It's ", "")
    word = word.replace("be ", "").replace("I ", "").replace("get ", "")
    return word


def translate(word):
    papago_translate_client_id = "id"
    papago_translate_client_secret = "password"
    # id, password 대신 파파고 NMT api를 발급받으면 제공되는 client id와 secret key를 대신 입력해주세요.
    # https://developers.naver.com/docs/papago/papago-nmt-overview.md 에서 자세한 내용을 확인할 수 있습니다.

    source = lang_check(word)
    if source == "ko":
        target = "en"
    else:
        source = "en"
        target = "ko"
    url = f"https://openapi.naver.com/v1/papago/n2mt"
    data = f"source={source}&target={target}&text={word}"

    request = urllib.request.Request(url)
    request.add_header("X-Naver-Client-Id", papago_translate_client_id)
    request.add_header("X-Naver-Client-Secret", papago_translate_client_secret)
    response = urllib.request.urlopen(request, data=data.encode("utf-8"))
    rescode = response.getcode()
    if rescode == 200:
        response_body = response.read()
        result = json.loads(response_body.decode("utf-8"))["message"]["result"]
        translated_word = result["translatedText"]
        if source == "ko":
            translated_word = sentence_to_word(translated_word)
        return translated_word
    else:
        print("Error Code:" + rescode)


def find_antonyms(word):
    url = f"https://www.thesaurus.com/browse/{word}"

    response = requests.get(url)
    if response.status_code == 200:
        html = response.text
        soup = BeautifulSoup(html, "html.parser")
        n_of_word = 2
        words = []
        for i in range(1, n_of_word + 1):
            word = soup.select(
                f"#antonyms > div.css-ixatld.e15rdun50 > ul > li:nth-child({i}) > a"
            )
            if str(word) == "[]":
                pass
            else:
                word = str(word).split("<!-- -->")
                word = word[0].split('">')[1]
            words.append(word)
        return words
    else:
        words = []
        return words


def spell_correct(word):
    okt = Okt()
    text = word
    if "우다" in word:
        text = word[:-2] + "ㅂ다"
        text = split_syllables(text)
        text = join_jamos(text)
    hanspell_result = spell_checker.check(text)
    text = hanspell_result.checked
    return text.replace(" ", "")


def make_dataframe(csv_type, is_da):
    df_name = f"WAI-{csv_type}-origin"
    csv_name = f"WAI-para_{csv_type}.csv"

    word_df = pd.DataFrame(
        columns=("original", "antonyms")
    )
    word_list = pd.read_csv(csv_name)
    for word in tqdm(word_list[csv_type]):
        word = spell_correct(word)
        origin = word
        if not is_da:
            origin = word[:-1]
        translated_original = translate(origin)
        translated_antonyms = find_antonyms(translated_original)
        antonyms = []

        for word in translated_antonyms:
            if word:
                antonyms.append(translate(word))
                # antonyms.append(eng_to_kor_func(word))
            else:
                antonyms.append("")
        new_data = {
            "original": origin,
            "antonyms": antonyms,
        }
        new_df = pd.DataFrame(new_data)
        word_df = pd.concat([word_df, new_df])
    word_df.to_csv(f"./{df_name}.csv")


def main():
    #데이터셋이 동일한 폴더 안에 있어야 하는 코드로, 경로가 바뀐다면 조정이 필요합니다.
    origin = make_dataframe("VA", 1)


if __name__ == "__main__":
    main()
