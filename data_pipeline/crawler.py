import requests
from bs4 import BeautifulSoup
import pandas as pd
import os
from tqdm.notebook import tqdm
from google.colab import drive
import sys

song_info = []

date_list = [
    "20100101", "20100201", "20100301", "20100401", "20100501", "20100601",
    "20100701", "20100801", "20100901", "20101001", "20101101", "20101201",
    "20110101", "20110201", "20110301", "20110401", "20110501", "20110601",
    "20110701", "20110801", "20110901", "20111001", "20111101", "20111201",
    "20120101", "20120201", "20120301", "20120401", "20120501", "20120601",
    "20120701", "20120801", "20120901", "20121001", "20121101", "20121201",
    "20130101", "20130201", "20130301", "20130401", "20130501", "20130601",
    "20130701", "20130801", "20130901", "20131001", "20131101", "20131201",
    "20140101", "20140201", "20140301", "20140401", "20140501", "20140601",
    "20140701", "20140801", "20140901", "20141001", "20141101", "20141201",
    "20150101", "20150201", "20150301", "20150401", "20150501", "20150601",
    "20150701", "20150801", "20150901", "20151001", "20151101", "20151201",
    "20160101", "20160201", "20160301", "20160401", "20160501", "20160601",
    "20160701", "20160801", "20160901", "20161001", "20161101", "20161201",
    "20170101", "20170201", "20170301", "20170401", "20170501", "20170601",
    "20170701", "20170801", "20170901", "20171001", "20171101", "20171201",
    "20180101", "20180201", "20180301", "20180401", "20180501", "20180601",
    "20180701", "20180801", "20180901", "20181001", "20181101", "20181201",
    "20190101", "20190201", "20190301", "20190401", "20190501", "20190601",
    "20190701", "20190801", "20190901", "20191001", "20191101", "20191201",
    "20200101", "20200201", "20200301", "20200401", "20200501", "20200601",
    "20200701", "20200801", "20200901", "20201001", "20201101", "20201201",
    "20210101", "20210201", "20210301", "20210401", "20210501", "20210601",
    "20210701", "20210801", "20210901", "20211001", "20211101", "20211201",
    "20220101", "20220201", "20220301", "20220401", "20220501", "20220601",
    "20220701", "20220801", "20220901", "20221001", "20221101", "20221201",
    "20230101", "20230201", "20230301", "20230401", "20230501", "20230601",
    "20230701", "20230801", "20230901", "20231001"
]

count = 0

for day in tqdm(date_list):

    all_webpage = requests.get("https://music.bugs.co.kr/chart/track/week/total?chartdate=" + day)
    all_soup = BeautifulSoup(all_webpage.content, "html.parser")

    track_info_links = all_soup.find_all('a', class_='trackInfo', href=True)
    href_values = [link['href'] for link in track_info_links]

    for i in href_values:
        webpage = requests.get(i)
        soup = BeautifulSoup(webpage.content, "html.parser")

        # "lyricsContainer" 클래스를 가진 div 요소 찾기
        lyrics_div = soup.find('div', class_='lyricsContainer')

        # lyrics_div 안의 <p><xmp> 요소 찾기
        if lyrics_div:
            xmp_element = lyrics_div.find('p').find('xmp')

            if xmp_element:
                xmp_content = xmp_element.get_text()

                # xmp_content를 한 문장씩 리스트에 담기
                sentences = [sentence.strip() for sentence in xmp_content.split('\n') if sentence.strip()]

                # og:title 속성을 가진 <meta> 태그 찾기
                og_title_tag = soup.find('meta', {'property': 'og:title'})

                if og_title_tag:
                    og_title_content = og_title_tag.get('content')
                    og_title_list = og_title_content.split('/')

                    song_info.append(og_title_list + sentences)
            else:
              print("xml 요소를 찾을 수 없습니다.")


    count = count + 1

    print(day + " 크롤링을 완료 하였습니다.")

# ----------엑셀 파일로 저장 (백업 목적)-----------
    if (count%12 == 0): # 1년 단위로 업데이트

      df = pd.DataFrame(song_info)
      df = df.drop_duplicates([0])

      df.to_excel('song_info_' + day + '.xlsx')
