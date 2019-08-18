import requests
import json
import math
import csv
from itertools import repeat
from multiprocessing import Process, Pool

def test(k, n, filename):
    base_url = 'https://www.bigkinds.or.kr/api/news/search.do'

    params = {
        "indexName":"news",
        "searchKey": k, # 검색어
        "searchKeys":[
            {
            }
        ],
        "byLine":"",
        "searchFilterType":"1",
        "searchScopeType":"1",
        "searchSortType":"date",
        "sortMethod":"date",
        "mainTodayPersonYn":"",
        "startDate":"2009-01-02", # 시작일
        "endDate":"2019-08-17", # 종료일
        "newsIds":[
        ],
        "categoryCodes":[
        ],
        "providerCodes":[
        ],
        "incidentCodes":[
        ],
        "networkNodeType":"",
        "topicOrigin":"",
        "startNo":n, # 시작 페이지
        "resultNumber":100, # 결과출력개수
        "dateCodes":[
        ],
        "isTmUsable":'false',
        "isNotTmUsable":'false'
    }

    headers = {'Content-Type': 'application/json; charset=utf-8'}

    response = requests.post(url=base_url, data=json.dumps(params), headers=headers)

    test = response.json()

    # print(test["totalCount"]) # 총 데이터 갯수

    # for i in range(0, test["documentCount"]):
        # print(test["resultList"][i]["NEWS_ID"][9:21],test["resultList"][i]["TITLE"])
        # print("{0}, {1},".format(test["resultList"][i]["NEWS_ID"][9:21], test["resultList"][i]["TITLE"]))

    f = open(filename + ".csv", 'a', encoding='cp949', newline='')
    wr = csv.writer(f)
    for i in range(0, test["documentCount"]):
        wr.writerow([test["resultList"][i]["NEWS_ID"][9:21], test["resultList"][i]["TITLE"]])
    f.close()

    print(n, "번째 페이지 크롤링 완료")

def datacount(k):
    base_url = 'https://www.bigkinds.or.kr/api/news/search.do'

    params = {
        "indexName":"news",
        "searchKey": k, # 검색어
        "searchKeys":[
            {
            }
        ],
        "byLine":"",
        "searchFilterType":"1",
        "searchScopeType":"1",
        "searchSortType":"date",
        "sortMethod":"date",
        "mainTodayPersonYn":"",
        "startDate":"2009-01-02", # 시작일
        "endDate":"2019-08-17", # 종료일
        "newsIds":[
        ],
        "categoryCodes":[
        ],
        "providerCodes":[
        ],
        "incidentCodes":[
        ],
        "networkNodeType":"",
        "topicOrigin":"",
        "startNo":1, # 시작 페이지
        "resultNumber":100, # 결과출력개수
        "dateCodes":[
        ],
        "isTmUsable":'false',
        "isNotTmUsable":'false'
    }

    headers = {'Content-Type': 'application/json; charset=utf-8'}

    response = requests.post(url=base_url, data=json.dumps(params), headers=headers)

    test = response.json()

    return math.ceil(test["totalCount"]/100) # 총 데이터 갯수

if __name__== '__main__':
    keyword = input("검색 키워드 입력  : ")

    filename = input('저장할 파일 이름 : ')

    counts = int(datacount(keyword))

    print("총 페이지 수 : ", counts)

    f = open(filename + ".csv", 'w')
    f.write("date, title,\n")
    f.close()

    with Pool(processes=5) as pool:
        results = pool.starmap(test, zip(repeat(keyword), range(1, counts+1), repeat(filename)))