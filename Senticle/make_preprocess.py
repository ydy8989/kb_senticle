import pandas as pd
import Senticle.cnn_tool as tool
import os
import csv
from soynlp.noun import LRNounExtractor_v2
from soynlp.tokenizer import NounLMatchTokenizer

company_name = input("RawData File Name? : ")
data_path = company_name + '_labeled_data.csv'  # csv 파일로 불러오기

# =============================================================================
# contents는 각 기사 스트링으로 바꿔 리스트에 넣은거, points는 클래스 0or 1
# =============================================================================
contents, points = tool.loading_rdata(data_path)

if os.path.isfile('preprocessed_' + company_name + '.csv') == False:
    print("\n")
    print('"preprocessed_' + company_name + '.csv" deos not EXIST!')
    print('MAKE "preprocessed_' + company_name + '.csv" FILE... 가즈아~!!')
    print("\n")
    # doc = pd.read_csv(data_path, index_col='datetime')
    #
    #
    #
    # contents = []
    #
    # #todo : 길이 100 이하인 기사는 drop 코드 수정 요망. 판다스 수준에서. row drop 시키면됨.
    # for i in range(len(doc['text'])):
    #     if len(doc.iloc[i]['text']) > 100:
    #         contents.append(doc.iloc[i]['text']) #컨텐츠 리스트로 접근하면 안되고, 데이터프레임을 드랍시켜야함

    noun_extractor = LRNounExtractor_v2(verbose=True)
    nouns = noun_extractor.train_extract(contents, min_noun_frequency=20)

    match_tokenizer = NounLMatchTokenizer(nouns)

    # f = open('preprocessed_' + company_name + '.csv', 'w', newline='', encoding='utf-8')
    # fieldnames = ['text', 'num']
    # writer = csv.DictWriter(f, fieldnames=fieldnames)
    # writer.writeheader()
    noun_contents = []
    for j in range(len(contents)):
        temp_list = match_tokenizer.tokenize(contents[j])
        del_list2 = []
        for i in range(len(temp_list)):
            if len(temp_list[i]) == 1:  # 자른 워드 크기 1이면 삭제
                del_list2.append(i)
            else:
                pass
        del_list2.sort(reverse=True)
        for i in del_list2:
            try:
                del temp_list[i]
            except ValueError:
                pass
        temp_list = ' '.join(temp_list)
        noun_contents.append(temp_list)
        # writer.writerow({'text': temp_list, 'num': points[j]})
        if j % 10 == 0:
            print("{}개의 기사 중 {}번 기사 불용어처리후 저장완료~ ^오^".format(len(contents), j + 1))

    len(noun_contents)
    len(points)

    # f.close()
##################################################################################
# df = pd.read_csv('preprocessed_' + company_name + '.csv')
# df = df.dropna()
# contents = df.text
# points = df.num
#
# contents = contents.values.tolist()
# points = points.values.tolist()
# ################################


try:
    count = 0
    for i in range(len(noun_contents)):
        if type(noun_contents[i - count]) == float:
            del noun_contents[i]
            count += 1
except AttributeError:
    pass


print("클래스 갯수 : ", len(points))
print('기사 갯수 : ', len(noun_contents))
print("사전 생성 완료 >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")

# =============================================================================
# Tokenizing 후의 길이를 기준으로, 너무 길거나 짧은 기사 삭제
# =============================================================================
interval = int(input('얼마씩의 간격으로 기사 길이를 확인할지? (ex. 50) :'))
for county in range(0, 2000, 50):
    countx = 0
    for i in range(len(noun_contents)):
        if county < len(noun_contents[i]) < (county + interval):
            countx += 1
    print("기사 길이가 %d에서 %d 사이인 기사의 갯수 : " % (county, county + interval), countx)
print("기사 자르기 전 개수 :  ", len(noun_contents))
minlen = int(input("앞에서부터 몇 이하의 기사 길이까지 버리겠습니까?(숫자 입력) : "))
maxlen = int(input("몇 이상부터 끝까지의 기사 길이까지 버리겠습니까?(숫자 입력) : "))

# 자를 인덱스 리스트만 받아오기
del_list = []
for i in range(len(noun_contents)):
    if minlen < len(noun_contents[i]) < maxlen:
        pass
    else:
        del_list.append(i)
# del 함수를 사용하여, 뒤에서부터 지우기 위한 reverse sorting
del_list.sort(reverse=True)
for i in del_list:
    try:
        del noun_contents[i]
        del points[i]
    except ValueError:
        pass
print("기사 자른 후 남은 contents의 갯수 :  ", len(noun_contents))
print('====================================')
for county in range(0, 2000, interval):
    countx = 0
    for i in range(len(noun_contents)):
        if county < len(noun_contents[i]) < (county + interval):
            countx += 1
    print("기사 길이가 %d에서 %d 사이인 기사의 갯수 : " % (county, county + interval), countx)

#### 0 갯수랑 1 갯수랑 맞춰주기!!#####
print("-" * 30)
print('현재 남은 하락 기사의 갯수 :', len(points) - sum(points))
print("현재 남은 상승 기사의 갯수 :", sum(points))
print("-" * 30)

# =============================================================================
# while문 :
#   만약 y 선택할 시에, 차이가 나는 갯수만큼 앞에서 부터 자름.
# =============================================================================

while True:
    changeval = input("남은 상승/하락 기사의 갯수를 통일시겠습니까? (y/n) :")
    changeval = changeval.lower()

    if changeval == 'y':
        diff = abs(len(points) - sum(points) - sum(points))
        up_idx = []
        down_idx = []
        for i in range(len(noun_contents)):
            if points[i] == 0:
                down_idx.append(i)
            else:
                up_idx.append(i)
        up_idx.sort(reverse=True)
        down_idx.sort(reverse=True)
        up_idx = up_idx[:diff]
        down_idx = down_idx[:diff]

        if len(points) - sum(points) > sum(points):  # 하락 기사가 더 많을 때
            for i in down_idx:
                del noun_contents[i]
                del points[i]
        elif len(points) - sum(points) < sum(points): # 상승 기사가 더 많을 때
            for i in up_idx:
                del noun_contents[i]
                del points[i]
        else:
            pass

        print('현재 남은 하락 기사의 갯수 :', len(points) - sum(points))
        print("현재 남은 상승 기사의 갯수 :", sum(points))
        if len(points) - sum(points) == sum(points):
            print("갯수 맞춤 성공!!")
            break
    elif changeval == 'n':
        break
    else:
        pass


dfdf = pd.DataFrame(noun_contents,  columns=['text'])
dfdf['label'] = points
dfdf.to_csv('preprocessed_' + company_name + '.csv', index=True, header=True)
