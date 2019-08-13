import pandas as pd
import cnn_tool as tool
import os
import csv
from soynlp.noun import LRNounExtractor_v2
from soynlp.tokenizer import NounLMatchTokenizer

company_name = 'gold'#input("기업명 입력-영어(ex. SKhynix or posco)")
data_path =  company_name + '_labeled_data.csv'  # csv 파일로 불러오기

# contents는 각 기사 스트링으로 바꿔 리스트에 넣은거, points는 클래스 0or 1
contents, points = tool.loading_rdata(data_path)

#todo 토크나이징 함수 만드세요..
# 사전 파일 만들기
if os.path.isfile('preprocessed_' + company_name + '.csv') == False:
    print("\n")
    print('"preprocessed_' + company_name + '.csv" deos not EXIST!')
    print('MAKE "preprocessed_' + company_name + '.csv" FILE... 가즈아~!!')
    print("\n")
    doc = pd.read_csv(data_path, index_col='datetime')
    contents = []

    #todo : 길이 100 이하인 기사는 drop 코드 수정 요망. 판다스 수준에서. row drop 시키면됨.
    for i in range(len(doc['text'])):
        if len(doc.iloc[i]['text']) > 100:
            contents.append(doc.iloc[i]['text'])
    noun_extractor = LRNounExtractor_v2(verbose=True)
    nouns = noun_extractor.train_extract(contents, min_noun_frequency=20)

    match_tokenizer = NounLMatchTokenizer(nouns)

    f = open('preprocessed_' + company_name + '.csv', 'w', newline='', encoding='utf-8')
    fieldnames = ['text', 'num']
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    test = []
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
        test.append(temp_list)
        writer.writerow({'text': temp_list, 'num': points[j]})
        if j % 10 == 0:
            print("{}개의 기사 중 {}번 기사 불용어처리후 저장완료~ ^오^".format(len(contents), j + 1))
    f.close()

##################################################################################
#todo : 이 아래 두 블럭은 중복이니깐. 함수 다시 만들기 아래 부분으로만, 위에도 함수 따로 만들기.
df = pd.read_csv('preprocessed_' + company_name + '.csv')
df = df.dropna()
contents = df.text
points = df.num

contents = contents.values.tolist()
points = points.values.tolist()

try:
    contents = contents.values.tolist()
    points = points.values.tolist()
    count = 0
    for i in range(len(contents)):
        if type(contents[i - count]) == float:
            del contents[i]
            count += 1
except AttributeError:
    pass
print("클래스 갯수 : ", len(points))
print('기사 갯수 : ', len(contents))
print("사전 생성 완료 >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")

# 여기서 지워야함...#####################################################################

for county in range(0, 2000, 100):
    countx = 0
    for i in range(len(contents)):
        if county < len(contents[i]) < (county + 100):
            countx += 1
    print("기사 길이가 %d에서 %d 사이인 기사의 갯수 : " % (county, county + 100), countx)
print("기사 자르기 전 개수 :  ", len(contents))
minlen = int(input("앞에서부터 몇 이하의 기사 길이까지 버리겠습니까?(숫자 입력) : "))
maxlen = int(input("몇 이상부터 끝까지의 기사 길이까지 버리겠습니까?(숫자 입력) : "))

del_list = []
for i in range(len(contents)):
    if minlen < len(contents[i]) < maxlen:
        pass
    else:
        del_list.append(i)
del_list.sort(reverse=True)
for i in del_list:
    try:
        del contents[i]
        del points[i]
    except ValueError:
        pass
print("기사 자른 후 남은 contents의 갯수 :  ", len(contents))

for county in range(0, 2000, 100):
    countx = 0
    for i in range(len(contents)):
        if county < len(contents[i]) < (county + 100):
            countx += 1
    print("기사 길이가 %d에서 %d 사이인 기사의 갯수 : " % (county, county + 100), countx)

#### 0 갯수랑 1 갯수랑 맞춰주기!!#####
print("-" * 30)
print('현재 남은 하락 기사의 갯수 :', len(points) - sum(points))
print("현재 남은 상승 기사의 갯수 :", sum(points))
print("-" * 30)

while True:
    changeval = input("남은 상승/하락 기사의 갯수를 맞추시겠습니까? (y/n) :")
    changeval = changeval.lower()

    if changeval == 'y':
        diff = abs(len(points) - sum(points) - sum(points))
        up_idx = []
        down_idx = []
        for i in range(len(contents)):
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
                del contents[i]
                del points[i]
        elif len(points) - sum(points) < sum(points):
            for i in up_idx:
                del contents[i]
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

'''
content :
리스트 속 리스트 형식.
'''
print(len(contents))
print(contents[0])
print(contents[1])
print(type(contents))
