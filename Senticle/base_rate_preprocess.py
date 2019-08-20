import pandas as pd
import Senticle.cnn_tool as tool
import os
import csv
from soynlp.noun import LRNounExtractor_v2
from soynlp.tokenizer import NounLMatchTokenizer

company_name = 'rates'#input("RawData File Name? : ")
data_path = '../labeled_'+company_name+'.csv'# + '_labeled_data.csv'  # csv 파일로 불러오기

# =============================================================================
# contents는 각 기사 스트링으로 바꿔 리스트에 넣은거, points는 클래스 0or 1
# =============================================================================
contents, points = tool.loading_rdata(data_path)

if os.path.isfile('preprocessed_' + company_name + '.csv') == False:
    print("\n")
    print('"preprocessed_' + company_name + '.csv" deos not EXIST!')
    print('MAKE "preprocessed_' + company_name + '.csv" FILE... 가즈아~!!')
    print("\n")

    noun_extractor = LRNounExtractor_v2(verbose=True)
    nouns = noun_extractor.train_extract(contents, min_noun_frequency=20)

    match_tokenizer = NounLMatchTokenizer(nouns)

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


dfdf = pd.DataFrame(noun_contents,  columns=['text'])
dfdf['label'] = points
dfdf.to_csv('preprocessed_base_rates.csv', index=True, header=True)
