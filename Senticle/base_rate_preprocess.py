
import pandas as pd
import Senticle.cnn_tool as tool
import os
import csv
from soynlp.noun import LRNounExtractor_v2
from soynlp.tokenizer import NounLMatchTokenizer
from soynlp.utils import DoublespaceLineCorpus

company_name = 'article_threeClass'#input("RawData File Name? : ")
data_path = './labeled_'+company_name+'.csv'# + '_labeled_data.csv'  # csv 파일로 불러오기

# =============================================================================
# contents는 각 기사 스트링으로 바꿔 리스트에 넣은거, points는 클래스 0or 1
    # drop_zeor_label : 레이블 '상승','유지','하락' 중, '유지'에 해당하는 레이블 삭제
    # shuffle : 데이터 셔플
    # cutting : 상승, 하락 데이터 갯수 적은 쪽으로 통일.
# =============================================================================


contents, points = tool.loading_rdata(data_path, drop_zero_label=True, shuffle = True, cutting = True)
if os.path.isfile('preprocessed_' + company_name + '.csv') == False:
    print("\n")
    print('"preprocessed_' + company_name + '.csv" deos not EXIST!')
    print('MAKE "preprocessed_' + company_name + '.csv" FILE... 가즈아~!!')
    print("\n")

    noun_extractor = LRNounExtractor_v2(verbose=True)
    nouns = noun_extractor.train_extract(contents, min_noun_frequency=20)

    match_tokenizer = NounLMatchTokenizer(nouns)

# ==
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
    if j % 100 == 0:
        print("{}개의 기사 중 {}번 기사 불용어처리후 저장완료~ ^오^".format(len(contents), j + 1))



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

del_list = []
for i in range(len(dfdf)):
    print('delete null text..........')
    if len(dfdf['text'][i])==0:
        del_list.append(dfdf.index[i])
dfdf = dfdf.drop(del_list, axis = 0)
dfdf.to_csv('./preprocessed_article_threeClass.csv', index=True, header=True)