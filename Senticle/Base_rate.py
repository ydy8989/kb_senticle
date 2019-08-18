#-*- coding:utf-8 -*-

import pandas as pd
import numpy as np
import os
import datetime
'''
2009년부터 시작하기로함. 
    금리 변동성 3개 클래스로 분류
    0 : 앞으로 3개월 이내에 변동 없음
    1 : 앞으로 3개월 이내에 금리 상승 
    2 : 앞으로 3개월 이내에 금리 하락

주식처럼 매일 매일 변동하지 않기 때문에, 3개월을 기준으로 변동 정도를 3가지 클래스로 분류한다. 

금리 정보 : https://www.bok.or.kr/portal/singl/baseRate/list.do?dataSeCd=01&menuNo=200643
'''

# 금리 변동 시점(date)과 그 때의 금리(base_rate)
date = ['20090109','20090212','20100709','20101116','20110113','20110310','20110610','20120712','20121011','20130509','20140814','20141015','20150312','20150611','20160609','20171130','20181130','20190718']
base_rate = [2.5,2.0,2.25,2.5,2.75,3.0,3.25,3.0,2.75,2.5,2.25,2.0,1.75,1.5,1.25,1.5,1.75,1.5]
base_rate_updown = [-1, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 1, -1]

#사이사이 3개월 넘는 부분 채우기.
testtest = []
for i in range(len(date)-1):
    if datetime.datetime.strptime(date[i+1], '%Y%m%d')-datetime.datetime.strptime(date[i],  '%Y%m%d')>=datetime.timedelta(90):
        # (datetime.datetime.strptime(date[1 + 1], '%Y%m%d')-datetime.timedelta(90)).date()
        testtest.append(i+1)
        # print(i+1, date[i+1])

for i in testtest:
    print((datetime.datetime.strptime(date[i], '%Y%m%d') - datetime.timedelta(90)).date())


# pd.to_datetime(df['datetime'])
df = pd.read_csv('./Senticle/rates2.csv',error_bad_lines=False, header=None, names = ['date','headline'], encoding = 'utf-8')
df.shape
df.head(10)

df = df.set_index('date')
new_df_ind = df.index
new_df_val = df.text

new_df = pd.DataFrame(index=new_df_ind, data=0, columns=['label'])

# 날짜 아닌 형식의 인덱스는 리스트에 받아, 나중에 drop시키기
drop_index = []
for i in new_df.index:
    if len(i) != 14:
        drop_index.append(i)
df = df.drop(drop_index, axis=0)


#시간 형식으로 바꾸고, 정렬
df.index = pd.to_datetime(df.index)
df = df.sort_index()

#df에 label 달기.
for i in range(len(date)-1):
    # date[i]는 금리가 바뀌는 시점.
    # 바뀌는 시점 하루전은 아직 바뀌기 전임.
    # i = 16
    # date[i+1]-date[i]> 3개월, then date[i+1]-3개월 까지만 base_rate_updown[i]고, date[i] ~ (date[i+1]-3개월) 사이는 유지 : 0레이블
    if datetime.datetime.strptime(date[i+1], '%Y%m%d') - datetime.datetime.strptime(date[i], '%Y%m%d') < datetime.timedelta(90) :# 3개월 미만이면~~~~~
        df.loc[(datetime.datetime.strptime(date[i], '%Y%m%d')<=df.index) & (df.index<=datetime.datetime.strptime(date[i+1], '%Y%m%d')),'label'] = base_rate_updown[i]
    else: #3개월을 넘으면~
        # df.loc[(datetime.datetime.strptime(date[i], '%Y%m%d') < df.index) & (df.index <= datetime.datetime.strptime(date[i + 1], '%Y%m%d')), 'label'] =
        df.loc[(datetime.datetime.strptime(date[i + 1], '%Y%m%d') - datetime.timedelta(90))>= df.index,'label'] = base_rate_updown[i]
        df.loc[(datetime.datetime.strptime(date[i], '%Y%m%d'))< df.index,'label'] = 0

#TODO : 위에 레이블링 한번 다시 고쳐야함; ㅅㅂ.... 아니다 제대로 된건가...?



# print(new_df[-245:-220])
df.label.shift(1) - df.label


# timestamp랑 datetime이랑 연산 못하니깐 그거 참고해서 만들기
# 참고 사이트 : https://inma.tistory.com/96

new_df.index <= datetime.datetime.strptime(date[0], '%Y%m%d')

