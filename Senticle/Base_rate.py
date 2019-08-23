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

def load_csv(filepath):
    print('Loading........')
    df = pd.read_csv(filepath, error_bad_lines=False, header=None, names = ['date','text'], encoding = 'utf-8')

    #date to index and to datetime
    df = df.set_index('date')

    #null값 제거 및 이상치 제거.
    df = df.dropna(axis = 0)
    drop_index = []
    for i in df.index:
        if len(i) != 12:
            drop_index.append(str(i))
    df = df.drop(drop_index, axis=0)

    #인덱스 object to datetime
    df.index = pd.to_datetime(df.index.astype(str))
    #index sorting
    df = df.sort_index()
    print('Complete!!')
    return df

def label_df(basic_df):
    # 금리 변동 시점(date)과 그 때의 금리(base_rate)
    basic_df
    date = ['20090109', '20090212', '20100410', '20100709', '20100818', '20101116',
            '20110113', '20110310', '20110312', '20110610', '20120413', '20120712',
            '20120713', '20121011', '20130208', '20130509', '20140516', '20140814',
            '20141015', '20141212', '20150312', '20150313', '20150611', '20160311',
            '20160609', '20170901', '20171130', '20180901', '20181130', '20190419',] # 현재 기준금리 부분은 빼자, 테스트용으로 '20190718']
    # base_rate = [2.5,2.0,2.25,2.5,2.75,3.0,3.25,3.0,2.75,2.5,2.25,2.0,1.75,1.5,1.25,1.5,1.75,1.5]
    base_rate_updown = [-1, 0, 1, 0, 1, 1, 1, 0, 1, 0, -1, 0, -1, 0, -1, 0, -1, -1, 0, -1, 0, -1, 0, -1, 0, 1, 0, 1, 0, -1]

    #최초 금리 날짜 이전 데이터 있으면 삭제
    basic_df = basic_df[basic_df.index > datetime.datetime.strptime('20090109', '%Y%m%d')]

    #본격 레이블링
    for i in range(len(date)-1):
        basic_df.loc[(datetime.datetime.strptime(date[i], '%Y%m%d') <= basic_df.index) & (basic_df.index <= datetime.datetime.strptime(date[i + 1], '%Y%m%d')), 'label'] = base_rate_updown[i]

    #label이 NaN인 부분 : 앞으로 3개월 이내에 오를지 안오를지 모르기 때문이라고 가정
    # null값 처리
    basic_df = basic_df.dropna(axis = 0)
    return basic_df

if __name__ == '__main__':
    filepath = './Senticle/rates2.csv'
    basic_df = load_csv(filepath)
    import re
    basic_df['text'] = basic_df['text'].apply(lambda x:re.compile('[^ ㄱ-ㅣ가-힣]+').sub('',x))
    labeled_df = label_df(basic_df)
    labeled_df.to_csv('labeled_rates.csv',index=True, header=True)