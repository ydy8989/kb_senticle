import os
import pandas as pd
from soynlp.tokenizer import NounLMatchTokenizer
from soynlp.noun import LRNounExtractor_v2

import pickle
import tensorflow as tf
import numpy as np
import Senticle.cnn_tool as tool
from Senticle.main import TextCNN
from lime.lime_text import LimeTextExplainer
company = 'article'  # input('RawData File Name? :')

if 'nouns.data' not in os.listdir():
    data_path = './preprocessed_' + company + '_del.csv'
    doc = pd.read_csv(data_path)[['text', 'label']]
    contents = []
    points = []
    for i in range(0, len(doc['text'])):
        if len(str(doc['text'][i])) > 0:
            contents.append(doc['text'][i])
            points.append(doc['label'][i])
    noun_extractor = LRNounExtractor_v2(verbose=True)
    nouns = noun_extractor.train_extract(contents, min_noun_frequency=0)
    with open('./nouns.data', 'wb') as f:
        pickle.dump(nouns, f, pickle.HIGHEST_PROTOCOL)

SEQUENCE_LENGTH = 1400
NUM_CLASS = 2

def test(input_text):

    with tf.Session() as sess:

        vocab = tool.load_vocab(root_path+'/Senticle/'+company+'_vocab.txt')

        CNN = TextCNN(SEQUENCE_LENGTH, NUM_CLASS, len(vocab), 128, [3,4,5], 128)
        saver = tf.train.Saver()

        saver.restore(sess, root_path+'/Senticle/runs/1566669796/checkpoints/model-800')

        print('model restored')




        tokens = tool.model_tokenize(input_text)

        sequence = [tool.get_token_id(t, vocab) for t in tokens]

        x = []
        while len(sequence) > 0:
            seq_seg = sequence[:SEQUENCE_LENGTH]
            sequence = sequence[SEQUENCE_LENGTH:]

            padding = [1] * (SEQUENCE_LENGTH - len(seq_seg))
            seq_seg = seq_seg + padding

            x.append(seq_seg)

        feed_dict = {
            CNN.input_x: x,
            CNN.dropout_keep_prob:1.0
        }

        predict = sess.run([CNN.predictions], feed_dict)

        result = np.mean(predict)

        if result == 1.0:
            print('하락')
        else:
            print('상승')

        test = sess.run(CNN.final, feed_dict)

        print(test)

        def predict_fn(x):
            predStorage = []
            for i in x:
                tokens = tool.model_tokenize(i)
                sequence = [tool.get_token_id(t, vocab) for t in tokens]
                text = []
                if len(sequence) > 0:
                    seq_seg = sequence[:SEQUENCE_LENGTH]
                    sequence = sequence[SEQUENCE_LENGTH:]

                    padding = [1] * (SEQUENCE_LENGTH - len(seq_seg))
                    seq_seg = seq_seg + padding

                    text.append(seq_seg)
                else:
                    padding = [0] * (SEQUENCE_LENGTH)
                    text.append(padding)

                feed_dict = {
                    CNN.input_x: text,
                    CNN.dropout_keep_prob: 1.0
                }

                scores = sess.run(CNN.final, feed_dict)


                predStorage.append(np.squeeze(scores))

            return np.array(predStorage)

        explainer = LimeTextExplainer(class_names=['상승', '하락'])
        print('LIME 시각화 준비중.....(대기)')
        exp = explainer.explain_instance(input_text, predict_fn, num_features=6, num_samples=1200)
        exp.save_to_file(root_path+'/LIME_visualization.html')

if __name__=='__main__':
    root_path = '/home/ydy8989/PycharmProjects/kb_senticle'
    input_text = input('평가할 뉴스 입력 (엔터없이 한 줄로 입력): ')
    temp = test(input_text)

'''
[상승기사 예시1] - 예측 성공
2금융권 연체율도 일제히 상승 LTV 60% 초과 주택대출 3분의1 경기 침체가 장기화하면서 취약층이 주로 이용하는 2금융권과 저소득ㆍ저신용자를 대상으로 한 서민금융 대출상품의 연체율이 치솟고 있다. 실물경기 둔화로 벌이는 시원찮은데 시장금리 상승으로 빚 부담은 늘면서 제때 대출이자를 갚지 못하고 있다는 얘기다. 1,500조원에 달하는 가계부채의 가장 약한 고리인 취약층의 부실화 우려가 커지는 상황이다.7일 국회 정무위원회 소속 이태규 바른미래당 의원이 금융감독원에서 받은 자료에 따르면 올 들어 정부가 운영하는 서민금융상품의 연체액이 급증하는 추세다. 서민금융진흥원의 보증을 기반으로 금융권이 저신용자에게 빌려주는 햇살론은 2016년 말 평균 2.19%였던 연체율(대위변제율)이 올해 7월 말 8.1%로 3.7배 급증했다. 은행에 비해 저신용 대출자가 많은 저축은행의 연체율(상반기 4.8%)과 비교해도 높은 수준이다.햇살론의 높은 연체율은 취약층의 빚 부담이 그만큼 크다는 걸 의미한다. 실제 차주가 대출을 상환하지 못해 서민금융진흥원이 대신 갚아준 건수(누적 기준)는 2016년 5,201건에서 지난해 말 3만2,825건으로 늘더니 올해 7월엔 6만684건으로 배 가까이 급증했다. 저신용자일수록 연체율이 급증했다. 2016년 말 신용 8등급 차주의 연체율은 6.01%였는데 올해 들어선 19.85%로 3.3배 급증했다.대부업체 연체율도 뛰고 있다. 2016년 말 4.8%였던 대부업체 연체율은 올해 7월 말엔 6.3%로 1.5%포인트 커졌다. 특히 대부업 연체자 중에서도 60대 이상 남성 노인과 30세 미만 청년층의 연체율 상승이 두드러진다. 60세 이상 남성의 대부업 평균 연체율은 지난해 말 6.2%에서 올해 7월 9.8%로 3.6%포인트 급등했다. 60대 남성 노인 10명 중 1명은 빚을 제때 못 갚고 있는 셈이다. 19세 이상 30세 미만 남성의 연체율은 8.4%로 그 뒤를 이었다. 취업준비생을 포함한 30세 미만 청년층과 경제활동이 거의 없는 60대 이상 은퇴 노년층이 다른 계층에 비해 심각한 경제적 빈곤에 시달리고 있는 것으로 풀이된다.업권별로는 1금융권에 견줘 2금융권 연체율 오름폭이 더 크다. 은행의 가계대출 연체율은 지난해 6월과 올해 6월 0.25%로 같지만 같은 기간 보험은 0.49%에서 0.54%로, 상호금융은 1.38%에서 1.42%로 올랐다. 상대적으로 저신용자들이 집중되는 저축은행은 4.34%에서 4.8%로, 여신전문금융사는 3.33%에서 3.62%로 뛰었다.한국은행이 하반기 중 기준금리를 인상하면 취약층은 물론 주택담보대출 차주 역시 상당한 타격을 받을 걸로 보인다. 현재 차주가 집값의 60% 넘게(LTV 60% 초과) 빌린 은행권 주택대출 규모는 전체의 3분의 1 수준인 153조원으로 추산된다. 정부는 신규 대출을 억제하는 차원에서 이달 중순 가계대출 규제의 최종판인 총체적상환능력(DSR) 규제 방안을 내놓고 내달부터 곧바로 시중은행부터 시행에 들어갈 예정이다.

[상승기사 예시2] - 예측 실패
이주열 한국은행 총재가 우리나라 경제 성장률을 하향 전망하면서도 올해 안에 금리를 올리겠단 의지를 내비쳤습니다. 최근 정부 고위 당국자들의 잇따른 금리 인상 언급에 대해서는 통화 정책의 중립성을 강조했습니다. 홍진아 기자가 보도합니다. 이주열 한국은행 총재는 3%에서 2.9%로 낮췄던 올해 성장률 전망치를 조금 더 낮출 가능성이 있다고 말했습니다. 어느 정도의 하향 조정은 여러 지표상으로도 예측된 상황입니다. 이 총재는 그러면서 "잠재성장률 수준의 성장세가 이어지고 물가가 목표 수준에 근접해나간다는 판단이 서면 '금융 안정'도 비중 있게 고려할 시점"이라고 강조했습니다. 여기서 금융안정은 금리 인상을 말합니다.이 총재가 그동안 금리에 대해서만큼은 신중에 신중을 기해 발언수위를 최대한 낮춰온 점을 생각하면 이번 언급은 금리 인상을 시장에 확실히 예고한 것으로 읽힙니다.가계부채 증가속도가 소득이 느는 것보다 빨라 금융안정을 해칠 위험이 크고, 대외적으로는 미국이 이미 12월 금리 인상을 예고해놓고 있어 한미 금리 차가 더 벌어진다면 자본유출 우려가 커진다는 점을 들었습니다.이 총재는 최근 이낙연 국무총리에 이어 김현미 국토부 장관이 부동산 시장 안정을 위한 금리 인상 필요성을 언급한 것에 대해서도 처음으로 입장을 밝혔습니다.이 총재는 "외부 의견을 너무 의식해 금리 인상 필요에도 인상을 하지 않거나, 인상이 적절치 않은데 인상하는 결정은 내리지 않으려 한다"며 "주택 가격 상승은 복합적인 요인이 작용한 결과인데 현시점에서 주된 요인을 따지는 논쟁은 바람직하지 않고, 정책 당국자들이 협력해 나가는 노력이 필요하다"고 말했습니다.이달과 다음 달 올해 두 번 남은 금융통화위원회 가운데 언제 기준금리를 올릴지는 아직까지 관측이 엇갈리고 있습니다.

[하락기사 예시1] - 예측 성공
미·중 무역갈등이 격화되고 있는 가운데 미국 대형은행 JP모건이 미국의 2분기 국내총생산(GDP) 성장률 전망치를 2.2%에서 1.0%로 낮췄다고 로이터통신 등이 24일(현지 시각) 보도했다. JP모건은 GDP 성장률 전망치를 하향 조정한 이유로 일부 경제지표 부진과 미·중 무역갈등 악화를 꼽았다. JP모건은 "4월 소매판매가 부진했고 내구재 수주도 나빴다"며 "이는 2분기 경제활동이 1분기보다 급격히 악화되고 있다는 것을 나타낸다"고 했다. 앞서 이날 미 상무부는 4월 내구재 수주 실적이 지난달 대비 2.1% 감소했다고 발표했다. 미국의 소매판매는 같은 기간 0.2% 감소했다. 5월 제조업 구매관리자지수(PMI)도 50.6으로 금융위기 직후인 2009년 9월 이후 약 10년 만에 최저치로 떨어졌다. 또 JP모건은 미·중 무역갈등에 따른 불확실성이 커지면서 미 경제성장을 더디게 만들고 있다고 지적했다. 미 연방준비제도이사회(FRB·연준)의 기준금리 정책과 관련해 JP모건은 "금리 인상과 인하 가능성이 비슷한 수준으로 있다"고 전망했다. 미국의 분기 GDP 성장률은 지난해 2분기 4%대로 정점을 찍은 뒤 3분기 3.4%, 4분기 2.2%로 급격히 하락했다. 올해 1분기 GDP 성장률은 3.2%를 기록했으나, 이는 향후 잠정치, 확정치에서 수정될 수 있다.

[하락기사 예시2] - 예측 

'''