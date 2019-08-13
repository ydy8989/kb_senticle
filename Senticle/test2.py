import pandas as pd
from soynlp.tokenizer import NounLMatchTokenizer
from soynlp.noun import LRNounExtractor_v2

import pickle
import tensorflow as tf
import numpy as np
import cnn_tool as tool
from main import TextCNN
from lime.lime_text import LimeTextExplainer

data_path = 'preprocessed_gold.csv'

doc = pd.read_csv(data_path)

contents = []
points = []

for i in range(0, len(doc['text'])):
    if len(str(doc['text'][i])) > 100:
        contents.append(doc['text'][i])
        points.append(doc['num'][i])

noun_extractor = LRNounExtractor_v2(verbose=True)
nouns = noun_extractor.train_extract(contents, min_noun_frequency=50)

with open('./nouns_gold.data', 'wb') as f:
    pickle.dump(nouns, f, pickle.HIGHEST_PROTOCOL)

SEQUENCE_LENGTH = 1400
NUM_CLASS = 2


def test2():

    with tf.Session() as sess:

        vocab = tool.load_vocab('gold_vocab.txt')

        CNN = TextCNN(SEQUENCE_LENGTH, NUM_CLASS, len(vocab), 128, [3,4,5], 128)
        saver = tf.train.Saver()

        saver.restore(sess, './runs/1565598160/checkpoints/model-100')

        print('model restored')

        input_text = input('평가할 뉴스 입력 : ')

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

        exp = explainer.explain_instance(input_text, predict_fn, num_features=6, num_samples=1400)
        key_list = exp.as_list()
        exp.save_to_file('./oi.html')

if __name__=='__main__':
    temp = test2()