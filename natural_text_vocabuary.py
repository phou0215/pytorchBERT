import urllib
import sys
import os
import pandas as pd
from konlpy.tag import Okt
from nltk import FreqDist
import numpy as np
import matplotlib.pyplot as plt


def file_check():
    if os.path.isfile('naver_movie_ratings.txt'):
        print('테스트 데이터 존재함')
    else:
        urllib.request.urlretrieve("https://raw.githubusercontent.com/e9t/nsmc/master/ratings.txt", filename=file_name)
        print('테스트 데이터 다운로드 완료')

file_name = 'naver_movie_ratings.txt'
stop_list=['의', '가', '이', '은', '들', '는', '좀', '잘', '걍', '과', '도', '를', '으로', '자', '에', '와', '한', '하다']
file_check()
data = pd.read_csv(file_name, sep='\t')
# 데이터프레임에 저장

separate_data = data[:200]
separate_data['document'] = separate_data['document'].str.replace(r"[^ㄱ-ㅎㅏ-ㅣ가-힣 ]", "", regex=True)
tokenizer = Okt()
tokenized = []
for sentence in separate_data['document']:
    temp = tokenizer.pos(sentence)
    # temp = [word[0] for word in temp if not word[0] in stop_list and word[1] in ['Verb', 'Noun']]
    temp = [word[0] for word in temp if not word[0] in stop_list]
    tokenized.append(temp)
vocab = FreqDist(np.hstack(tokenized))
vocab_size = 500
vocab = vocab.most_common(vocab_size)
print('단어 집합의 크기: {}'.format(len(vocab)))
word_to_index = {word[0]:idx+2 for idx, word in enumerate(vocab)}
word_to_index['unk'] = 0
word_to_index['pad'] = 1

embedding = []
for line in tokenized:
    temp = []
    for word in line:
        try:
            temp.append(word_to_index[word])
        except KeyError:
            temp.append(word_to_index['unk'])
    embedding.append(temp)

# 동일한 길이로 padding
max_len = max(len(wl) for wl in embedding)
# print('리뷰의 최대 길이 : %d' % max_len)
# print('리뷰의 최소 길이 : %d' % min(len(wl) for wl in embedding))
# print('리뷰의 평균 길이 : %f' % (sum(map(len, embedding))/len(embedding)))
# plt.hist([len(s) for s in embedding], bins=100)
# plt.xlabel('length of sample')
# plt.ylabel('number of sample')
# plt.show()
for line in embedding:
    if len(line) < max_len:
        line += [word_to_index['pad']]*(max_len-len(line))

print(embedding)
