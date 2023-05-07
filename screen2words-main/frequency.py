from nltk import word_tokenize, pos_tag
import numpy as np
import json,math
import random
from tqdm import tqdm
from collections import Counter ,defaultdict
import re,nltk
import re
import pandas as pd
import csv

if __name__ == '__main__':
    df = pd.read_csv('./screen_summaries.csv')
    cnt_word=Counter()
    for x in tqdm(df['summary']):
        cnt_word.update(x.split(' '))
    count_words=cnt_word.most_common()

    res_words=[]
    # nltk.download()
    text = nltk.word_tokenize("And now for something completely different")
    tmp = nltk.pos_tag(text)
    print(tmp)
    for index_words in range(0,len(count_words)):
        text=nltk.word_tokenize(count_words[index_words][0])
        count=nltk.pos_tag(text)[0][1]
        res_words.append([count_words[index_words][0],count_words[index_words][1],count])

    #结果写入csv
    with open('calculate.csv', mode='w', newline='') as predict_file:
        csv_writer = csv.writer(predict_file)
        for row in range(len(res_words)):
            csv_writer.writerow(res_words[row])

    # 结果写入csv
    with open('top200NN.csv', mode='w', newline='') as predict_file:
        csv_writer = csv.writer(predict_file)
        nn_count=0
        for row in range(len(res_words)):
            if(nn_count>=200):
                break
            else:
                if(res_words[row][2]=='NN'):
                    nn_count+=1
                    csv_writer.writerow(res_words[row])
        # 结果写入csv
    with open('top200JJ.csv', mode='w', newline='') as predict_file:
        csv_writer = csv.writer(predict_file)
        jj_count = 0
        for row in range(len(res_words)):
            if (jj_count >= 200):
                break
            else:
                if (res_words[row][2] == 'JJ'):
                    jj_count += 1
                    csv_writer.writerow(res_words[row])

    #print(res_words)




#
# text = "I am learning Natural Language Processing on Analytic Vidhya"
# tokens = word_tokenize(text)
# print(pos_tag(tokens))
#
# word="I'm a boby, I'm a girl. When it is true, it is ture. thit are cats, the red is red."
# word=word.replace(',','').replace('.','')
# word=word.split()
# print(word)
# setword=set(word)
# for i in setword:
#     count=word.count(i)
#     print(i,'出现次数：',count)
