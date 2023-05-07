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
from mydifflib import get_close_matches_indexes
#页面分类
#把名词词频取前200页面进行分类，分类后将screenId以及其页面类型写入csv中
if __name__ == '__main__':
    df = pd.read_csv('./screen_summaries.csv')#读入
    cmp_labels=[
        'Calendar Page','Calculator Page','Camera Page','Address Page','Translator Page','Schedule Page','Email Page',
        'Chat Page','Timer Page', 'Question Page','Weather Page','GPS Page','Install Page','Product Page','Search Page',
        'News Page','Guide Page','Start Page','Update Page','Introduction Page','Home Page','Login Page', 'Pop Window',
        'Blank Page']
    cmp_words=[
         ['calendar','calender'],
         ['calculator'],
         ['camera'],
         ['address'],
         ['translator'],
         ['schedule'],
         ['email', 'mail', 'send', 'post'],
         ['chat', 'conversation'],
         ['timer', 'clock', 'time', 'alarm'],
         ['question'],
         ['weather', 'forecast'],
         ['navigator', 'location', 'map', 'gps', 'locator'],
         ['install'],
         ['product'],
         ['search'],
         ['news', 'article'],
         ['continue', 'reminder', 'guide', 'instruction'],
         ['start', 'setup', 'initial'],
         ['update', 'upgrade'],
         ['introduction', 'description'],
         ['home', 'profile', 'user', 'person', 'personal'],
         ['sign','password','login','log','registration','log-in','signing','enter','verification','access','confirmation'],
         ['notification','alert','popup','pop-up','pop','push','window','privacy','accept','policy','agreement','permission','error','license'],
         ['blank', 'empty', 'no']

    ]
    res=[]
    words_summary = df['summary'].to_list()
    id_summary=df['screenId']
    visited=[]
    index_row=0
    for tmp_row in cmp_words:
        res_index=[]
        for tmp_word in tmp_row:
            res_index=res_index+df[df['summary'].str.find(tmp_word)!=-1]['screenId'].to_list()
        for tmpi in res_index:
            if(tmpi not in visited):
                visited.append(tmpi)
                res.append([tmpi,cmp_labels[index_row]])
        index_row+=1


    #结果写入csv
    with open('ricoLabel.csv', mode='w', newline='') as predict_file:

        csv_writer = csv.writer(predict_file)
        csv_writer.writerow(["screenId","ricoLabel"])
        for row in range(len(res)):
            csv_writer.writerow(res[row])


    # 统计结果，共计16637份数据(调整顺序前)
    # ('Login', 3667), ('Pop', 3346), ('Window', 3346), ('Blank', 2898), ('Home', 2140), ('GPS', 833),
    # ('Search', 761), ('News', 542), ('Guide', 422), ('Start', 308), ('Product', 291), ('Timer', 229),
    # ('Email', 196), ('Introduction', 177), ('Question', 158), ('Chat', 148), ('Weather', 115),
    # ('Update', 100), ('Camera', 85), ('Install', 70), ('Schedule', 51), ('Translator', 48),
    # ('Calculator', 37), ('Address', 13), ('Calender', 2)

    #(调整优先级顺序后)
    # ('Pop', 2673), ('Window', 2673), ('Login', 2136), ('Home', 2012), ('Search', 1129),
    # ('Guide', 1091), ('GPS', 1089), ('News', 820), ('Start', 690), ('Email', 681),
    # ('Blank', 667), ('Timer', 525), ('Product', 498), ('Weather', 491), ('Chat', 443),
    # ('Question', 320), ('Camera', 235), ('Update', 229), ('Calender', 189), ('Address', 184),
    # ('Install', 178), ('Introduction', 172), ('Schedule', 127), ('Translator', 80), ('Calculator', 71)
    df = pd.read_csv('./ricoLabel.csv')
    cnt_word = Counter()
    for x in tqdm(df['ricoLabel']):
        cnt_word.update(x.split(' '))
    count_words = cnt_word.most_common()
    print(count_words)