#coding=utf-8
import re
import gensim
import pandas as pd
import jieba
import xlrd

def extract_words():
    # 语料预处理
    def open_file(file="D:/workSpace/nlp/tc/复旦大学中文文本分类语料.xlsx"):
        workbook = xlrd.open_workbook(file) # 表
        worksheet = workbook.sheet_by_name("sheet1") # 工作区
        return [worksheet.row_values(row) for row in range(worksheet.nrows)]

    def remove(text):
        keep_chars = r'[^\u4e00-\u9fa5]' # 所有中文编码
        return re.sub(keep_chars, "", text)

    def open_stop(file="D:/workSpace/nlp/tc/stopwords.txt"):
        stopwords = [line.strip() for line in open(file, "r", encoding="utf-8").readlines()]
        return stopwords

    def split_sentence(sentence):
        split_list = jieba.cut(sentence.strip())
        stopwords = open_stop()
        res_list = []
        for word in split_list:
            if word not in stopwords:
                res_list.append(word)
        return " ".join(res_list)
    inputs = open_file()[1:]
    words = []
    for line in inputs:
        words.append((line[0], split_sentence(remove(line[1]))))
    df = pd.DataFrame(words,columns=['label','words'])
    df.to_csv("article_features_train_raw.csv",encoding='utf_8_sig',index=False)
# extract_words()

def word2vec():
    # 训练词向量模型
    df = pd.read_csv("article_features_train_raw.csv")
    df = df.dropna(axis=0, how="any") # 删除含nan的行
    # print(df["label"].drop_duplicates().values.tolist())
    sent_list = df["words"].values.tolist()  # 句子列表
    word_list = [word.split() for word in sent_list] # 词表列表
    # 训练词向量模型
    model = gensim.models.Word2Vec(word_list,
                                   sg=1,
                                   size=100,
                                   window=3,
                                   iter=5,
                                   min_count=3,
                                   negative=3,
                                   sample=0.001,
                                   hs=1)
    model.wv.save_word2vec_format('./word2vec_model.txt', binary=False)
# word2vec()

def useword2vec():
    model = gensim.models.KeyedVectors.load_word2vec_format("word2vec_model.txt")
    print(model.most_similar("足球", topn=10))
# useword2vec()
