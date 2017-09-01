
from konlpy.tag import Twitter
from timeit import default_timer as timer
from pprint import pprint
from matplotlib import font_manager, rc
font_fname = "C:/Windows/Fonts/malgun.ttf"
font_name = font_manager.FontProperties(fname=font_fname).get_name()
rc('font', family=font_name)
import nltk
import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
from konlpy.tag import Kkma
from konlpy.corpus import kolaw
from nltk import collocations
import os
from glob import glob
from gensim.models import doc2vec
#from gensim.models.doc2vec import TaggedDocument
from collections import namedtuple
from sklearn.linear_model import LogisticRegression

import sys
import util
import time
from tqdm import *
from copy import copy

import csv

D2VModel = doc2vec.Doc2Vec()
doc2vecFile = '../../doc2vec.model'

Rating_Train_File = "../../ratings_train.txt"
Rating_Test_File = "../../ratings_test.txt"

Train_Token_File = "../../train_token.csv"
Test_Token_File = "../../test_token.csv"

Train_Token_Idx = "../../tarin_idx.txt"
Test_Token_Idx = "../../test_idx.txt"

pos_tagger = Twitter()

def read_data(filename):
    with open(filename, 'r', encoding = 'utf-8') as f:
        data = [line.split("\t") for line in f.read().splitlines()]
        data = data[1:] # header extract.
        return data

def tokenize(doc):
    return ['/'.join(t) for t in pos_tagger.pos(doc, norm=True, stem=True)]

# func sentiment classification with doc2vec (gensim)
def procGensim(d1, modelName):
    TaggedDocument = namedtuple('TaggedDocument', 'words tags')
    tagged_train_docs = [TaggedDocument(d, [c]) for d, c in d1]

    doc_vectorizer = doc2vec.Doc2Vec(size=300, alpha=0.025, min_alpha=0.025, seed=1234)
    doc_vectorizer.build_vocab(tagged_train_docs)

    s_time = time.time()
    for epoch in tqdm(range(10), ncols = 47, ascii = True, desc = 'doc2vec_train'):
        doc_vectorizer.train(tagged_train_docs, total_examples=1, epochs=5)
        doc_vectorizer.alpha -= 0.002
        doc_vectorizer.min_alpha = doc_vectorizer.alpha
    print(time.time() - s_time)

    # save this model
    print('Saved Data: ' + """ + modelName """)
    doc_vectorizer.save(modelName)
    return doc_vectorizer

# func end

def simpleTestD2V(word):
    retStr = D2VModel.most_similar(word)
    pprint(retStr)

if __name__=='__main__':
    print('Train or See a Result?(T/R)')
    input_data = input('T/R\n')
    if input_data == 't' or input_data == 'T':
        train_data = read_data(Rating_Train_File)
        train_doc = [(tokenize(row[1]), row[2]) for row in tqdm(train_data, ncols = 47, ascii = True, desc = 'tarin_doc')]
        #train_doc = train_doc[:1000]
        #test_doc = read_data(Rating_Test_File)
        #test_doc = test_doc[:1000]
        D2VModel = procGensim(train_doc, doc2vecFile)
    else:
        D2VModel = doc2vec.Word2Vec.load(doc2vecFile)

    tarStr = '재미있는뎅/Noun'
    print('가장 가까운 의미: ' + tarStr)
    simpleTestD2V(tarStr)
    print('끝 @.@')

    #train_x = [D2VModel.infer_vector(doc.words) for doc in tqdm(tagged_train_docs, ncols = 47, ascii = True, desc = 'tarin_x')]
    #train_y = [doc.tags[0] for doc in tqdm(tagged_train_docs, ncols = 47, ascii = True, desc = 'tarin_x')]
    #print(len(train_x))
    #print(len(train_x[0]))

    #test_x = [D2VModel.infer_vector(doc.words) for doc in tqdm(tagged_test_docs, ncols = 47, ascii = True, desc = 'tarin_x')]
    #test_y = [doc.tags[0] for doc in tqdm(tagged_test_docs, ncols = 47, ascii = True, desc = 'tarin_x')]
    #print(len(test_x))
    #print(len(test_x[0]))

    #classifier = LogisticRegression(random_state=1234)
    #classifier.fit(train_x, train_y)
    #print(classifier.score(test_x, test_y))

    #while True:
    #    print("학습 확인을 시작합니다.")
    #    print("확인하고 싶은 단어를 입력해 주세요")
    #    print("ex) 공포/Noun")
    #    print("종료를 원하시면 q 또는 Q 를 입력해 주세요")
    #    input_data = input("학습 단어를 입력해 주세요 :")
    #    if input_data == 'q' or input_data == 'Q':
    #        break
    #    pprint(D2VModel.most_similar(input_data))
    #    print('\n')
