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
import msvcrt

D2VModel = doc2vec.Doc2Vec()
doc2vecFile = '../../doc2vec_Doc.model'

Rating_Train_File = "../../ratings_train.txt"
Rating_Test_File = "../../ratings_test.txt"

pos_tagger = Twitter()

def read_data(filename):
    with open(filename, 'r', encoding = 'utf-8') as f:
        data = [line.split("\t") for line in f.read().splitlines()]
        data = data[1:] # header extract.
        return data

def tokenize(doc):
    return [t for t in pos_tagger.morphs(doc)]
    #return ['/'.join(t) for t in pos_tagger.pos(doc, norm=True, stem=True)]

class LabeledLineSentence(object):
    def __init__(self, dataList):
        self.data = dataList
    def __iter__(self):
        for uid, line in enumerate(self.data):
            #print('uid: ' + str(uid))
            yield doc2vec.LabeledSentence(line, ['SENT_%s' %uid])
    def to_array(self):
        self.sentences = []
        for uid, line in enumerate(self.data):
            self.sentences.append(doc2vec.LabeledSentence(line, ['SENT_%s' %uid]))
        return self.sentences
        

# func sentiment classification with doc2vec (gensim)
def procGensim(d1, modelName):
    tagged_train_docs = LabeledLineSentence(d1)

    doc_vectorizer = doc2vec.Doc2Vec(size=300, alpha=0.025, min_alpha=0.025, seed=1234)
    doc_vectorizer.build_vocab(tagged_train_docs)
    #doc_vectorizer.build_vocab(tagged_train_docs.to_array())

    s_time = time.time()
    for epoch in tqdm(range(10), ncols = 47, ascii = True, desc = 'doc2vec_train'):
        doc_vectorizer.train(tagged_train_docs, total_examples=1, epochs=5)
        doc_vectorizer.alpha -= 0.002
        doc_vectorizer.min_alpha = doc_vectorizer.alpha
    print(time.time() - s_time)

    # save this model
    print('Saved Data: ' + """ + modelName """)
    doc_vectorizer.save(modelName)

    # discard unneeded model memory.
    doc_vectorizer.delete_temporary_training_data(keep_doctags_vectors=True, keep_inference=True)

    return doc_vectorizer

# func end

def similar_Word(word):
    retStr = D2VModel.most_similar(word)
    pprint(retStr)

def similar_Doc(tag):
    infer_vector = D2VModel.infer_vector(tag)
    retStr = D2VModel.most_similar([infer_vector])
    pprint(retStr)


if __name__=='__main__':
    print('Train or See a Result?(T/R)')
    key = msvcrt.getch()
    if key == b't' or key == b'T':
        train_data = read_data(Rating_Train_File)
        # truncate 1000 from original's one for test.
        #train_data = train_data[:1000]
        train_doc = [tokenize(row[1]) for row in tqdm(train_data, ncols = 47, ascii = True, desc = 'tarin_doc')]
        #train_doc = [(tokenize(row[1]), row[2]) for row in tqdm(train_data, ncols = 47, ascii = True, desc = 'tarin_doc')]
        D2VModel = procGensim(train_doc, doc2vecFile)
    else:
        D2VModel = doc2vec.Word2Vec.load(doc2vecFile)
        train_data = read_data(Rating_Train_File)
        train_data = train_data[:1000]

    print('단어/문장(w/d)?')
    key = msvcrt.getch()
    if key == b'w' or key == b'W':
        tarStr = input('단어를 입력해 주세요. ==> ')
        print('단어: '+tarStr)
        similar_Word(tarStr)
    else:
        print('문장번호/단문(n/s)?')
        key = msvcrt.getch()
        if key == b'n' or key == b'N':
            tarID = input('문장을 대표하는 번호를 입력하세요 ==> ')
            tarStr = 'SENT_' + tarID
            print('문장: ' + train_data[int(tarID)][1])
            similar_Doc(tarStr)
            #similar_Word(tarStr)
        else:
            tarStr = input('단문을 입력하세요 ==> ')
            similar_Doc(tarStr)
            #similar_Word(tarStr)

    print('끝 @.@')

