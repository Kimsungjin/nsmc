from konlpy.tag import Twitter
from timeit import default_timer as timer
from pprint import pprint
from matplotlib import font_manager, rc
#font_fname = 'c:/windows/fonts/gulim.ttc'
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

Rating_Train_File = "../../ratings_train.txt"
Rating_Test_File = "../../ratings_test.txt"

Train_Token_File = "../../train_token.csv"
Test_Token_File = "../../test_token.csv"

Train_Token_Idx = "../../tarin_idx.txt"
Test_Token_Idx = "../../test_idx.txt"

pos_tagger = Twitter()

def read_csv(file_name):
    csvList = []
    with open(file_name, 'r', encoding = 'utf-8') as f:
        rdr = csv.reader(f)
        for row in rdr:
            csvList.append((row[:-1], row[-1]))
        return csvList
        

def write_csv(file_name, contents):
    with open(file_name, 'a', encoding = 'utf-8', newline = '') as f:
        csv.writer(f).writerow(contents)

def read_data(filename):
    with open(filename, 'r', encoding = 'utf-8') as f:
        data = [line.split("\t") for line in f.read().splitlines()]
        data = data[1:] # header extract.
        return data

def tonkenize(doc):
    # norm, stem is optional

    #temp_str = ''
    #for t in pos_tagger.pos(doc, norm=True, stem=True):
    #    temp_str += '/'.join(t) + ','

    #temp = temp_str[:-1]
    #return temp

    return ['/'.join(t) for t in pos_tagger.pos(doc, norm=True, stem=True)]

# func def save/read token
def procToken(path, t1, t2):
    r_csv = tonkenize(t1)
    r_csv.append(t2)
    write_csv(path, r_csv)
    return (r_csv[:-1], t2)
# func end

def tokenizing():
    # data packing..
    #pstart = time.time()

    # data tokenizing.
    print('@@@ Start Tokenizing @@@\n')
    tstart = time.time()

    train_doc = []
    train_data = read_data(Rating_Train_File)
    train_idx =  0
    if os.path.exists(Train_Token_Idx):
        train_idx = int(util.read_txt(Train_Token_Idx))

    if os.path.exists(Train_Token_File) and train_idx + 1 == len(train_data):
        train_doc = read_csv(Train_Token_File)
    else:
        #print(len(train_data))
        #print(len(train_data[0]))
        #train_doc = [procToken(Train_Token_File, row[1], row[2]) for row in tqdm(train_data, ncols = 47, ascii = True, desc = 'tarin_doc')]
        train_doc = [(tonkenize(row[1]), row[2]) for row in tqdm(train_data, ncols = 47, ascii = True, desc = 'tarin_doc')]
        temp_train_doc = copy(train_doc)
        for idx, r in tqdm(enumerate(temp_train_doc), ncols = 47, ascii = True, desc = 'tarin_csv'):
            if idx <= train_idx:
                continue

            r[0].append(r[1])
            write_csv(Train_Token_File, r[0])
            util.write_txt(str(idx), Train_Token_Idx)

    test_doc = []
    test_data = read_data(Rating_Test_File)
    test_idx = 0
    if os.path.exists(Test_Token_Idx):
        test_idx = int(util.read_txt(Test_Token_Idx))

    if os.path.exists(Test_Token_File) and test_idx + 1 == len(test_data):
        test_doc = read_csv(Test_Token_File)
    else:
        #print(len(test_data))
        #print(len(test_data[0]))
        #test_doc = [procToken(Test_Token_File, row[1], row[2]) for row in tqdm(test_data, ncols = 47, ascii = True, desc = 'test_doc')]
        test_doc = [(tonkenize(row[1]), row[2]) for row in tqdm(test_data, ncols = 47, ascii = True, desc = 'test_doc')]
        temp_test_doc = copy(test_doc)
        for idx, r in tqdm(enumerate(temp_test_doc), ncols = 47, ascii = True, desc = 'test_csv'):
            if idx <= test_idx:
                continue

            r[0].append(r[1])
            write_csv(Test_Token_File, r[0])
            util.write_txt(str(idx), Test_Token_Idx)
    
    #print(len(train_data))
    #print(len(train_data[0]))
    #print(len(test_data))
    #print(len(test_data[0]))
    #pend = time.time()
    #elapsed = pend - pstart
    #print('data packing time : ', elapsed)

    # data tokenizing.
    #print('@@@ Start Tokenizing @@@\n')
    #tstart = time.time()

    #train_doc = [(tonkenize(row[1]), row[2]) for row in tqdm(train_data, ncols = 47, ascii = True, desc = 'tarin_doc')]
    #for r in tqdm(train_doc, ncols = 47, ascii = True, desc = 'tarin_csv'):
    #    w = r[0]
    #    w.append(r[1])
    #    write_csv(Train_Token_File, w)
    #train_doc = []
    #for row in tqdm(train_data, ncols = 47, ascii = True, desc = 'tarin_doc'):
    #    train_doc_temp = tonkenize(row[1])
    #    train_doc.append((train_doc_temp, row[2]))
    #    train_doc_temp.append(row[2])
    #    write_csv("../../train_token.csv", train_doc_temp)
        #util.write_txt(train_doc_temp + ',' + row[2] + '\n', "../../train_token.csv")

    #test_doc = [(tonkenize(row[1]), row[2]) for row in tqdm(test_data, ncols = 47, ascii = True, desc = 'test_doc')]
    #for r in tqdm(test_doc, ncols = 47, ascii = True, desc = 'test_csv'):
    #    w = r[0]
    #    w.append(r[1])
    #    write_csv(Test_Token_File, w)
    #test_doc = []
    #for row in tqdm(test_data, ncols = 47, ascii = True, desc = 'test_doc'):
    #    test_doc_temp = tonkenize(row[1])
    #    test_doc.append((test_doc_temp, row[2]))
    #    test_doc_temp.append(row[2])
    #    write_csv("../../test_token.csv", test_doc_temp)
        #util.write_txt(test_doc_temp + ',' + row[2] + '\n', "../../test_token.csv")


    print('@@@ End Tokenizing @@@\n')
    tend = time.time()
    elapsed = tend - tstart
    print('data packing time : ', elapsed)
    #pprint(train_doc[0])

    return (train_doc, test_doc)

# func def collocations
def procCollocations():
    realPath = os.path.realpath('../../ratings_train.txt')
    measures = collocations.BigramAssocMeasures()
    col_doc = kolaw.open('constitution.txt').read()
    print('\nCollocations among tagged words:')
    tagged_words = Kkma().pos(col_doc)
    finder = collocations.BigramCollocationFinder.from_words(tagged_words)
    pprint(finder.nbest(measures.pmi, 10))

    print('\nCollocations among words:')
    words = [w for w, t in tagged_words]
    ignored_words = [u'안녕']
    finder = collocations.BigramCollocationFinder.from_words(words)
    finder.apply_word_filter(lambda w: len(w) < 2 or w in ignored_words)
    finder.apply_freq_filter(3)
    pprint(finder.nbest(measures.pmi, 10))

    print('\nCollocations among tags:')
    tags = [t for w, t in tagged_words]
    finder = collocations.BigramCollocationFinder.from_words(tags)
    pprint(finder.nbest(measures.pmi, 5))
# func end

# func def sentiment classification with term-existance
def existTerm(d1, d2, text):
    selected_words = [f[0] for f in text.vocab().most_common(2000)]

    def term_exists(doc):
        return {'exists({})'.format(word): (word in set(doc)) for word in selected_words}

    train_docs = d1[:10000]
    test_docs = d2[:10000]

    train_xy = [(term_exists(d), c) for d, c in train_docs]
    test_xy = [(term_exists(d), c) for d, c in test_docs]

    classifier = nltk.NaiveBayesClassifier.train(train_xy)
    print(nltk.classify.accuracy(classifier, test_xy))
    classifier.show_most_informative_features(10)
    print('func done')
# func end

# func sentiment classification with doc2vec (gensim)
def procGensim(d1, d2):
    TaggedDocument = namedtuple('TaggedDocument', 'words tags')
    tagged_train_docs = [TaggedDocument(d, [c]) for d, c in d1]
    tagged_test_docs = [TaggedDocument(d, [c]) for d, c in d2]

    doc_vectorizer = doc2vec.Doc2Vec(size=300, alpha=0.025, min_alpha=0.025, seed=1234)
    doc_vectorizer.build_vocab(tagged_train_docs)

    s_time = time.time()
    for epoch in tqdm(range(10), ncols = 47, ascii = True, desc = 'doc2vec_train'):
        doc_vectorizer.train(tagged_train_docs, total_examples=1, epochs=5)
        doc_vectorizer.alpha -= 0.002
        doc_vectorizer.min_alpha = doc_vectorizer.alpha
    print(time.time() - s_time)

    train_x = [doc_vectorizer.infer_vector(doc.words) for doc in tqdm(tagged_train_docs, ncols = 47, ascii = True, desc = 'tarin_x')]
    train_y = [doc.tags[0] for doc in tqdm(tagged_train_docs, ncols = 47, ascii = True, desc = 'tarin_x')]
    print(len(train_x))
    print(len(train_x[0]))

    test_x = [doc_vectorizer.infer_vector(doc.words) for doc in tqdm(tagged_test_docs, ncols = 47, ascii = True, desc = 'tarin_x')]
    test_y = [doc.tags[0] for doc in tqdm(tagged_test_docs, ncols = 47, ascii = True, desc = 'tarin_x')]
    print(len(test_x))
    print(len(test_x[0]))

    classifier = LogisticRegression(random_state=1234)
    classifier.fit(train_x, train_y)
    print(classifier.score(test_x, test_y))

    while True:
        print("학습 확인을 시작합니다.")
        print("확인하고 싶은 단어를 입력해 주세요")
        print("ex) 공포/Noun")
        print("종료를 원하시면 q 또는 Q 를 입력해 주세요")
        input_data = input("학습 단어를 입력해 주세요 :")
        if input_data == 'q' or input_data == 'Q':
            break
        pprint(doc_vectorizer.most_similar(input_data))
        print('\n')
# func end


if __name__=='__main__':
    train_doc, test_doc =  tokenizing()
    procGensim(train_doc, test_doc)
    tockens = [t for d in train_doc for t in d[0]]
    print(len(tockens))
    text = nltk.Text(tockens, name ='NMSC')
    print(text)
    print(len(text.tokens))
    print(len(set(text.tokens)))
    #existTerm(train_doc, test_doc, text)
    pprint(text.vocab().most_common(10))
    text.plot(50)

    #procCollocations()
