from konlpy.tag import Twitter
from timeit import default_timer as timer
from pprint import pprint
from matplotlib import font_manager, rc
#font_fname = 'c:/windows/fonts/gulim.ttc'
font_fname = "C:/Windows/Fonts/malgun.ttf"
font_name = font_manager.FontProperties(fname=font_fname).get_name()
rc('font', family=font_name)
import nltk

pos_tagger = Twitter()

def read_data(filename):
    with open(filename, 'r', encoding = 'utf-8') as f:
        data = [line.split("\t") for line in f.read().splitlines()]
        data = data[1:] # header extract.
    return data

def tonkenize(doc):
    # norm, stem is optional
    return ['/'.join(t) for t in pos_tagger.pos(doc, norm=True, stem=True)]

def tokenizing():
    # data packing..
    pstart = timer()

    train_data = read_data('../../ratings_train.txt')
    test_data = read_data('../../ratings_test.txt')
    print(len(train_data))
    print(len(train_data[0]))
    print(len(test_data))
    print(len(test_data[0]))

    pend = timer()
    elapsed = pend - pstart
    print('data packing time : ', elapsed)

    # data tokenizing.
    print('@@@ Start Tokenizing @@@')
    tstart = timer()
    train_doc = [(tonkenize(row[1]), row[2]) for row in train_data]
    test_doc = [(tonkenize(row[1]), row[2]) for row in test_data]
    print('@@@ End Tokenizing @@@')
    tend = timer()
    elapsed = tend - tstart
    print('data packing time : ', elapsed)
    pprint(train_doc[0])

    return (train_doc, test_doc)


if __name__=='__main__':
    train_doc, test_doc =  tokenizing()
    tockens = [t for d in train_doc for t in d[0]]
    print(len(tockens))
    text = nltk.Text(tockens, name ='NMSC')
    print(text)
    print(len(text.tokens))
    print(len(set(text.tokens)))
    pprint(text.vocab().most_common(10))
    text.plot(50)



