from nltk.corpus import stopwords
from nltk.stem.porter import *
import numpy as np
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

def StemWord(word, stemmer=None):
    if stemmer is None:
        stemmer = PorterStemmer()
    return stemmer.stem(word)

def removeStopWords(data):
    stop = stopwords.words('english')
    for i, row in data.iterrows():
        queryLine = row["query"].lower().split(" ")
        titleLine = row["product_title"].lower().split(" ")
        descriptionLine = row["product_description"].lower().split(" ")
        queryLine = (" ").join([z for z in queryLine if z not in stop])
        titleLine = (" ").join([z for z in titleLine if z not in stop])
        descriptionLine = (" ").join([z for z in descriptionLine if z not in stop])
        data.set_value(i, "query", queryLine)
        data.set_value(i, "product_title", titleLine)
        data.set_value(i, "product_description", descriptionLine)
    # for i in xrange(len(data[])):
    # queryLine = data[i,2].lower().split(" ")
    # titleLine = data[i,3].lower().split(" ")
    #     descriptionLine = data[i,4].lower().split(" ")
    #     queryLine = (" ").join([z for z in queryLine if z not in stop])
    #     titleLine = (" ").join([z for z in titleLine if z not in stop])
    #     descriptionLine = (" ").join([z for z in descriptionLine if z not in stop])
    #     data[i,2] = queryLine
    #     data[i,3] = titleLine
    #     data[i,4] =  descriptionLine
    return data


def nGramSimilarity(s1, s2, n):
    s1 = set(getNgramsWords(s1, n))
    s2 = set(getNgramsWords(s2, n))
    if len(s1.union(s2)) == 0:
        return 0
    else:
        return float(len(s1.intersection(s2))) / float(len(s1.union(s2)))


def getNgramsWords(data, n):
    pattern = re.compile(r"(?u)\b\w+\b")
    wordList = pattern.findall(data)
    Ngrams = []
    if n > len(wordList):
        return []
    for i, word in enumerate(wordList):
        ngram = wordList[i:i + n]
        if len(ngram) == n:
            Ngrams.append(tuple(ngram))
    return Ngrams


def relevancyTuples(group, row, column, ngrams):
    ngrams = range(1, ngrams + 1)
    weightRait = dict()
    for rating in range(1, 5):
        tempDic = dict()
        for ngram in ngrams:
            tempDic[ngram] = [0, 0]
        weightRait[rating] = tempDic
    for i in xrange(len(group)):
        if group[i, 0] != row[0]:
            for ngram in ngrams:
                similarity = nGramSimilarity(row[column], group[i, column], ngram)
                weightRait[group[i, 4]][ngram][1] += similarity
                weightRait[group[i, 4]][ngram][0] += 1
    return weightRait


def extract_features_fromData(data):
    pattern = re.compile(r"(?u)\b\w+\b")
    featurs = np.zeros(shape=(data.shape[0], 9,))
    for i in xrange(len(data)):
        query = set(x.lower() for x in pattern.findall(data[i, 1]))
        title = set(x.lower() for x in pattern.findall(data[i, 2]))
        description = set(x.lower() for x in pattern.findall(data[i, 3]))
        if len(title) > 0:
            featurs[i, 0] = float(len(query.intersection(title))) / float(len(title))
            featurs[i, 1] = float(len(query.intersection(title))) / float(len(query))
        if len(description) > 0:
            featurs[i, 2] = float(len(query.intersection(description))) / float(
                len(description))
            featurs[i, 3] = float(len(query.intersection(description))) / float(
                len(query))
        featurs[i, 4] = len(query)
        featurs[i, 5] = len(title)
        featurs[i, 6] = len(description)
        twoGramsWordsQuery = set(getNgramsWords(data[i, 1], 2))
        twoGramsWordsTitle = set(getNgramsWords(data[i, 2], 2))
        twoGramsWordsDescription = set(getNgramsWords(data[i, 3], 2))
        featurs[i, 7] = len(twoGramsWordsQuery.intersection(twoGramsWordsTitle))
        featurs[i, 8] = len(twoGramsWordsQuery.intersection(twoGramsWordsDescription))
    return featurs