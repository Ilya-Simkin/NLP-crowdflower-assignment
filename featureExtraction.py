from nltk.stem.porter import *
import numpy as np
from bs4 import BeautifulSoup
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
from featureExtractionMethods import relevancyTuples, extract_features_fromData, StemWord


__author__ = 'darkSide'


def extractBagOfWordsFeatures(train, test):
    traindata = []
    testdata = []
    for i in xrange(len(train)):
        temp = [[], [], []]
        temp[0] = train[i, 1]
        temp[1] = train[i, 2]
        temp[2] = train[i, 3]
        ppp = temp[0] + ' ' + temp[1]  # + ' '+temp[2]
        traindata.append(ppp)
    trainLabels = train[:, 4]
    for i in xrange(len(test)):
        temp = [[], [], []]
        temp[0] = test[i, 1]
        temp[1] = test[i, 2]
        temp[2] = test[i, 3]
        ppp = temp[0] + ' ' + temp[1]  # + ' '+temp[2]
        testdata.append(ppp)
    if len(test[0]) >= 5:
        testLabels = test[:, 4]
    else:
        testLabels = []
    return (traindata, trainLabels, testdata, testLabels)


def extractBagOfWordsWithStemming(train, test):
    stemmer = PorterStemmer()
    trainData = []
    trainLabels = []
    testdata = []
    testLabels = []
    for i in xrange(len(train)):
        tempValueQuery = BeautifulSoup(train[i, 1]).get_text(" ")
        tempValueTitle = BeautifulSoup(train[i, 2]).get_text(" ")
        tempValueDescription = BeautifulSoup(train[i, 3]).get_text(" ")
        s1 = re.sub("[^a-zA-Z0-9]", " ", tempValueQuery).split(" ")
        s2 = re.sub("[^a-zA-Z0-9]", " ", tempValueTitle).split(" ")
        s3 = re.sub("[^a-zA-Z0-9]", " ", tempValueDescription).split(" ")
        tempValueQuery = []
        tempValueTitle = []
        tempValueDescription = []
        for word in s1:
            tempValueQuery.append(StemWord(word, stemmer))
        for word in s2:
            tempValueTitle.append(StemWord(word, stemmer))
        for word in s3:
            tempValueDescription.append(StemWord(word, stemmer))
        ans = [[], [], []]
        ans[0] = tempValueQuery
        ans[1] = tempValueTitle
        ans[2] = tempValueDescription
        ppp = (" ").join(ans[0]) + ' ' + (" ").join(ans[1]) + ' ' + (" ").join(ans[2])
        trainData.append(ppp)
    trainLabels.append(train[:, 4])
    for i in xrange(len(test)):
        tempValueQuery = BeautifulSoup(test[i, 1]).get_text(" ")
        tempValueTitle = BeautifulSoup(test[i, 2]).get_text(" ")
        tempValueDescription = BeautifulSoup(test[i, 3]).get_text(" ")
        s1 = re.sub("[^a-zA-Z0-9]", " ", tempValueQuery).split(" ")
        s2 = re.sub("[^a-zA-Z0-9]", " ", tempValueTitle).split(" ")
        s3 = re.sub("[^a-zA-Z0-9]", " ", tempValueDescription).split(" ")
        tempValueQuery = []
        tempValueTitle = []
        tempValueDescription = []
        for word in s1:
            tempValueQuery.append(StemWord(word, stemmer))
        for word in s2:
            tempValueTitle.append(StemWord(word, stemmer))
        for word in s3:
            tempValueDescription.append(StemWord(word, stemmer))
        ans = [[], [], []]
        ans[0] = tempValueQuery
        ans[1] = tempValueTitle
        ans[2] = tempValueDescription
        ppp = (" ").join(ans[0]) + ' ' + (" ").join(ans[1]) + ' ' + (" ").join(ans[2])
        testdata.append(ppp)
    if len(test[0]) >= 5:
        testLabels.append(test[:, 4])
    else:
        testLabels = []
    return (trainData, trainLabels, testdata, testLabels)


def extract_Complex_Features(train, test):
    ngramss = 2
    featursTrain = np.zeros(shape=(train.shape[0], 3 + 2 * ngramss * 4,))
    featursTest = np.zeros(shape=(test.shape[0], 3 + 2 * ngramss * 4,))
    for i in range(len(train)):
        group = train[train[i, 1] == train[:, 1]]
        q_mean = group[:, 4].mean()
        featursTrain[i, 0] = q_mean
        testGroup = np.where(test[:, 1] == train[i, 1])
        q_median = np.median(group[:, 4])
        featursTrain[i, 1] = q_median
        avg_variance = group[:, 5].mean()
        featursTrain[i, 2] = avg_variance
        featursTest[testGroup[0], 0] = q_mean
        featursTest[testGroup[0], 1] = q_median
        featursTest[testGroup[0], 2] = avg_variance
        for n in range(2, 4):
            weights = relevancyTuples(group, train[i], n, ngramss)
            for rating in weights:
                for ngram in weights[rating]:
                    if weights[rating][ngram][0] != 0:
                        k = (rating - 1 ) % 4
                        l = 4 * (ngram - 1 )
                        t = (n - 2) * 8
                        featursTrain[i, 3 + (k + l + t)] = float(weights[rating][ngram][1]) / float(
                            weights[rating][ngram][0])
                        # else:
                        # print variable_name +" 0"
                        # pass
    for i in range(len(test)):
        group = train[test[i, 1] == train[:, 1]]
        for n in range(2, 4):
            weights = relevancyTuples(group, test[i], n, ngramss)
            for rating in weights:
                for ngram in weights[rating]:
                    if weights[rating][ngram][0] != 0:
                        k = (rating - 1 ) % 4
                        l = 4 * (ngram - 1 )
                        t = (n - 2) * 8
                        featursTest[i, 3 + (k + l + t)] = float(weights[rating][ngram][1]) / float(
                            weights[rating][ngram][0])
                        # else:
                        # # print variable_name +" 0"
                        # pass
    return featursTrain, featursTest


def extract_featurs(train, test):
    return extract_features_fromData(train), extract_features_fromData(test)


def extract(train, test):
    trainFeaturs, testFeaturs = extract_featurs(train, test)
    featureTrain, featuresTest = extract_Complex_Features(train, test)
    trainlbl = train[:, 4]
    if len(test[0]) > 5:
        testlbl = test[:, 4]
    else:
        testlbl = []
    train = np.concatenate((trainFeaturs, featureTrain), axis=1)
    test = np.concatenate((testFeaturs, featuresTest), axis=1)
    return train, trainlbl, test, testlbl

