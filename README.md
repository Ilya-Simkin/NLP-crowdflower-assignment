# NLP-crowdflower-assignment
an assignment in a course to develop an NLP based search relevance estimation by machine learning ensemble approache

###An home work assignment in a course to develop an NLP based search relevance estimation by machine learning ensemble approache

###introdaction
* this report will explain the work that was done by use to get the best we can in the kaggle CrowdFlower compatition 

####the Code itself is in the files in this directory 

###the authors of this work are : 
* Ilya Simkin, id : 305828188
* Or May-Paz, id : 301804134

*we will split the work we done into three main phases :
*   the data input handaling 
*   the feature extraction 
*   and the model creation phase 

## data input handaling :
first lets show how we handeleld the data input from the train and the test files :
its is a simple csv read:
```{r load_data, message=FALSE, results='hide'}
def loadData(trainPath, testPath):
    train = pd.read_csv(trainPath).fillna("")
    test = pd.read_csv(testPath).fillna("")
    return train, test
```

this part create the K-fold of the data we chose to use 5 as a k but we tested with 10 as well just didnt got better result and that was faster so ...
```{r kfold, message=FALSE, results='hide'}
def create_KfoldSets(dataSet, n_folds=5):
    splitSet = []
    if n_folds == 1:
        indexs = range(len(dataSet["query"]))
        random.shuffle(indexs)
        splitPoint = int(len(dataSet["query"]) * 0.8)
        trainData = dataSet.loc[indexs[:splitPoint]].as_matrix()
        testData = dataSet.loc[indexs[splitPoint:]].as_matrix()
        splitSet.append((trainData, testData))
    else:
        kfolds = SKF(dataSet["query"], n_folds=n_folds)
        for trainIndex, testIndex in kfolds:
            trainData = dataSet.loc[trainIndex].as_matrix()
            testData = dataSet.loc[testIndex].as_matrix()
            splitSet.append((trainData, testData))
    return splitSet
```

*this is the "main", it is the driver of all the feature extraction that we explain in the moment and creation of files of features that are ready for the machine learning models we made :
```{r main, message=FALSE, results='hide'}

def main(removeStopWordss=True):
    path = './pkls/'
    train, test = loadData("./dataSamples/train.csv", "./dataSamples/test.csv")
    if removeStopWordss:
        train = removeStopWords(train)
        test = removeStopWords(test)
    splitTrainSet = create_KfoldSets(train, n_folds=5)
    bagOfWords1 = []
    bagOfWordsWithStamming = []
    featuresSets = []
    for X_train, X_test in splitTrainSet:
        bowFeatures1 = extractBagOfWordsFeatures(X_train, X_test)
        bagOfWords1.append(bowFeatures1)
        bowFeatures2 = extractBagOfWordsWithStemming(X_train, X_test)
        bagOfWordsWithStamming.append(bowFeatures2)
        trainset, trainlbl, testset, testlbl = extract(X_train, X_test)
        newTrainSet = np.zeros(shape=(trainset.shape[0], trainset.shape[1] + 1,), dtype=float)
        for i in range(len(trainset)):
            newTrainSet[i, 1:] = trainset[i]
            newTrainSet[i, 0] = X_train[i, 0]
        newtestSet = np.zeros(shape=(testset.shape[0], testset.shape[1] + 1,), dtype=float)
        for i in range(len(testset)):
            newtestSet[i, 1:] = testset[i]
            newtestSet[i, 0] = X_test[i, 0]
        trainset = newTrainSet
        testset = newtestSet
        featuresSets.append((trainset, trainlbl, testset, testlbl))
    pickle.dump(featuresSets, open(path + 'featuresSetKfold.pkl', 'w'))
    pickle.dump(bagOfWords1, open(path + 'bagOfWords1Kfold.pkl', 'w'))
    pickle.dump(bagOfWordsWithStamming, open(path + 'bagOfWordsWithStammingKfold.pkl', 'w'))
    indexs = range(len(train["query"]))
    trainData = train.loc[indexs[:]].as_matrix()
    indexs = range(len(test["query"]))
    testData = test.loc[indexs[:]].as_matrix()
    bowFeatures1 = extractBagOfWordsFeatures(trainData, testData)
    pickle.dump(bowFeatures1, open(path + 'bagOfWords1fullDataset.pkl', 'w'))
    bowFeatures2 = extractBagOfWordsWithStemming(trainData, testData)
    pickle.dump(bowFeatures2, open(path + 'bagOfWordsWithStammingFullDataSet.pkl', 'w'))
    trainset, trainlbl, testset, testlbl = extract(trainData, testData)
    idTrain = train["id"].as_matrix()
    idTest = test["id"].as_matrix()
    newTrainSet = np.zeros(shape=(trainset.shape[0], trainset.shape[1] + 1,), dtype=float)
    for i in range(len(trainset)):
        newTrainSet[i, 1:] = trainset[i]
        newTrainSet[i, 0] = idTrain[i]
    newtestSet = np.zeros(shape=(testset.shape[0], testset.shape[1] + 1,), dtype=float)
    for i in range(len(testset)):
        newtestSet[i, 1:] = testset[i]
        newtestSet[i, 0] = idTest[i]
    trainset = newTrainSet
    testset = newtestSet
    pickle.dump((trainset, trainlbl, testset, testlbl), open(path + 'fullDataSetFeatures.pkl', 'w'))
```
#### but first some extra helpfull functions :
* the stemmer function using the nltk package to stemm one word at the time:
```{r stemmer, message=FALSE, results='hide'}
def StemWord(word, stemmer=None):
    if stemmer is None:
        stemmer = PorterStemmer()
    return stemmer.stem(word)
```
*the remove stop words function, it also use the nltk package to remove stop word; those are words that wont help us as data extraction because they are to often used in the english language.
```{r stopwords, message=FALSE, results='hide'}
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
```
* now we display the ngrams creation functions they create the ngrams that are all the combinations of n words (we use 1 and 2 ) in different parts as the query and as in the title and the artical in the data set :
```{r ngrams, message=FALSE, results='hide'}

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

```
###part two : the feature extraction:

####the feature extraction will be splet to 3 parts the baisic featurs extraction that extract the normal featurs from the data:
* some samples of features we get in this part are :
 1. query tokens in title
 2. percent query tokens in the title
 3. query tokens inthe description
 4. the query length
 5. the description and title length
 6. after that we use the n grams we made and check how many of them from the query where in the title and vice versa.
 7. and so on ...

the code of the baisice feature extraction : 
```{r baisicFextraction, message=FALSE, results='hide'}
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
```
* the complex features extrction in here we extract the more statistic features most of them need some extra work on all the data set to work here we extract featurs like relevance of ngrams and theyr mean and variance the similarity function itself in the feature extraction file and wont be presented here .
* the function recive the train and test sets as for us it is each of the train and the test sets created by the kfold methods. and also once for the full train and test sets for the final feature extraction that we will use in the end.
```
def extract_Complex_Features(train, test):
    ngramss = 2 # here we chose to use up to 2 grams sets because 3 was a little havey and gave little results improvment,
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
        for n in range(2, 4): # once the feature extracted for the title and once for the article body against the title 
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
        for n in range(2, 4): # once the feature extracted for the title and once for the article body against the title 
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
```



