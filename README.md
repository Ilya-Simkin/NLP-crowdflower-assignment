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

###part two : the feature extraction:


