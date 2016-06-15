import pickle
import itertools
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer


def KappaOnCrossValidation(model, kfoldData, features):
    scoreCount, scoreTotal = 0.0, 0
    testData = []
    for trainFeatures, lblsTrain, testFeatures, lblsTest in kfoldData:
        trainFeatures = trainFeatures[:, features]
        testFeatures = testFeatures[:, features]
        lblsTrain = lblsTrain.astype(int)
        lblsTest = lblsTest.astype(int)
        model.fit(trainFeatures, lblsTrain)
        predictions = model.predict(testFeatures)
        scoreCount += 1
        score = weightedKappa(trueLabel=lblsTest, predictedLabel=predictions)
        scoreTotal += score
        print("Score " + str(scoreCount) + ": " + str(score))
        y_and_lblPredictions = pd.DataFrame({'y': lblsTest, 'lblPredictions': predictions})
        testData.append(y_and_lblPredictions)
    average_score = scoreTotal / float(scoreCount)
    print("avg score: " + str(average_score))
    return testData


def tfidfCrossValidation(tfv, pipeline, kfoldData):
    score_count, score_total = 0, 0.0
    test_data = []
    for featuresTrain, lblsTrain, featuresTest, lblsTest in kfoldData:
        tfv.fit(featuresTrain)
        featuresTrain = tfv.transform(featuresTrain)
        featuresTest = tfv.transform(featuresTest)
        if isinstance(lblsTrain, list):
            lblsTrain = lblsTrain[0]
        lblsTrain = lblsTrain.astype(int)
        pipeline.fit(featuresTrain, lblsTrain)
        predictions = pipeline.predict(featuresTest)
        score_count += 1
        if isinstance(lblsTest, list):
            lblsTest = lblsTest[0]
        lblsTest = lblsTest.astype(int)
        score = weightedKappa(lblsTest, predictions)
        score_total += score
        print("Score " + str(score_count) + ": " + str(score))
        y_and_lblPredictions = pd.DataFrame({'y': lblsTest, 'lblPredictions': predictions})
        test_data.append(y_and_lblPredictions)
    average_score = score_total / float(score_count)
    print("avg score: " + str(average_score))
    return test_data


def ActivateModelAndformatOutput(model, train, test, features):
    trainFeatures, y_set = train
    trainset = trainFeatures[:, features]
    trainlbl = y_set.astype(int)
    testSet = test[0][:, features]
    model.fit(trainset, trainlbl)
    predictions = model.predict(testSet)
    ids = test[0][:, 0]
    ids = ids.astype(int)
    submission = pd.DataFrame({"id": ids, "prediction": predictions})
    return submission


def trainRF(train, test, KfoldDataSet, features=None):
    if features is None:
        features = range(KfoldDataSet[0][0])
    model = RandomForestClassifier(n_estimators=300, n_jobs=-1, min_samples_split=10, random_state=1,
                                   class_weight='auto')
    RFcrossValidationTest = None
    if toTestModel:
        RFcrossValidationTest = KappaOnCrossValidation(model, KfoldDataSet, features)
    rf_final_predictions = ActivateModelAndformatOutput(model, train, test, features)
    return RFcrossValidationTest, rf_final_predictions


def trainSVC(train, test, KfoldDataSet, features=None):
    if features is None:
        features = range(KfoldDataSet[0][0])
    scl = StandardScaler()
    svm_model = SVC(random_state=1, class_weight={1: 2, 2: 1.5, 3: 1, 4: 1})
    model = Pipeline([('scl', scl), ('svm', svm_model)])
    svcCrossValidationTest = None
    if toTestModel:
        svcCrossValidationTest = KappaOnCrossValidation(model, KfoldDataSet, features)
    svc_final_predictions = ActivateModelAndformatOutput(model, train, test, features)
    return svcCrossValidationTest, svc_final_predictions


def trainAdaBoost(train, test, KfoldDataSet, features=None):
    if features is None:
        features = range(KfoldDataSet[0][0])
    model = AdaBoostClassifier(n_estimators=200, learning_rate=0.5)
    adaBoostCrossValidationTest = None
    if toTestModel:
        adaBoostCrossValidationTest = KappaOnCrossValidation(model, KfoldDataSet, features)
    adaboost_final_predictions = ActivateModelAndformatOutput(model, train, test, features)
    return adaBoostCrossValidationTest, adaboost_final_predictions


def Baggingclassifiers(train, test, KfoldDataSet, features=None):
    if features is None:
        features = range(KfoldDataSet[0][0])
    model = BaggingClassifier(n_estimators=200, n_jobs=-1, random_state=1)  # random_state=1,
    baggingCrossValidationTest = None
    if toTestModel:
        baggingCrossValidationTest = KappaOnCrossValidation(model, KfoldDataSet, features)
    baggingFinalPredictions = ActivateModelAndformatOutput(model, train, test, features)
    return baggingCrossValidationTest, baggingFinalPredictions


def trainTFIDF1(bow1features, bow1kfold, test):
    # if features is None:
    # features = range(KfoldDataSet[0][0])
    idx = (test[0][:, 0]).astype(int)
    trainset1, y_v1, testset1, testlbl_v1 = bow1features
    tfv = TfidfVectorizer(min_df=3, max_features=None, strip_accents='ascii', analyzer='word', token_pattern=r'\w{1,}',
                          ngram_range=(1, 2), use_idf=1, smooth_idf=1, sublinear_tf=1, stop_words='english')
    pipeline = Pipeline([('svd', TruncatedSVD(n_components=400)), ('scl', StandardScaler()), ('svm', SVC(C=10))])
    tfidf1CrossValidationTest = None
    if toTestModel:
        tfidf1CrossValidationTest = tfidfCrossValidation(tfv, pipeline, bow1kfold)
    tfv.fit(trainset1)
    X_train = tfv.transform(trainset1)
    X_test = tfv.transform(testset1)
    y_v1 = (y_v1.astype(int))
    pipeline.fit(X_train, y_v1)
    predictions = pipeline.predict(X_test)
    finalResults = pd.DataFrame({"id": idx, "prediction": predictions})
    return tfidf1CrossValidationTest, finalResults


def trainTFIDF2(bow21features, bow2kfold, test):
    idx = (test[0][:, 0]).astype(int)
    tfv = TfidfVectorizer(min_df=5, max_df=500, max_features=None, strip_accents='ascii', analyzer='word',
                          token_pattern=r'\w{1,}', ngram_range=(1, 2), use_idf=True, smooth_idf=True, sublinear_tf=True,
                          stop_words='english')
    pipeline = Pipeline(
        [('svd', TruncatedSVD(n_components=200, algorithm='randomized', n_iter=5, random_state=None, tol=0.0)),
         ('scl', StandardScaler(copy=True, with_mean=True, with_std=True)),
         ('svm',
          SVC(C=10.0, kernel='rbf', degree=3, gamma='auto', coef0=0.0, shrinking=True, probability=False, tol=0.001,
              cache_size=200, class_weight=None, verbose=False, max_iter=-1, random_state=None))])
    tfidf2CrossValidationTest = None
    if toTestModel:
        tfidf2CrossValidationTest = tfidfCrossValidation(tfv, pipeline, bow2kfold)
    trainData, lblsTrain, testData, lblstest = bow21features
    tfv.fit(trainData)
    X_train = tfv.transform(trainData)
    X_test = tfv.transform(testData)
    if isinstance(lblsTrain, list):
        lblsTrain = lblsTrain[0]
    lblsTrain = (lblsTrain.astype(int))
    pipeline.fit(X_train, lblsTrain)
    predictions = pipeline.predict(X_test)
    finalResults = pd.DataFrame({"id": idx, "prediction": predictions})
    return tfidf2CrossValidationTest, finalResults


def weightedKappa(trueLabel, predictedLabel):
    rater_a = trueLabel
    rater_b = predictedLabel
    min_rating = None
    max_rating = None
    rater_a = np.array(rater_a, dtype=int)
    rater_b = np.array(rater_b, dtype=int)
    if not (len(rater_a) == len(rater_b)):
        raise Exception("prediction and test set labels not of the same length")
    if min_rating is None:
        min_rating = min(min(rater_a), min(rater_b))
    if max_rating is None:
        max_rating = max(max(rater_a), max(rater_b))
    assert (len(rater_a) == len(rater_b))
    if min_rating is None:
        min_rating = min(rater_a + rater_b)
    if max_rating is None:
        max_rating = max(rater_a + rater_b)
    num_ratings = int(max_rating - min_rating + 1)
    conf_mat = [[0 for i in range(num_ratings)]
                for j in range(num_ratings)]
    for a, b in zip(rater_a, rater_b):
        conf_mat[a - min_rating][b - min_rating] += 1
    num_ratings = len(conf_mat)
    scoredItem = float(len(rater_a))
    hist_rater_a = histogram(rater_a, min_rating, max_rating)
    hist_rater_b = histogram(rater_b, min_rating, max_rating)
    numerator = 0.0
    denominator = 0.0
    for i in range(num_ratings):
        for j in range(num_ratings):
            expected_count = (hist_rater_a[i] * hist_rater_b[j] / scoredItem)
            d = pow(i - j, 2.0) / pow(num_ratings - 1, 2.0)
            numerator += d * conf_mat[i][j] / scoredItem
            denominator += d * expected_count / scoredItem
    return (1.0 - numerator / denominator)


def histogram(ratings, min_rating=None, max_rating=None):
    if min_rating is None:
        min_rating = min(ratings)
    if max_rating is None:
        max_rating = max(ratings)
    num_ratings = int(max_rating - min_rating + 1)
    hist_ratings = [0 for x in range(num_ratings)]
    for r in ratings:
        hist_ratings[r - min_rating] += 1
    return hist_ratings


def main():
    global toTestModel
    global path
    saveFiles = True
    checkWeights = True
    toTestModel = True
    path = './pkls/'
    data = pickle.load(open(path + 'fullDataSetFeatures.pkl', 'r'))
    train = (data[0], data[1])  # features and lables for train set
    test = (data[2], data[3])  # features and lables for test set
    kfoldDataSet = pickle.load(open(path + 'featuresSetKfold.pkl', 'r'))
    bow1features = pickle.load(open(path + 'bagOfWords1fullDataset.pkl', 'r'))
    bow21features = pickle.load(open(path + 'bagOfWordsWithStammingFullDataSet.pkl', 'r'))
    bow1kfold = pickle.load(open(path + 'bagOfWords1Kfold.pkl', 'r'))
    bow2kfold = pickle.load(open(path + 'bagOfWordsWithStammingKfold.pkl', 'r'))
    trainset, trainlbl, testset, testlbl = pickle.load(open(path + 'fullDataSetFeatures.pkl', 'r'))

    features = range(1, 29)
    if saveFiles:
        CVTestRF, rfPredictions = trainRF(train, test, kfoldDataSet, features)
        CVTestSvc, svcPredictions = trainSVC(train, test, kfoldDataSet, features)
        CVTestAB, adaboostPredictions = trainAdaBoost(train, test, kfoldDataSet, features)
        CVTestBagging, baggingPredictions = Baggingclassifiers(train, test, kfoldDataSet, features)
        CVTestTFIDF1, tfidf1Predictions = trainTFIDF1(bow1features, bow1kfold, test)
        CVTestTFIDF2, tfidf2Predictions = trainTFIDF2(bow21features, bow2kfold, test)
        fataToSave = (( CVTestRF, rfPredictions),
                      (CVTestSvc, svcPredictions),
                      ( CVTestAB, adaboostPredictions),
                      (CVTestBagging, baggingPredictions),
                      (CVTestTFIDF1, tfidf1Predictions),
                      (  CVTestTFIDF2, tfidf2Predictions))
        pickle.dump(fataToSave, open(path + 'allSets.pkl', 'w'))
    else:

        (( CVTestRF, rfPredictions),
         (CVTestSvc, svcPredictions),
         ( CVTestAB, adaboostPredictions),
         (CVTestBagging, baggingPredictions),
         (CVTestTFIDF1, tfidf1Predictions),
         (  CVTestTFIDF2, tfidf2Predictions)) = pickle.load(open(path + 'allSets.pkl', 'r'))
    preds = [CVTestRF,
             CVTestSvc,
             CVTestAB,
             CVTestBagging,
             CVTestTFIDF1,
             CVTestTFIDF2]
    weights = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.55, 0.6, 0.65]
    if checkWeights:
        weightsCombinations = []
        weightsCombinations.extend(itertools.product(weights, repeat=6))
        goodWeighting = []
        for w in weightsCombinations:
            if sum(w) == 1.0:
                goodWeighting.append(w)
        max_average_score = 0
        max_weights = None
        for weights in goodWeighting:
            total_score = 0
            for i in range(5):
                trueLbls = preds[0][i]['y']
                # tt = preds[i].values[:,0]
                # y_true = preds[i].values[:,0]
                tempArr = []
                for x in range(6):
                    tempArr.append(weights[x] * preds[x][i]['lblPredictions'].astype(int).reset_index())
                weighted_prediction = sum(tempArr)
                # pp = weighted_prediction.as_matrix()
                weighted_prediction = [round(p) for p in weighted_prediction['lblPredictions']]
                total_score += weightedKappa(trueLbls, weighted_prediction)
            average_score = total_score / 5.0
            if average_score > max_average_score:
                max_average_score = average_score
                max_weights = weights
        print "Best weights: " + str(max_weights)
        print "final score: " + str(max_average_score)
    else:
        max_weights = (0.4, 0.15, 0.05, 0.1, 0.15, 0.15)
    preds = [rfPredictions, svcPredictions, adaboostPredictions, baggingPredictions, tfidf1Predictions,
             tfidf2Predictions]
    weighted_prediction = sum([max_weights[x] * preds[x]["prediction"].astype(int) for x in range(6)])
    weighted_prediction = [int(round(p)) for p in weighted_prediction]
    submission = pd.DataFrame({"id": testset[:, 0].astype(int), "prediction": weighted_prediction})
    submission.to_csv('ensembledSubmission.csv', index=False)


if __name__ == '__main__':
    main()
