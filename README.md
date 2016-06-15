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
* first lets show how we handeleld the data input from the train and the test files :
  its is a simple csv read:
```{r load_data, message=FALSE, results='hide'}
def loadData(trainPath, testPath):
    train = pd.read_csv(trainPath).fillna("")
    test = pd.read_csv(testPath).fillna("")
    return train, test
    ```
  
###part two : the feature extraction:
