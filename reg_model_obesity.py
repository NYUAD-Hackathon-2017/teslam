import csv
import numpy as np
from sklearn import datasets, linear_model
from sklearn.linear_model import LassoCV
from sklearn.linear_model import Lasso
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
import random
import sys
import math

from sklearn import preprocessing
import matplotlib.pyplot as plt

states = {
        'US-AK': 'Alaska',
        'US-AL': 'Alabama',
        'US-AR': 'Arkansas',
        'US-AS': 'American Samoa',
        'US-AZ': 'Arizona',
        'US-CA': 'California',
        'US-CO': 'Colorado',
        'US-CT': 'Connecticut',
        'US-DC': 'District of Columbia',
        'US-DE': 'Delaware',
        'US-FL': 'Florida',
        'US-GA': 'Georgia',
        #'US-GU': 'Guam',
        #'US-HI': 'Hawaii',
        'US-IA': 'Iowa',
        'US-ID': 'Idaho',
        'US-IL': 'Illinois',
        'US-IN': 'Indiana',
        'US-KS': 'Kansas',
        'US-KY': 'Kentucky',
        'US-LA': 'Louisiana',
        'US-MA': 'Massachusetts',
        'US-MD': 'Maryland',
        'US-ME': 'Maine',
        'US-MI': 'Michigan',
        'US-MN': 'Minnesota',
        'US-MO': 'Missouri',
        #'US-MP': 'Northern Mariana Islands',
        'US-MS': 'Mississippi',
        'US-MT': 'Montana',
        #'US-NA': 'National',
        'US-NC': 'North Carolina',
        'US-ND': 'North Dakota',
        'US-NE': 'Nebraska',
        'US-NH': 'New Hampshire',
        'US-NJ': 'New Jersey',
        'US-NM': 'New Mexico',
        'US-NV': 'Nevada',
        'US-NY': 'New York',
        'US-OH': 'Ohio',
        'US-OK': 'Oklahoma',
        'US-OR': 'Oregon',
        'US-PA': 'Pennsylvania',
        #'US-PR': 'Puerto Rico',
        'US-RI': 'Rhode Island',
        'US-SC': 'South Carolina',
        'US-SD': 'South Dakota',
        'US-TN': 'Tennessee',
        'US-TX': 'Texas',
        'US-UT': 'Utah',
        'US-VA': 'Virginia',
        'US-VI': 'Virgin Islands',
        'US-VT': 'Vermont',
        'US-WA': 'Washington',
        'US-WI': 'Wisconsin',
        'US-WV': 'West Virginia',
        'US-WY': 'Wyoming'
}

'''
This function is used to read Y (true) values for the given disease
Required: File Location, nFlag: telling the function if y values should be normalized
Returns: List of (NORMALIZED if nFlag = True) y values in terms of tuples : (state,yvalue)
'''

def readY(filename,nFlag):
    fin = open(filename, 'rb')
    reader = csv.reader(fin)
    counter = 0
    #Gotten these values so that I can only focus on these
    stateNames = states.values()
    stateYtuples = []
    #Let's just first read the values in a tuple i.e [(state,y),(state,y)..]
    for row in reader:
        if(counter > 2):
            state = str(row[0])
            #ensure we are only taking the states we have available
            if(state in stateNames):
                if (str(row[1]) != 'No Data'):
                    yValue = float(row[1])
                    stateYtuples.append((state,yValue))
                else:
                    yValue = 0.0
                    stateYtuples.append((state,yValue))
        counter+=1
    #sort all by states in alphabetical roder
    sortedTuples = sorted(stateYtuples,key=lambda x:x[0])

    ############## Normalization Of Y's #############################
    if(nFlag == True):   
        maxPrevalence = max(p for (_,p) in sortedTuples)
        sortedTuples = [(s,float(p)*100/maxPrevalence) for (s,p) in sortedTuples]
    return sortedTuples

def readX(filename):
    #print filename
    fin = open(filename, 'rb')
    reader = csv.reader(fin)
    counter = 0
    #Gotten these values so we can focus only on these states
    stateNames = states.values()
    stateXtuples = []
    #Let's just first read the values in a tuple i.e [(state,x),(state,x)..]
    for row in reader:
        if(counter > 2):
            state = str(row[0])
            #ensure we are only taking the states we have available
            if(state in stateNames):
                xValue = float(row[1])
                stateXtuples.append((state,xValue))
        counter+=1
    #sort all by states in alphabetical roder
    sortedTuples = sorted(stateXtuples,key=lambda x:x[0])
    #print sortedTuples
    #sys.exit()
    return sortedTuples

def firstReg(Xs,Ys,trainYears,testYears,yearTrainBegin):
    train = []
    test = []
    print len(Ys)
    #Let's first deal with Ys
    '''
    Ys are of the form [[(s1,v1),(s2,v2),(s3,v3)],
                        [(s1,v1'),(s2,v2'),(s3,v3')],...]
    What we want is just the values [[v1],[v2],[v3],[v1'],[v2'],...]
    '''
    YtrainTup = []
    YtestTup = []
    for i in range(len(Ys)):
        YForThisYear = Ys[i]
        if ((yearTrainBegin+i) in trainYears):
            YtrainTup.append(YForThisYear)
        else:
            YtestTup.append(YForThisYear)
    
    #Now let us format Ys so they can be used in our reg model
    yTrain = [[i[1]] for i in [item for sublist in YtrainTup for item in sublist]]
    yTest =  [[i[1]] for i in [item for sublist in YtestTup for item in sublist]]

    #Let us now deal with the Xs
    '''
    Xs are of the form [
                       [[(s1,v1diabetic),(s2,v2diabetic),(s3,v3diabetic)],
                       [(s1,v1Obesity),(s2,v2Obesity),(s3,v3Obesity)],...],
                       [2012],[2013],...]
    What we want is just the values [[v1,v2,v3],...[v1',v2',v3'],...] where
    v1 represents diabetic,obesity,hypertension for state 1 for the year 2011
    and v1' represents diabetic,obesity,hypertension for the state 1 for 2012,
    and so on
    '''
    
    XtrainTup = []
    XtestTup = []
    for i in range(len(Xs)):
        XForThisYear = Xs[i]
        if ((yearTrainBegin+i) in trainYears):
            XtrainTup.append(XForThisYear)
        else:
            XtestTup.append(XForThisYear)
    #Now let us format the Xs do they can be used in reg model
    xTrain = []#[[]] * len(yTrain) 
    xTest = []#[[]] * len(yTest)
    ####################
    #This code is dependent on the number of states
    ###########################################
    numOfStates = 50
    for yearNumber in range(len(XtrainTup)):
        yearLst = XtrainTup[yearNumber]
        #each list in YearLst is 50 states long
        
        for i in range(numOfStates):
            rowFeature = []
            #Go through 50 states
            for keywordIndex in range(len(yearLst)):
                #print yearLst[keywordIndex][i]
                #print yearLst[keywordIndex][i]
                #sys.exit()
                rowFeature.append(yearLst[keywordIndex][i][1])
                
            xTrain.append(rowFeature)
            
    for yearNumber in range(len(XtestTup)):
        yearLst = XtestTup[yearNumber]
        #each list in YearLst is 50 states long
        
        for i in range(numOfStates):
            rowFeature = []
            #Go through 50 states
            for keywordIndex in range(len(yearLst)):
                #print yearLst[keywordIndex][i]
                rowFeature.append(yearLst[keywordIndex][i][1])
                
            xTest.append(rowFeature)
        
    train.append(xTrain)
    train.append(yTrain)
    test.append(xTest)
    test.append(yTest)
    return [train,test]

#This function will return the score we picked corresponding to the alpha we want to pick
def pickAlpha(scores):
    #cloning the array
    tempScores = scores[:]
    tempScores.sort()
    optimum = tempScores[0]
    #Now we need to find some score upto 10% of the optimum
    rangeOfAlpha = 5 #%
    found = False
    index = 0
    while(not found):
        print index
        currentScore = tempScores[index]
        errorPercent = ((currentScore * 100.0/optimum) - 100)
        print "error percent",str(errorPercent)
        if(errorPercent > rangeOfAlpha):
            found = True
            return tempScores[index-1]
        else:
            index+=1

'''
Requires: A list of tuples of the form (alpha,scoresLst)
          where scoresLst is the list of scores gotten by
          using the alpha for n-fold cross-validation, n
          being the length of the scoresLst
          scores in scoresLst should be all positive
Returns:  optimal alpha value 1 std deviation away
Link for reference:
                    https://en.wikipedia.org/wiki/
                    Unbiased_estimation_of_standard_deviation
'''

def pickAlpha(alphasAndScores):
    #How many stds away?
    away = 1
    #Lets go through each alpha and find the std dev
    std_devs = []  #For each alpha, std devs
    avgScores = [] #For each alpha, sum(scores)/n
    alphas = [alpha for (alpha,_) in alphasAndScores]
    for (alpha,scoresLst) in alphasAndScores:
        n = len(scoresLst)
        xHat = sum(scoresLst)*1.0/n
        summation = 0
        #Go through each score and find its squared distance from mean
        for x in scoresLst:
            summation += pow(xHat - x,2)
        #Now that we have the summation lets divide it by n-1
        std = math.sqrt(1.0/(n-1) * summation)
        std_devs.append(std)
        avgScores.append(xHat)
    #Note all the lists have 1-1 correspondence in terms of indices
    tempAvgScores = avgScores[:]
    tempAvgScores.sort()
    #Now optimal avg score should be on index 0
    optimumScore = tempAvgScores[0]
    #Now I need to find which index does this optimum Score belong to
    optimumIndex = avgScores.index(optimumScore)

    #########################################
    return alphas[optimumIndex]
    #########################################
    
    #We can have deviation 1 std-away from optimum
    acceptableScore = optimumScore + away*std_devs[optimumIndex]
    #print(optimumScore)
    #print(acceptableScore)
    #print(tempAvgScores[-1])
    #print (alphasAndScores[optimumIndex])
    #Now I need to move from optimumScore such that I find a value
    #which is greater than the acceptableScore
    found = False
    nearestScore = optimumScore #nearest to the acceptable
    i = 1 #skipping over the optimum obviously otherwise we would have picked it
    while(not found):
        if(acceptableScore < tempAvgScores[i]):
            nearestScore = tempAvgScores[i-1]
            found = True
        i+=1
    
    #Now we need to find the actual Index of the nearestScore
    alphaIndex = avgScores.index(nearestScore)
    print "optimal Alpha",str(alphas[optimumIndex])
    #return alphas[alphaIndex]
    return alphas[alphaIndex]

def main():
    #Let user write the name of the file with the keywords we want to read
    #filename = raw_input("Type location of the file to read keywords:")
    
    #############################
    filename = "Data/Middle East/Arabic/related_keywords_for_USA.txt"
    #print(filename)
    #############################

    #open the file to read all the keywords
    fin = open(filename,"r")
    #Let's read line by line
    line = fin.readline().rstrip()
    #Make a list of keywords we read
    keywords = []
    while(line):
        word = str(line)
        keywords.append(word)
        line = fin.readline().rstrip()
    #print(keywords)

    
    
    #NCD = raw_input("Type disease you are looking at:")

    #######################
    NCD = "Obesity"
    yearTrainBegin = 2011
    yearTrainEnd = 2015 #4
    yearTestBegin = 2015 #4
    yearTestEnd = 2016  #5   
    #######################

    #yearTrainBegin = int(raw_input("Type year from which to begin the training:"))
    #yearTrainEnd = int(raw_input("Type year until which the training should end:"))

    #yearTestBegin = int(raw_input("Type year from which to begin the testing:"))
    #yearTestEnd = int(raw_input("Type year until which the testing should end:"))

    ########################
    yearsTrain = range(yearTrainBegin,yearTrainEnd)
    yearsTest = range(yearTestBegin,yearTestEnd)
    ########################
    
    #yearsTrain = range(yearTrainBegin,yearTrainEnd)
    #yearsTest = range(yearTestBegin,yearTestEnd)

    #Now we read all the Y values from all the years

    #Let us now read all the X valuees from all the years
    #folderY = raw_input("Type in the folder where all the Y values exist:")

    ###############
    folderY = "Data/Obesity/True Values/"
    ###############

    #nFlag = bool(raw_input("Should the Y values be normalized?"))

    ###############
    nFlag = False
    ###############
    
    YsAcrossYears = []
    for year in range(yearTrainBegin,yearTestEnd):
        filename = folderY+str(NCD)+"_"+str(year)+".csv"
        YsAcrossYears.append(readY(filename,nFlag))

####################################################################################################################################################

    #Now this is all reading spatial data
        
    #Let us now read all the X valuees from all the years
    #folderXSpatial = raw_input("Type in the folder where all the X values exist for spatial data:")

    ##########
    folderXSpatial = "Data/Obesity/Spatial/Round 1/"
    ##########
    
    #This will be a 3d list
    XsAcrossYears = []
    for year in range(yearTrainBegin,yearTestEnd):
        xsInAYear = []
        for keyword in keywords:
            if(year == 2011):
                filename = folderXSpatial+str(keyword)+"_"+str(year)+".csv"
            else:
                filename = folderXSpatial+str(keyword)+"_"+str(year)+"_adjustedByPopPen.csv"
            xsInAYear.append(readX(filename))
        XsAcrossYears.append(xsInAYear)
 
    #Now I need to do something with this data.
    #This first regression is just to concatenate different years without normalization
    trainTest = firstReg(XsAcrossYears,YsAcrossYears,yearsTrain,yearsTest,yearTrainBegin)

    
    train = trainTest[0]
    test = trainTest[1]
    #train should have 2 things [x,y]
    xTrain = train[0]
    yTrain = train[1]
    xTest = test[0]
    yTest = test[1]

    xTrain = preprocessing.scale(xTrain)

    #scaler = preprocessing.StandardScaler().fit(xTrain)
    
    #xTest = scaler.transform(xTest) 
    xTest = preprocessing.scale(xTest)
    
    lasso = linear_model.Lasso(random_state=0)
    alphas = np.arange(0.01,1,0.01)
    #print(alphas)
    #alphas = [0.1,0.01,1,10,2.473]
    #alphas = [0.6]
    #alphas = [0.25]
    #alphas = [1.4]
    scores = list()
    scores_std = list()
    n_folds = 10
    counter = 0
    alphasAndScores = []
    for alpha in alphas:
        if(counter % 10 == 0):
            print counter
        lasso.alpha = alpha
        this_scores = cross_val_score(lasso, xTrain, yTrain, cv=n_folds, n_jobs=1,scoring='r2')
        #positive_scores = [score * -1.0 for score in this_scores]
        #alphasAndScores.append((alpha,positive_scores))
        scores.append(np.mean(this_scores))
        scores_std.append(np.std(this_scores))
        counter+=1
    #scores,scores_std = np.array(scores), np.array(scores_std)
    #scores = [score * -1.0 for score in scores] #Just made all scores positive
    
    #Now m will be the score that corresponds to the alpha we want to pick
    m = max(scores)
    #alfa = pickAlpha(alphasAndScores)
    #alfa = 0.08
    indexLst = [i for i,j in enumerate(scores) if j == m]
    alfaIndex = indexLst[0]
    alfa = alphas[alfaIndex]
    #alfa = 0.01
    print(alfa)
    #alfa = alfa
    #Cretae a regression model
    alfa = 0.13
    regr = linear_model.Lasso(alpha=alfa)
    regr.fit(xTrain,yTrain)
    #The coefficients
    print ('Coefficients: ',regr.coef_)
    print ('Intercept: ',regr.intercept_)
    predictions = regr.predict(xTest)
    true = yTest
    preds = []
    #print predictions
    for x in np.nditer(predictions):
        preds.append(float(x))
    #print xTest[0]
    
    ###Now let's write all the predictions
    fout = open("Data/Middle East/Arabic/predictions_obesity.txt","w")
    for i in range(len(preds)):
        fout.write(str(true[i][0]))
        fout.write("\t")
        fout.write(str(preds[i]))
        fout.write("\n")
    fout.close()

main()
