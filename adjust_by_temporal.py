import csv
import sys

states = {
        'Sharjah': 'Sharjah',
        'Ajman': 'Ajman',
        'Dubai': 'Dubai',
        'Ras al Khaimah': 'Ras al Khaimah',
        'Abu Dhabi':'Abu Dhabi',
        'Fujairah':'Fujairah'
}



def calcPopulationSums(state2prev):
    #Firstly I need a dictioanry from states to population
    return [prev for (state,prev) in state2prev]

def readXTemporal(filename):
    fin = open(filename,'rb')
    reader = csv.reader(fin)
    counter = 0
    yearTuples = []
    yearPrev = "2011"
    yearNow = "2011"
    yearSum = 0
    for row in reader:
        if(counter > 2):
            yearMonth = str(row[0])
            lst = yearMonth.split("-")
            yearNow = lst[0]
            
            if(yearNow != yearPrev):
                yearTuples.append((yearPrev,yearSum*1.0/12))
                yearSum = float(row[1])
                yearPrev = yearNow
            else:
                yearSum += float(row[1])
                
        counter+=1
    yearTuples.append((yearPrev,yearSum*1.0/12))
    #yearTuples = yearTuples[7:]
    #print yearTuples
    maxPrevalence = max(p for (_,p) in yearTuples)
    yearTuples = [(y,float(p)*100/maxPrevalence) for (y,p) in yearTuples]
    print yearTuples
    return yearTuples


#Reading all the keywords
keywords = []
filename = "related_keywords_for_USA.txt"
fin = open(filename,"r")
line = fin.readline().rstrip()
while(line):
    keywords.append(line)
    line = fin.readline().rstrip()
fin.close()

stateNames = states.values()

#Now let us read the temporal data
#Let us go through each keyword one by one
for keyword in keywords:
    #Folder which contains temporal data
    folderTemporal = "Temporal_English/"
    #Reading all the temporal values
    yearTuples = readXTemporal(folderTemporal+keyword+".csv")
    
    yearTrainBegin = 2011
    yearTrainEnd = 2014
    yearTestBegin = 2014
    yearTestEnd = 2016
    
    #Reading all the years
    #Will be saving the average across states for all the years
    sumsSpatial = []
    year2state2prev = []
    for year in range(yearTrainBegin,yearTestEnd+1):
        state2prev = []
        stateValues = []
        
        #Reading the spatial data for say cholesterol year by year
        folderSpatial = "English/"
        filename = folderSpatial +keyword + "_" + str(year) + "_averaged.csv"

        #Open the file as csv 
        fin = open(filename,'rb')
        reader = csv.reader(fin)
        counter = 0

        #Reading row by row
        for row in reader:
            if(counter > 2):
                state = str(row[0])
                #ensure we are only taking the states we have available
                if(state in stateNames):
                    stateValues.append(float(row[1]))
                    state2prev.append((state,float(row[1])))
            counter += 1
        
        #print state2prev
        #print "\n"
        year2state2prev.append(state2prev) #This is a list where 2004+index represents the year, then each index points to a list from all states to prev
        #Now here instead of sum(stateValues), it needs to change to population times that value
        populationSums = calcPopulationSums(state2prev)
        #sumsSpatial.append((year,sum(stateValues)))
        sumsSpatial.append((year,sum(populationSums)))
        fin.close()
    
    #print sumsSpatial
    #For a given keyword:
    #We have all the sums across states for all the years
    #We have the normalized web search activity across years
    #Now we need to go through each year except 2004 of course, to adjust the values
    for year in range(yearTrainBegin,yearTestEnd+1):
        #Lets write the adjusted values in a new file
        fileOutName = folderSpatial + keyword + "_" + str(year) + "_adjusted" + ".csv"
        fileOut = open(fileOutName,'wb')
        writer = csv.writer(fileOut)
        counter = 0
        print keyword
        #print yearTuples
        #print sumsSpatial
        if(yearTuples[0][1] != 0.0 and sumsSpatial[year-yearTrainBegin][1] != 0.0):
            #print yearTuples
            #print yearTuples[year-2004]
            Tfactor = (yearTuples[year-yearTrainBegin][1]/yearTuples[0][1]) #changing this from year-2004-1 to 0
            Sfactor = (sumsSpatial[0][1]/sumsSpatial[year-yearTrainBegin][1]) #changing this from year-2004-1 to 0
            factor = Tfactor * Sfactor
        elif(yearTuples[0][1] == 0.0 and sumsSpatial[year-yearTrainBegin][1] != 0.0):
            print "HERE"
            Tfactor = (yearTuples[year-yearTrainBegin][1]/1.0)
            Sfactor = (sumsSpatial[0][1]/sumsSpatial[year-yearTrainBegin][1])
            factor = Tfactor * Sfactor
        elif(yearTuples[0][1] != 0.0 and sumsSpatial[year-yearTrainBegin][1] == 0.0):
            print "HERE"
            Tfactor = (yearTuples[year-yearTrainBegin][1]/yearTuples[0][1])
            Sfactor = (sumsSpatial[0][1]/1)
            factor = Tfactor * Sfactor
        else:
            print "HERE"
            Tfactor = (yearTuples[year-yearTrainBegin][1]/1)
            Sfactor = (sumsSpatial[0][1]/1)
            factor = Tfactor * Sfactor
            
        #Now that we have the factor. Let's adjust the values and write
        writer.writerow(["Category: All categories"])
        writer.writerow([])
        writer.writerow(["Region",keyword+":("+str(year)+")"])
        state2prev = year2state2prev[year-yearTrainBegin]
        for state,prev in state2prev:
            writer.writerow([state,prev*factor])
        #fileOut.flush()
        #Now it's time to write all the states
        fileOut.close()

