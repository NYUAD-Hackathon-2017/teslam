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

s = states.values()
yearStart = 2011
yearEnd = 2016
keywords = []

#Now let me first open the keywords
kfin = open("related_keywords_for_USA.txt","r")
line = kfin.readline().rstrip()
while(line):
    keywords.append(line)
    line = kfin.readline().rstrip()
kfin.close()



#Now let me read the samples and avergae them
for keyword in keywords:
    for year in range(yearStart,yearEnd+1):
        print keyword
        print str(year)
        #create an empty dictionary with states as keys
        dict1 = {}
        dict2 = {}
        dict3 = {}
        finalDict = {}
        for st in s:
            dict1[st] = 0
            dict2[st] = 0
            dict3[st] = 0
            finalDict[st] = 0
            
        f1 = open("English/"+str(keyword)+"_"+str(year)+".csv","rb")
        f2 = open("English/"+str(keyword)+"_"+str(year)+".csv","rb")
        fout = open("English/"+str(keyword)+"_"+str(year)+"_averaged.csv","wb")
        reader1 = csv.reader(f1)
        reader2 = csv.reader(f2)
        writer = csv.writer(fout)
        #1
        counter = 0
        for row in reader1:
            if(counter > 0):
                key = str(row[0])
                val = float(row[1])
                dict1[key] = val
            counter+=1
        f1.close()
        
        #2
        counter = 0
        for row in reader2:
            if(counter > 0):
                key = str(row[0])
                val = float(row[1])
                dict2[key] = val
            counter+=1
        f2.close()

        for st in s:
            finalDict[st] = (dict1[st]+dict2[st])/2.0

        #Now it's time to write
        writer.writerow(['Category: All categories'])
        writer.writerow([])
        writer.writerow(['Region',keyword+": ("+str(year)+")"])
        for key in finalDict.keys():
            row2write = [str(key),finalDict[key]]
            writer.writerow(row2write)
        fout.close()



