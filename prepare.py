import csv, pprint, util, random
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from collections import Counter
import numpy as np
def readData(path):
	'''
	This function will read data from a csv file
	and return the corresponding list
	'''
	f = open(path)
	return list(csv.reader(f))

def getSentiment(sentence):
	'''
	This function will return polarity 
	scores given a sentence
	'''
	sid = SentimentIntensityAnalyzer()
	ss = sid.polarity_scores(sentence)
	return ss

def parseUrl(url):
	'''
	This function returns categories in a url string
	'''
	return url.split('/')[2:5]


def isNegative(no):
	if no<0.0:
		return 0
	else:
		return 1

def underline(sentence, word):
	words = sentence.split()[:10]
	for wordd in words:
		if wordd == word:
			index = words.index(wordd)
			words[index] = '\033[91m'+wordd+'\033[0m'
	return " ".join(words)
			




def main():
	path = 'inflationHeadlines.csv'
	data = readData(path)
	no =500
	data = [x for x in data if 'business' in x[3] and 'seeds' not in x[3]]
	sentences = [x[1] for x in data if x[7]=='0'][:no]
	ydat = [x[7] for x in data if x[7]=='0'][:no]
	ydat2 = [x[7] for x in data if x[7]=='1'][:no]
	sentencesPos = [x[1] for x in data if x[7]=='1'][:no]
	ydat.extend(ydat2)
	sentences.extend(sentencesPos)
	myStr = " ".join(sentences)
	dicti = util.buildDictionary(myStr)[0]
	maxLength = 10
	randomVectors = np.random.random_integers(0,2, size=[len(dicti),25])
	#xData = [util.binaryEncode(sent, dicti, maxLength, randomVectors) for sent in sentences]
	xData = [util.oneHotencode(sent, dicti, maxLength) for sent in sentences]
	yData = []
	for y in ydat:
		vector = [0]*2
		vector[int(y)] = 1
		yData.append(vector)
	c = list(zip(xData, yData, sentences))
	random.shuffle(c)
	xData, yData, sen = zip(*c)
	xD = [x[0] for x in xData]
	sL = [x[1] for x in xData]
	#print(yData[133], len(xData[133][0]), len(sL))
	split = int(0.65 * len(xData))
	dimen = len(xD[0][0])
	return xD[:split], sL[:split], yData[:split], dimen, maxLength, xD[split:], sL[split:], yData[split:], sen