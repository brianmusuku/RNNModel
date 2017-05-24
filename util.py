import re
import collections
import numpy as np

def clean_str(string):
        """
        Tokenization/string cleaning for all datasets except for SST.
        Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data
        """
        string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
        string = re.sub(r"\'s", " \'s", string)
        string = re.sub(r"\'ve", " \'ve", string)
        string = re.sub(r"n\'t", " n\'t", string)
        string = re.sub(r"\'re", " \'re", string)
        string = re.sub(r"\'d", " \'d", string)
        string = re.sub(r"\'ll", " \'ll", string)
        string = re.sub(r",", " , ", string)
        string = re.sub(r"!", " ! ", string)
        string = re.sub(r"\(", " \( ", string)
        string = re.sub(r"\)", " \) ", string)
        string = re.sub(r"\?", " \? ", string)
        string = re.sub(r"\s{2,}", " ", string)
        return string.strip().lower()

def build_vocab(sentences):
        """
        Builds a vocabulary mapping from word to index based on the sentences.
        Returns vocabulary mapping and inverse vocabulary mapping.
        """
        # Build vocabulary
        word_counts = collections.Counter(sentences)
        # Mapping from index to word
        vocabulary_inv = [x[0] for x in word_counts.most_common()]
        vocabulary_inv = list(sorted(vocabulary_inv))
        # Mapping from word to index
        vocabulary = {x: i for i, x in enumerate(vocabulary_inv)}
        return [vocabulary, vocabulary_inv]

def buildDictionary(myStr):
        myStr = clean_str(myStr)
        words = myStr.split()
        return build_vocab(words)

def oneHotencode(sentence, dictionary, maxLength):
        '''
        Takes in a sentence and returns one hot encoding ector
        representation given a dictionary
        '''
        encoded = []
        sentence = clean_str(sentence)
        words = sentence.split()[:maxLength]
        indices = [dictionary[w] for w in words]
        dimen = len(dictionary)
        for index in indices:
                vector = [0 ]* dimen
                vector[index] = 1
                encoded.append(vector)
        remainder = maxLength-len(indices)
        if remainder>0:
                for i in range(remainder):
                        vector = [0 ]* dimen
                        encoded.append(vector)
        return encoded, len(indices)

def binaryEncode(sentence, dictionary, maxLength, randomVectors):
        encoded = []
        sentence = clean_str(sentence)
        words = sentence.split()[:maxLength]
        indices = [dictionary[w] for w in words]
        dimen = len(dictionary)
        #[[int(x) for x in list(np.random.normal(100, 100, 25))] for i in range(dimen)]
        for index in indices:
                #encoded.append(list(randomVectors[index]))
                encoded.append([int(x) for x in list(np.binary_repr(index, width=25))])
        remainder = maxLength-len(indices)
        if remainder>0:
                for i in range(remainder):
                        vector = [0 ]* 25
                        encoded.append(vector)
        return encoded, len(indices)






myStr = 'Builds a vocabulary mapping from word to index based on the sentences.Returns vocabulary mapping and inverse vocabulary mapping.'
dicti = buildDictionary(myStr)[0]
#print(oneHotencode(myStr, dicti, 25))
#print(binaryEncode(myStr, dicti, 25)[0])
