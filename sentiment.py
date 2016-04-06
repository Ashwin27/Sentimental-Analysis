# -*- coding:Latin-1 -*

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

from nltk.tag import PerceptronTagger
from nltk.data import find
from nltk.chunk.regexp import RegexpParser

PICKLE = "averaged_perceptron_tagger.pickle"
AP_MODEL_LOC = 'file:'+str(find('taggers/averaged_perceptron_tagger/'+PICKLE))
tagger = PerceptronTagger(load=False)
tagger.load(AP_MODEL_LOC)
pos_tag = tagger.tag

import os
import csv

train = open('Data/train.tsv')
reader = csv.reader(train, delimiter = '\t')

# Removes "a", "the" and other neutral words that 
# do not affect the meaning
stop_words = set(stopwords.words("english"))

ps = PorterStemmer()

result_words = []
result_numbers = []
first_row = True

for row in reader:
	if first_row:
		first_row = False
		continue
	
	result_words = row[2].split()
	result_numbers.append( row[0:2] + row[3:] )
	break

print(result_words)
print(result_numbers)

filtered_sentence = []
filtered_sentence = [w for w in result_words if not w in stop_words]
print(filtered_sentence)

for w in filtered_sentence:
	print(ps.stem(w))

tag_sentence = pos_tag(filtered_sentence)
print(tag_sentence)

chunkGram = """ Chunk: { <RB.?>* <JJ?>* <NN>? } """
chunkParser = RegexpParser(chunkGram)
chunked = chunkParser.parse(tag_sentence)
chunked.draw()

os.system("pause")