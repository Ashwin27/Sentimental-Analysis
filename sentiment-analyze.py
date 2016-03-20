from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

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

