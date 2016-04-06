
# reviews - text with score
# reviews_text/text - text of movie reviews
# 

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

import os
import csv

pos_rev = []
somewhat_pos_rev = []
neutral_rev = []
somewhat_neg_rev = []
neg_rev = []

def read_reviews():
	train = open('Data/train.tsv')
	return list(csv.reader(train, delimiter = '\t'))


def extract_text(reviews):

	allwords = []
	first_row = True
	for i in range(len(reviews) - 100000):
		
		# Skip the first row
		if first_row:
			first_row = False
			continue

		allwords.append(word_tokenize(reviews[i][2]))

	return allwords


def process_text(text):

	return text


def categorize_reviews(reviews):
	for review in reviews:
		if(int(review[3]) == 0):
			neg_rev.append(review)
		elif(int(review[3]) == 1):
			somewhat_neg_rev.append(review)
		elif(int(review[3]) == 2):
			neutral_rev.append(review)
		elif(int(review[3]) == 3):
			somewhat_pos_rev.append(review)
		elif(int(review[3]) == 4):
			pos_rev.append(review)
		else:
			print("Review out of range. Rejecting...")


reviews = read_reviews()
allwordlists = review_text(reviews)

allwords = []
for wordlist in allwordlists:
	for words in wordlist:
		allwords.append(wordlist)


'''
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
'''
