
# reviews - text with score
# reviews_text/text - text of movie reviews
# 

import nltk

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

from nltk.classify.scikitlearn import SklearnClassifier
import pickle

from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC

import os
import csv

pos_rev = []
somewhat_pos_rev = []
neutral_rev = []
somewhat_neg_rev = []
neg_rev = []

def read_reviews():
	train = open('Data/train.tsv')
	reader = list(csv.reader(train, delimiter = '\t'))
	reader.pop(0)
	return reader

def extract_text(reviews):

	allwordlists = []
	for i in range(len(reviews) - 100000):		
		allwordlists.append(word_tokenize(reviews[i][2]))
	
	return allwordlists

def extract_allwords(reviews):

	allwords = []
	for i in range(len(reviews) - 100000):		
		for word in word_tokenize(reviews[i][2]):
			allwords.append(word)
	return allwords


def process_text(text):

	stop_words = set(stopwords.words("english"))
	ps = PorterStemmer()

	processed_text = []

	for w in text:
		if w in stop_words:
			continue

		if w.isalpha():
			processed_text.append(ps.stem(w.lower()))

	return processed_text


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

def find_features(review_text):
	words = word_tokenize(review_text)
	words = process_text(words)

	features_exist = {}
	for word in features:
		features_exist[word] = (word in words)

	return features_exist

reviews = read_reviews()
allwords = extract_allwords(reviews)

processed_words = process_text(allwords)

features = nltk.FreqDist(processed_words)
features = features.most_common(300)
features = [key for (key, value) in features]


featuresets = [(find_features(review[2]), review[3]) for review in reviews]

testing_set = featuresets[:10000]
training_set =  featuresets[10000:20000]

classifier = nltk.NaiveBayesClassifier.train(training_set)
print("Original Naive Bayes Algo accuracy percent:", (nltk.classify.accuracy(classifier, testing_set))*100)
classifier.show_most_informative_features(15)

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
