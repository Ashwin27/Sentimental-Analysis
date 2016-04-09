
# reviews - text with score
# reviews_text/text - text of movie reviews
# 

import nltk

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.sentiment.sentiment_analyzer import SentimentAnalyzer
from nltk.sentiment.utils import *

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
	for i in range(len(reviews)):		
		allwordlists.append(word_tokenize(reviews[i][2]))
	
	return allwordlists

def extract_allwords(reviews):

	allwords = []
	for i in range(len(reviews)):		
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
features = features.most_common(1000)
features = [key for (key, value) in features]


featuresets = [(find_features(review[2]), review[3]) for review in reviews]

testing_set = featuresets[130000:150000]
training_set =  featuresets[:130000]

# Run only once
classifier = nltk.NaiveBayesClassifier.train(training_set)

#classifier_f = open("naive_bayes.pickle", "rb")
#classifier = pickle.load(classifier_f)
#classifier_f.close()


# Naive Bayes
print("Original Naive Bayes Algo accuracy percent:", (nltk.classify.accuracy(classifier, testing_set))*100)
classifier.show_most_informative_features(15)

filename = "naive_bayes.pickle"
save_classifier = open(filename, "wb")
pickle.dump(classifier, save_classifier)
save_classifier.close()

# Multinomial Naive Bayes
MNB_classifier = SklearnClassifier(MultinomialNB())
MNB_classifier.train(training_set)
print("MNB_classifier accuracy percent:", (nltk.classify.accuracy(MNB_classifier, testing_set))*100)

filename = "multinomial_naive_bayes.pickle"
save_classifier = open(filename, "wb")
pickle.dump(classifier, save_classifier)
save_classifier.close()


# Bernouli Classifier
BernoulliNB_classifier = SklearnClassifier(BernoulliNB())
BernoulliNB_classifier.train(training_set)
print("BernoulliNB_classifier accuracy percent:", (nltk.classify.accuracy(BernoulliNB_classifier, testing_set))*100)

save_classifier = open("bernoulli_naive_bayes.pickle", "wb")
pickle.dump(classifier, save_classifier)
save_classifier.close()

LogisticRegression_classifier = SklearnClassifier(LogisticRegression())
LogisticRegression_classifier.train(training_set)
print("LogisticRegression_classifier accuracy percent:", (nltk.classify.accuracy(LogisticRegression_classifier, testing_set))*100)

save_classifier = open("logistic_regression.pickle", "wb")
pickle.dump(classifier, save_classifier)
save_classifier.close()

SGDClassifier_classifier = SklearnClassifier(SGDClassifier())
SGDClassifier_classifier.train(training_set)
print("SGDClassifier_classifier accuracy percent:", (nltk.classify.accuracy(SGDClassifier_classifier, testing_set))*100)

save_classifier = open("SGDClassifier.pickle", "wb")
pickle.dump(classifier, save_classifier)
save_classifier.close()

SVC_classifier = SklearnClassifier(SVC())
SVC_classifier.train(training_set)
print("SVC_classifier accuracy percent:", (nltk.classify.accuracy(SVC_classifier, testing_set))*100)

save_classifier = open("SVCClassifier.pickle", "wb")
pickle.dump(classifier, save_classifier)
save_classifier.close()

LinearSVC_classifier = SklearnClassifier(LinearSVC())
LinearSVC_classifier.train(training_set)
print("LinearSVC_classifier accuracy percent:", (nltk.classify.accuracy(LinearSVC_classifier, testing_set))*100)

save_classifier = open("Linear_SVCClassifier.pickle", "wb")
pickle.dump(classifier, save_classifier)
save_classifier.close()

'''
NuSVC_classifier = SklearnClassifier(NuSVC())
NuSVC_classifier.train(training_set)
print("NuSVC_classifier accuracy percent:", (nltk.classify.accuracy(NuSVC_classifier, testing_set))*100)

save_classifier = open("NU_SVCClassifier.pickle", "wb")
pickle.dump(classifier, save_classifier)
save_classifier.close()
'''