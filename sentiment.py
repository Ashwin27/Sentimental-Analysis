
# reviews - text with score
# reviews_text/text - text of movie reviews
# 

import nltk

from numpy import asarray

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

from nltk.sentiment.sentiment_analyzer import SentimentAnalyzer
from nltk.sentiment.util import *

from nltk.collocations import *

from nltk.classify.scikitlearn import SklearnClassifier
import pickle

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline

from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
clf1 = LogisticRegression(random_state=1)
clf2 = RandomForestClassifier(random_state=1)

import os
import csv
import types

pos_rev = []
somewhat_pos_rev = []
neutral_rev = []
somewhat_neg_rev = []
neg_rev = []

def read_reviews(fileName):
	train = open(fileName)
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

	text = mark_negation(text, True)

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

training_reviews = read_reviews('Data/train.tsv')
#allwords = extract_allwords(reviews)
#processed_words = process_text(allwords)

training_set = []
training_labels = []
for review in training_reviews:
	training_set.append(review[2])
	training_labels.append(review[3])

X_train = asarray(training_set)

#clf1 = OneVsOneClassifier(LinearSVC())
clf2 = OneVsRestClassifier(LinearSVC())
#clf3 = RandomForestClassifier()

#eclf = VotingClassifier(estimators=[('ovrSVM', clf2), ('rf', clf3)], voting='soft')

text_clf = Pipeline([('vect', CountVectorizer( ngram_range=(1,2))),
                      ('tfidf', TfidfTransformer()),
                      ('clf', clf2),
])

text_clf = text_clf.fit(X_train, asarray(training_labels))

testing_reviews = read_reviews('Data/test.tsv')
testing_set = [(review[0], review[2]) for review in testing_reviews]

print(testing_set[0:2])

csvfile = open('submission.csv', 'w+')
spamwriter = csv.writer(csvfile, delimiter=',')
spamwriter.writerow(['PhraseId', 'Sentiment'])

for (PhraseId, Sentence) in testing_set:
	predicted = text_clf.predict(asarray([Sentence]))
	spamwriter.writerow([PhraseId, str(predicted[0])])