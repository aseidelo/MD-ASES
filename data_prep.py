import json
import re
import os.path
import time
from nltk import sent_tokenize, word_tokenize
from nltk.tag.perceptron import PerceptronTagger
from nltk.tokenize import ToktokTokenizer
import loader
from langdetect import detect

toktok = ToktokTokenizer()
tagger = PerceptronTagger()
N_REVIEWS = 8021122
N_BUSINESS = 209393

TRAIN_SIZE_REVIEWS = int(N_REVIEWS*0.7)
TEST_SIZE_REVIEWS = N_REVIEWS - TRAIN_SIZE_REVIEWS

TRAIN_SIZE = int(N_BUSINESS*0.7)
VAL_SIZE = int(N_BUSINESS*0.15)
TEST_SIZE = N_BUSINESS - (TRAIN_SIZE + VAL_SIZE)

POST_TRAIN_SENTIMENT = [1272139, 578585, 3572492]
POST_TEST_SENTIMENT = [480842, 219477, 1346932]

LIMITED_TEST_SENTIMENT = [125056, 44173, 253135]
LIMITED_TRAIN_SENTIMENT = [322912, 114514, 647866]

LIMITED_TRAIN_BALANCED = []


def my_tokkenize(text, vocab):
	paragraphs = text.split('\n')
	sentences = []
	for paragraph in paragraphs:
		sentences = sentences + sent_tokenize(paragraph)
	sentences_words = []
	for sent in sentences:
		words = []
		tokkens = word_tokenize(sent.lower())
		for word in tokkens:
			try:
				words.append(vocab[word])
			except:
				pass
		sentences_words.append(words)
	return sentences_words, sentences

def data_prep(in_name, out_name):
	vocab, n_vocab = loader.load_corpora_vocab()
	sentiment_count = [0,0,0]
	with open('yelp/' + in_name + '.json') as file:
		k = 0
		for line in file:
			business_dict = json.loads(line)
			business_id = ''
			for b in business_dict:
				business_id = b
			business = business_dict[business_id]
			to_use_reviews = {}
			delta_sentiment_count = [0,0,0]
			for review_id in business:
				if(len(to_use_reviews)<= 20):
					review = business[review_id]
					text = review['text']
					try:
						if(detect(text) == 'en'):
							lemmas, = my_tokkenize(text.lower(), vocab)[0]
							if(len(lemmas) > 4):
								stars = int(review['stars'])
								useful = int(review['useful'])
								sentiment = None
								if(stars == 1 or stars == 2):
									sentiment = 0
								elif(stars == 3):
									sentiment = 1
								else:
									sentiment = 2
								delta_sentiment_count[sentiment] = delta_sentiment_count[sentiment] + 1
								to_use_reviews[review_id] = {'sentiment' : sentiment, 'text' : text, 'useful' : useful, 'lemmas' : lemmas}
					except Exception as e:
						print(e)
				else:
					break
			if(len(to_use_reviews) >= 10):
				for i in range(len(sentiment_count)):
					sentiment_count[i] = sentiment_count[i] + delta_sentiment_count[i]
				buss = {business_id: to_use_reviews}
				loader.serialize_structure(buss, 'yelp/' + out_name, 'a+')
			k = k + 1
			print(k)
	return sentiment_count


def create_train_test_sets(train_file_name, test_file_name):
	train_file = open(train_file_name+'.json', 'w')
	test_file = open(test_file_name+'.json', 'w')
	with open('yelp/business_review_dataset.json') as file:
		i = 0
		for line in file:
			if(i < TRAIN_SIZE):
				train_file.write(line)
			else:
				test_file.write(line)
			i = i + 1
			if(i%10000==0):
				print(i, N_BUSINESS)

def create_train_val_test_sets(dataset_file_name, train_file_name, val_file_name, test_file_name, init=0, end=N_BUSINESS):
	vocab, n_vocab = loader.load_corpora_vocab()
	with open('yelp/' + dataset_file_name + '.json') as file:
		i = 0
		for line in file:
			if(i > init and i < end):
				entry = json.loads(line)
				business = None
				business_id = None
				for b in entry:
					business_id = b
					business = entry[b]
				business_sentiment = 0
				n_reviews = 0
				to_save_business = {'business_id' : business_id, 'mean_sentiment': 0 , 'reviews' : {}}
				for review_id in business:
					review = business[review_id]
					text = review['text']
					try:
						language = detect(text)
						if(language == 'en'):
							lemmas, sentences = my_tokkenize(text, vocab)
							n_reviews = n_reviews + 1
							stars = int(review['stars'])
							sent = 0
							if stars == 3:
								sent = 1
							elif stars == 4 or stars == 5:
								sent = 2
							business_sentiment = business_sentiment + sent
							to_save_business['reviews'][review_id] = {'lemmas' : lemmas, 'text' : sentences, 'sentiment' : sent, 'stars' : stars}
					except:
						pass
				if(len(to_save_business['reviews']) > 0):
					business_sentiment = int((float(business_sentiment)/n_reviews) + 0.5)
					to_save_business['mean_sentiment'] = business_sentiment
					if(i < TRAIN_SIZE):
						loader.serialize_structure2(to_save_business, 'yelp/'+train_file_name, 'a+')
					elif(i > TRAIN_SIZE and i < TRAIN_SIZE + VAL_SIZE):
						loader.serialize_structure2(to_save_business, 'yelp/'+val_file_name, 'a+')
					else:
						loader.serialize_structure2(to_save_business, 'yelp/'+test_file_name, 'a+')
				elif(i > end):
					break
			i = i + 1
			print(i, N_BUSINESS)

def sentiment_train_prep(in_name, out_name):
	vocab, n_vocab = loader.load_corpora_vocab()
	vocab_freq = {}
	with open('yelp/' + in_name + '.json') as in_file:
		sent_order = [1, 0, 2]
		i = 0
		j = 0
		for line in in_file:
			review = json.loads(line)
			stars = int(review['stars'])
			sent = 0
			if stars == 3:
				sent = 1
			elif stars == 4 or stars == 5:
				sent = 2
			text = review['text']
			tokk_sentences = my_tokkenize(text.lower(), vocab)[0]
			if(sent == sent_order[j]):
				j = j + 1
				if j == 3:
					j = 0
				for sentence in tokk_sentences:
					for tokk in sentence:
						if(tokk not in vocab_freq):
							vocab_freq[tokk] = [0, 0, 0]
						vocab_freq[tokk][sent] = vocab_freq[tokk][sent] + 1
			if(i%10000==0):
				print(i, TRAIN_SIZE_REVIEWS)
			if(i >= TRAIN_SIZE_REVIEWS):
				break
			i = i + 1
	loader.serialize_structure(vocab_freq, 'yelp/' + out_name)			
	return vocab_freq

def sentiment_post_processing(in_name, out_name):
	with open('yelp/' + in_name + '.json') as in_file:
		for line in in_file:
			word = json.loads(line)
			word_lemma = None
			word_freq = None
			for w in word:
				word_lemma = w
				word_freq = word[w]
			total_freq = sum(word_freq)
			if(total_freq >= 5):
				use_word = False
				prob = []
				for i in range(len(word_freq)):
					prob.append(float(word_freq[i])/total_freq)
					if(prob[i] >= 0.73):
						use_word = True
				if(use_word):
					loader.serialize_structure({word_lemma : prob}, 'yelp/' + out_name, 'a+')


def tfidf_train_prep(in_name, out_name):
	with open('yelp/' + in_name + '.json') as in_file:
		i = 0
		j = 0
		for line in in_file:
			business_id = ''
			buss = json.loads(line)
			for b in buss:
				business_id = b
			business = buss[business_id]
			business_vocab_freq = {}
			for review_id in business:
				review = business[review_id]
				sentences_lemmas = review['lemmas']
				for sentence in sentences_lemmas:
					for tokk in sentence:
						if(tokk not in business_vocab_freq):
							business_vocab_freq[tokk] = 0
						business_vocab_freq[tokk] = business_vocab_freq[tokk] + 1
			loader.serialize_structure({business_id : business_vocab_freq}, 'yelp/' + out_name, 'a+')
			if(i%10000==0):
				print(i, TRAIN_SIZE_REVIEWS)
			if(i >= TRAIN_SIZE_REVIEWS):
				break
			i = i + 1

def count_sentiment(in_name):
	count = [0,0,0]
	with open('yelp/' + in_name + '.json') as in_file:
		for line in in_file:
			buss = json.loads(line)
			for b in buss:
				for review in buss[b]:
					sent = int(buss[b][review]['sentiment'])
					count[sent] = count[sent] + 1
	return count

def count_stars(in_name):
	count = [0,0,0,0,0]
	with open('yelp/' + in_name + '.json') as in_file:
		for line in in_file:
			buss = json.loads(line)
			for review in buss['reviews']:
				stars = int(buss['reviews'][review]['stars'])
				count[stars-1] = count[stars-1] + 1
	return count

def normalize_validation(validation_file_name):
	stars_initial_count = [184697, 91544, 120765, 240432, 514747]
	stars_final_count = [0, 0, 0, 0, 0]
	stars_order = [2,3,1,4,5]
	considered_reviews = {} # business_id -> review_id
	normalized_businesses = {}
	finished_businesses = []
	batch_size = 1000
	batch_count = 0
	i = 0
	restart_file = False
	dont_stop = True
	with open('yelp/'+validation_file_name+'.json', 'r') as file:
		while(dont_stop):
			dont_stop = False
			batch = []
			j = 0
			for line in file:
				if(j >= batch_count*batch_size and j < batch_count*batch_size + batch_size):
					batch.append(line)
				if(len(batch) >= batch_size):
					dont_stop = True
					break
				j = j + 1
			while(True):
				for line in batch:
					business_id = str(line[17:39])
					if(business_id not in finished_businesses):
						new_business = json.loads(line)
						#print(business_id)
						if(business_id not in considered_reviews):
							considered_reviews[business_id] = []
						if(len(new_business['reviews']) > len(considered_reviews[business_id])):
							for review_id in new_business['reviews']:
								#print('  ' + review_id)
								if(review_id not in considered_reviews[business_id]):
									if(int(new_business['reviews'][review_id]['stars'])==stars_order[i]):
										if(business_id not in normalized_businesses):
											normalized_businesses[business_id] = {'business_id' : business_id, 'reviews' : {}, 'mean_sentiment' : new_business['mean_sentiment']}
										considered_reviews[business_id].append(review_id)
										normalized_businesses[business_id]['reviews'][review_id] = new_business['reviews'][review_id]
										stars_final_count[stars_order[i]-1] = stars_final_count[stars_order[i]-1] + 1
										i = i + 1
										if(i>=5):
											i = 0
										restart_file = True
										print(stars_final_count, min(stars_initial_count) - min(stars_final_count))
								if(restart_file):
									break
						else:
							finished_businesses.append(business_id)
							#print(finished_businesses)
					#else:
					#	print('Nao entrou')
					if(restart_file):
						break
				if(restart_file is False):
					break
				restart_file = False
			batch_count = batch_count + 1
	print('Final', stars_final_count)
	for b in normalized_businesses:
		loader.serialize_structure2(normalized_businesses[b], 'yelp/business_review_validationset_balanced2', 'a+')


if __name__ == '__main__':
	#loader.create_business_db(int(N_BUSINESS/8))
	#create_train_test_sets('yelp/TRAINbusiness_review_db', 'yelp/TESTbusiness_review_db')
	#print(data_prep('TESTbusiness_review_db', 'TESTLIMITEDbusiness_review_db'))
	#sentiments = data_prep('TRAINbusiness_review_db', 'TRAINLIMITEDbusiness_review_db')
	#print(sentiments)
	#print(sentiment_post_processing('TRAINSENTIMENTBALENCED', 'POSTTRAINSENTIMENTBALENCED'))
	#print(tfidf_train_prep('TESTLIMITEDbusiness_review_db', 'BUSINESSLEMMASFREQ'))
	#create_train_val_test_sets('business_review_dataset', 'business_review_trainset', 'business_review_validationsetCERTO', 'business_review_testset', TRAIN_SIZE, TRAIN_SIZE + VAL_SIZE)
	#create_train_val_test_sets('business_review_dataset', 'business_review_trainset', 'business_review_validationsetCERTO', 'business_review_testsetCERTO', TRAIN_SIZE + VAL_SIZE, N_BUSINESS)
	#print(count_stars('business_review_validationset'))
	normalize_validation('business_review_validationset')