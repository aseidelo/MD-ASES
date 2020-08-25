import loader
import json
from nltk import sent_tokenize, word_tokenize
from nltk.tag.perceptron import PerceptronTagger
from nltk.tokenize import ToktokTokenizer
toktok = ToktokTokenizer()
tagger = PerceptronTagger()

def compute_adj_priors(adj_sentiment, limit):
	#adj_sentiment = loader.load_yelp_adj_sentiment()
	#print(adj_sentiment)
	#loader.serialize_structure(adj_sentiment, 'adj_stars')
	limited = {}
	NH = [0,0,0,0,0]
	for adj in adj_sentiment:
		total = 0
		for star in adj_sentiment[adj]:
			total = total + star
		if(total > limit):
			limited[adj] = adj_sentiment[adj]
			for i in range(len(NH)):
				NH[i] = NH[i] + adj_sentiment[adj][i]
	for adj in limited:
		total_limited = sum(limited[adj])
		for i in range(len(limited[adj])):
			limited[adj][i] = float(limited[adj][i]) /total_limited
	total_NH = sum(NH)
	for i in range(len(NH)):
		NH[i] = float(NH[i]) / total_NH
	return limited, [1./5, 1./5, 1./5, 1./5, 1./5]

def compute_adj_posteriors(NeH, NH):
	Ne = 0.
	for i in range(len(NeH)):
		Ne = Ne + float(NeH[i])*NH[i]
	posterior = []
	for i in range(len(NeH)):
		try:
			posterior.append(float(NeH[i] * NH[i]) / Ne)
		except:
			posterior.append(0.)
	return posterior

def compute_business_sentence_aspect_sentiment(NeH_dict, NH, business_relevant_aspects, business_reviews_sentences, serialize=True):
	total = len(business_reviews_sentences)
	i = 0
	business_aspect_sentiment_overall = {}
	for business in business_reviews_sentences:
		print(str(i)+'/'+str(total))
		relevant_aspects = business_relevant_aspects[business]
		relevant_aspects_prob = dict.fromkeys(relevant_aspects, NH)
		#print(relevant_aspects_prob)
		#business_relevant_aspects[business][i][1] = NH for i in range(len(business_relevant_aspects[business]))
		for review in business_reviews_sentences[business]:
			business_reviews_sentences_prob = []
			for sentence in business_reviews_sentences[business][review]:
				review_sentence_aspects_prob = {}
				for aspect in sentence[1]:
					if(aspect in relevant_aspects):
						review_sentence_aspects_prob[aspect] = NH
						for adj in sentence[2]:
							if(adj in NeH_dict):
								relevant_aspects_prob[aspect] = compute_adj_posteriors(NeH_dict[adj], relevant_aspects_prob[aspect])
								review_sentence_aspects_prob[aspect] = compute_adj_posteriors(NeH_dict[adj], review_sentence_aspects_prob[aspect])
							#	business_relevant_aspects[business][][1]
							#	business_sentences[business]
				if(len(relevant_aspects_prob) > 0 and len(review_sentence_aspects_prob) > 0):
					business_reviews_sentences_prob.append([sentence[0], review_sentence_aspects_prob])
			business_reviews_sentences[business][review] = business_reviews_sentences_prob
		#print(relevant_aspects_prob)
		business_aspect_sentiment_overall[business] = {}
		for aspect in relevant_aspects_prob:
			print(aspect, relevant_aspects_prob[aspect])
			classification = max((val, idx) for (idx, val) in enumerate(relevant_aspects_prob[aspect]))[1]
			business_aspect_sentiment_overall[business][aspect] = [classification, relevant_aspects_prob[aspect]]
		i = i + 1
	if(serialize):
		loader.serialize_structure(business_reviews_sentences, 'business_review_sentence_aspect_sentiment', agregation='a+')
		loader.serialize_structure(business_aspect_sentiment_overall, 'business_aspect_sentiment', agregation='a+')

def compute_error(vec1, vec2):
	error = 0
	for i in range(len(vec1)):
		error = error + abs((vec1[i] - vec2[i]))
	return error

def choose_business_simmilar_sentiment_sentences():
	with open('business_aspect_sentiment.json', 'r') as file:
		i = 0
		for line in file:
			# loading business
			business_general_info = json.loads(line)
			business_name = ''
			smaller_error_sentences = {}
			for business in business_general_info:
				business_name = business
			business_general_info = business_general_info[business_name]
			business_review_sentence_info = None
			with open('business_review_sentence_aspect_sentiment.json', 'r') as file2:
				for line in file2:
					new_business_info = json.loads(line)
					buss = ''
					for business in new_business_info:
						buss = business
					if(buss == business_name):
						business_review_sentence_info = new_business_info[business]
						break
			# getting business general aspect sentiment
			general_aspects_sentiment = {}
			smaller_error = {}
			for aspect in business_general_info:
				general_aspects_sentiment[aspect] = business_general_info[aspect][1]
				smaller_error_sentences[aspect] = []
				smaller_error[aspect] = 99.
			# iterationg over reviews and sentences
			# get the most simmilar sentence
			for review in business_review_sentence_info:
				for sentence in business_review_sentence_info[review]:
					for aspect in sentence[1]:
						new_error = compute_error(general_aspects_sentiment[aspect], sentence[1][aspect])
						if(new_error < smaller_error[aspect]):
							smaller_error_sentences[aspect] = [review, sentence[0]]
							smaller_error[aspect] = new_error
			# find the sentence and store it
			best_sentence = {}
			for aspect in business_general_info:
				to_find_business = business_name
				try:
					to_find_review = smaller_error_sentences[aspect][0]
					to_find_sentence_index = smaller_error_sentences[aspect][1]
					with open('yelp/yelp_academic_dataset_review.json', 'r') as file3:
						for line in file3:
							entry = json.loads(line)
							line_business = entry['business_id']
							if(line_business == to_find_business):
								line_review = entry['review_id']
								if(line_review == to_find_review):
									line_text = entry['text']
									line_sentences = sent_tokenize(line_text)
									best_sentence[aspect] = line_sentences[to_find_sentence_index]
									break
				except:
					pass						
			to_store = {business_name : best_sentence}
			loader.serialize_structure(to_store, 'business_best_sentences', 'a+')
			i = i + 1
			print(i)

def test_business_sentiment(test_file, sentiment_prob_file):
	confusion_mat = [[0,0,0], [0,0,0], [0,0,0]]
	vocab_sentiment = loader.load('yelp/'+sentiment_prob_file)
	with open('yelp/'+test_file+'.json', 'r') as file:
		for line in file:
			business_id = None
			business = None
			json_line = json.loads(line)
			for b in json_line:
				business_id = b
				business = json_line[b]
			for review_id in business:
				review_sentiment_prob_distribution = [0.22, 0.11, 0.67]
				sentence_lemmas = business[review_id]['lemmas']
				actual_sentiment = business[review_id]['sentiment']
				for sentence in sentence_lemmas:
					for lemma in sentence:
						if(lemma in vocab_sentiment):
							review_sentiment_prob_distribution = compute_adj_posteriors(vocab_sentiment[lemma], review_sentiment_prob_distribution)
				sentiment_inference = max((val, idx) for (idx, val) in enumerate(review_sentiment_prob_distribution))[1]
				if(review_sentiment_prob_distribution == [1./3, 1./3, 1./3]):
					sentiment_inference = 1
				confusion_mat[sentiment_inference][actual_sentiment] = confusion_mat[sentiment_inference][actual_sentiment] + 1
	acc = float(confusion_mat[0][0] + confusion_mat[1][1] + confusion_mat[2][2])/sum(sum(confusion_mat,[]))
	return confusion_mat, acc

if __name__ == '__main__':
	print(test_business_sentiment('TESTLIMITEDbusiness_review_db', 'POSTTRAINSENTIMENTBALENCED'))

