import loader
import data_prep
import sentiment
import aspect
import json

TRAIN_SET = 'business_review_trainset'
VALIDATION_SET = 'business_review_validationset_balanced2'
TEST_SET = 'business_review_testset'
SENTIMENT_LEVELS = 2
TEST_SENTIMENT_LEVELS = 2
EXAMPLES = ['OizzqU3hEB1o3XlM5lWbcg', 'weAPN7qT5u4oppbkdzfhrA', 'dRY-GEj8ZRdv1rCy59i8nw', 'peKjMgbSHa7g7gyTJFd6WQ']

def train (train_file_name, validation_file_name, N, highest_probs, balance=False):
	lemmas_sentiment_freq, lemmas_document_freq, N_documents, mean_sentiment_prob_distribution, sentence_sentiment_prob_distribution = process_dataset_lemmas_freq(train_file_name, balance_sentiment=balance)
	#mean_sentiment_prob_distribution = [1./3, 1./3, 1./3]
	#sentence_sentiment_prob_distribution = [1./3, 1./3, 1./3]
	print('Training hiperparameters <n> of sentences and <h> probability limitation')
	print(highest_probs)
	print(N)
	best_models = []
	for n in N:
		best_h = highest_probs[0]
		best_acc = 0.
		for h in highest_probs:
			sentiment_model = train_sentiment(lemmas_sentiment_freq, h)
			confusion_mat = test(validation_file_name, n, sentiment_model, lemmas_document_freq, N_documents, mean_sentiment_prob_distribution, sentence_sentiment_prob_distribution)
			acc, pres, rec, F1 = evaluate(confusion_mat)
			print('n = ' + str(n) + ' | h = ' + str(h) + '| N evidences : ' + str(len(sentiment_model)))
			repport_results(acc, pres, rec, F1, confusion_mat)
			new_acc = acc
			if(new_acc > best_acc):
				best_h = h
				best_acc = new_acc
		best_models.append(train_sentiment(lemmas_sentiment_freq, best_h))
	return best_models, lemmas_document_freq, N_documents, mean_sentiment_prob_distribution, sentence_sentiment_prob_distribution

def train_sentiment_classifier (train_file_name, validation_file_name, highest_probs, balance=False):
	lemmas_sentiment_freq, lemmas_document_freq, N_documents, mean_sentiment_prob_distribution, sentence_sentiment_prob_distribution = process_dataset_lemmas_freq(train_file_name, balance_sentiment=balance)
	#mean_sentiment_prob_distribution = [1./3, 1./3, 1./3]
	#sentence_sentiment_prob_distribution = [1./3, 1./3, 1./3]
	print('Training hiperparameter <h> probability limitation')
	print(highest_probs)
	best_models = None
	best_h = highest_probs[0]
	best_acc = 0.
	for h in highest_probs:
		sentiment_model = train_sentiment(lemmas_sentiment_freq, h)
		confusion_mat = test_sentiment(validation_file_name, sentiment_model, lemmas_document_freq, N_documents, mean_sentiment_prob_distribution, sentence_sentiment_prob_distribution)
		acc, pres, rec, F1 = evaluate(confusion_mat)
		print('h = ' + str(h) + '| N evidences : ' + str(len(sentiment_model)))
		repport_results(acc, pres, rec, F1, confusion_mat)
		new_acc = acc
		if(new_acc > best_acc):
			best_h = h
			best_acc = new_acc
	best_models = train_sentiment(lemmas_sentiment_freq, best_h)
	return best_models, lemmas_document_freq, N_documents, mean_sentiment_prob_distribution, sentence_sentiment_prob_distribution, best_h


def train_sentiment(lemmas_freq, h):
	lemmas_prob = {}
	for lemma in lemmas_freq:
		lemma_freq = []
		total = sum(lemmas_freq[lemma])
		if(total >= 1):
			for i in range(len(lemmas_freq[lemma])):
				lemma_freq.append(float(lemmas_freq[lemma][i])/total)
			if(max(lemma_freq) >= h):
				lemmas_prob[lemma] = lemma_freq
		else:
			lemma_freq = [1./SENTIMENT_LEVELS]*SENTIMENT_LEVELS #[1./3, 1./3, 1./3] if SENTIMENT_LEVELS == 3 else [1./2, 1./2]
			if(max(lemma_freq) >= h):
				lemmas_prob[lemma] = lemma_freq			
	return lemmas_prob

def map_stars_sentiment(stars, n_sentiments):
	return int(float((stars-1)*(n_sentiments-1))/(5-1) + 0.5)

def map_sentiment(val, n_sentiments_in, n_sentiments_out):
	return int(float((val)*(n_sentiments_out - 1))/(n_sentiments_in - 1) + 0.5)

def test_sentiment(test_file_name, sentiment_model, lemmas_document_freq, N_documents, mean_sentiment_prob_distribution, sentence_sentiment_prob_distribution):
	confusion_mat = [[0]*SENTIMENT_LEVELS for i in range(SENTIMENT_LEVELS)]# [[0,0,0],[0,0,0],[0,0,0]] if SENTIMENT_LEVELS==3 else [[0,0],[0,0]]
	with open ('yelp/' + test_file_name + '.json') as file:
		for line in file:
			entry = json.loads(line)
			business = entry['reviews']
			business_id = entry['business_id']
			for review in business:
				# process summary sentiment
				infered_sentiment = infer_review_sentiment(business[review]['lemmas'], sentiment_model, sentence_sentiment_prob_distribution)
				# compare with actual mean sentiment
				actual_sentiment = map_stars_sentiment(int(business[review]['stars']), SENTIMENT_LEVELS)
				#print(infered_sentiment, actual_sentiment)
				if(SENTIMENT_LEVELS==2):
					if(int(business[review]['stars'])!=3): # sentimento neutro 3, não conta
						confusion_mat[infered_sentiment][actual_sentiment] = confusion_mat[infered_sentiment][actual_sentiment] + 1
				else:
					confusion_mat[infered_sentiment][actual_sentiment] = confusion_mat[infered_sentiment][actual_sentiment] + 1
	return confusion_mat

def test(test_file_name, n, sentiment_model, lemmas_document_freq, N_documents, mean_sentiment_prob_distribution, sentence_sentiment_prob_distribution):
	confusion_mat = [[0]*TEST_SENTIMENT_LEVELS for i in range(TEST_SENTIMENT_LEVELS)] # [[0,0,0],[0,0,0],[0,0,0]] if SENTIMENT_LEVELS==3 else [[0,0],[0,0]]
	with open ('yelp/' + test_file_name + '.json') as file:
		for line in file:
			entry = json.loads(line)
			business = entry['reviews']
			business_id = entry['business_id']
			summary_lemmas, summary_text, mean_sentiment = forward(n, business, lemmas_document_freq, N_documents, sentiment_model, sentence_sentiment_prob_distribution, business_id)
			# process summary sentiment
			infered_sentiment = infer_sentiment(summary_lemmas, sentiment_model, mean_sentiment_prob_distribution)
			# compare with actual mean sentiment
			#mean_sentiment = int(entry['mean_sentiment'])
			'''
			if(SENTIMENT_LEVELS==2):
				if(mean_sentiment==1):
					mean_sentiment=0
				elif(mean_sentiment==2):
					mean_sentiment=1
			'''
			#print(infered_sentiment, mean_sentiment)
			if(business_id in EXAMPLES):
				print(summary_text)
				print(infered_sentiment)
				print(mean_sentiment)
			try:
				confusion_mat[infered_sentiment][mean_sentiment] = confusion_mat[infered_sentiment][mean_sentiment] + 1
			except:
				print(infered_sentiment, mean_sentiment)
				raise
	return confusion_mat

def forward(n, business, lemmas_document_freq, N_documents, sentiment_model, sentence_sentiment_prob_distribution, business_id):
	business_lemmas_freq = process_business_lemmas_freq(business)
	max_tfidf_lemmas = process_max_tfidf(n, business_lemmas_freq, lemmas_document_freq, N_documents)
	# process mean sentiment
	max_tfidf_mean_sentiment = process_mean_aspect_sentiment(max_tfidf_lemmas, business, sentiment_model, sentence_sentiment_prob_distribution)
	if(business_id in EXAMPLES):
		print(max_tfidf_mean_sentiment)
	#print(max_tfidf_mean_sentiment)
	# process summary
	summary_lemmas = dict.fromkeys(max_tfidf_lemmas, '')
	summary_text = dict.fromkeys(max_tfidf_lemmas, '')
	min_error = dict.fromkeys(max_tfidf_lemmas, 99.)
	mean_sentiment = 0
	for review in business:
		sentences = business[review]['lemmas']
		plain_text_senteces = business[review]['text']
		i = 0
		for sent in sentences:
			plain_text = plain_text_senteces[i]
			summary_lemmas, summary_text, min_error = process_sentence_aspect_sentiment_simmilarity(sent, plain_text, max_tfidf_mean_sentiment, summary_lemmas, summary_text, min_error, sentiment_model)
			i = i + 1
		mean_sentiment = mean_sentiment + int(business[review]['stars']) - 1
	if(business_id in EXAMPLES):
		print(mean_sentiment)
	mean_sentiment = map_sentiment(float(mean_sentiment)/len(business), 5, TEST_SENTIMENT_LEVELS)
	if(business_id in EXAMPLES):
		print(mean_sentiment)
	return summary_lemmas, summary_text, mean_sentiment

def infer_review_sentiment(review, sentiment_model, mean_sentiment_prob_distribution):
	#print(summary)
	#print(mean_sentiment_prob_distribution)
	sentiment_dist = mean_sentiment_prob_distribution
	for sentence in review:
		for lemma in sentence:
			if(lemma in sentiment_model):
				sentiment_dist = sentiment.compute_adj_posteriors(sentiment_model[lemma], sentiment_dist)
				#print(lemma, sentiment_dist)
	sentiment_inference = max((val, idx) for (idx, val) in enumerate(sentiment_dist))[1]
	return sentiment_inference

def infer_sentiment(summary, sentiment_model, mean_sentiment_prob_distribution):
	#print(summary)
	#print(mean_sentiment_prob_distribution)
	sentiment_dist = mean_sentiment_prob_distribution
	for sentence_lemma in summary:
		for lemma in summary[sentence_lemma]:
			if(lemma in sentiment_model):
				sentiment_dist = sentiment.compute_adj_posteriors(sentiment_model[lemma], sentiment_dist)
				#print(lemma, sentiment_dist)
	sentiment_inference = max((val, idx) for (idx, val) in enumerate(sentiment_dist))[1]
	#if(TEST_SENTIMENT_LEVELS==3):
	#	if(max(sentiment_dist) > 0.3 and max(sentiment_dist) < 0.35):
	#		sentiment_inference = 1
	return map_sentiment(float(sentiment_inference), SENTIMENT_LEVELS, TEST_SENTIMENT_LEVELS)

def process_dataset_lemmas_freq(dataset_file_name, balance_sentiment=False):
	print('processing ' + dataset_file_name + ' lemmas freq')
	balance_sentiment_order = [1, 0, 2] if SENTIMENT_LEVELS==3 else [1, 0] 
	classes = [0]*SENTIMENT_LEVELS # [0,0,0] if SENTIMENT_LEVELS==3 else [0, 0] 
	initial_business_sentiment_prob_dist = [0]*SENTIMENT_LEVELS #[0,0,0] if SENTIMENT_LEVELS==3 else [0, 0] 
	initial_sentence_sentiment_prob_dist = [0]*SENTIMENT_LEVELS # [0,0,0] if SENTIMENT_LEVELS==3 else [0, 0] 
	i = 0
	lemmas_sentiment_freq = {}
	lemmas_document_freq = {}
	N_documents = 0
	with open('yelp/'+dataset_file_name+'.json') as file:
		for line in file:
			entry = json.loads(line)
			business = entry['reviews']
			business_id = entry['business_id']
			'''
			business_mean_sentiment = entry['mean_sentiment']
			if(SENTIMENT_LEVELS==2):
				if(business_mean_sentiment==1):
					business_mean_sentiment = 0
				elif(business_mean_sentiment==2):
					business_mean_sentiment = 1
			initial_business_sentiment_prob_dist[business_mean_sentiment] = initial_business_sentiment_prob_dist[business_mean_sentiment] + 1
			'''
			business_mean_sentiment = 0
			for review_id in business:
				sentences_lemmas = business[review_id]['lemmas']
				sentiment = map_stars_sentiment(int(business[review_id]['stars']), SENTIMENT_LEVELS)
				business_mean_sentiment = business_mean_sentiment + int(business[review_id]['stars'])
				#print(int(business[review_id]['stars']), sentiment)
				#print(classes)
				'''
				if(SENTIMENT_LEVELS==2):
					if(sentiment==1):
						sentiment = 0
					elif(sentiment==2):
						sentiment = 1
				'''
				lemmas_in_doc = []
				for sentence in sentences_lemmas:
					initial_sentence_sentiment_prob_dist[sentiment] = initial_sentence_sentiment_prob_dist[sentiment] + 1
					for lemma in sentence:
						classes[sentiment] = classes[sentiment] + 1
						if(lemma not in lemmas_in_doc):
							lemmas_in_doc.append(lemma)
						#if(balance_sentiment):
						#	if(sentiment != balance_sentiment_order[i]):
						#		break
						#	else:
						#		i = i + 1
						#		if(i > 2):
						#			i = 0
						if(lemma in lemmas_sentiment_freq):
							lemmas_sentiment_freq[lemma][sentiment] = lemmas_sentiment_freq[lemma][sentiment] + 1
						else:
							lemmas_sentiment_freq[lemma] = [0]*SENTIMENT_LEVELS # [0, 0, 0] if SENTIMENT_LEVELS == 3 else [0, 0]
							lemmas_sentiment_freq[lemma][sentiment] = 1
				for lemma in lemmas_in_doc:
					if(lemma in lemmas_document_freq):
						lemmas_document_freq[lemma] = lemmas_document_freq[lemma] + 1
					else:
						lemmas_document_freq[lemma] = 1
			business_mean_sentiment = map_stars_sentiment(float(business_mean_sentiment)/len(business), SENTIMENT_LEVELS)
			initial_business_sentiment_prob_dist[business_mean_sentiment] = initial_business_sentiment_prob_dist[business_mean_sentiment] + 1
			N_documents = N_documents + 1
			if(N_documents%1000==0):
				print(N_documents)
	print(lemmas_sentiment_freq['do'])
	if(balance_sentiment):
		less_class = min (classes)
		for lemma in lemmas_sentiment_freq:
			for i in range(len(lemmas_sentiment_freq[lemma])):
				lemmas_sentiment_freq[lemma][i] = int(lemmas_sentiment_freq[lemma][i] * (float(less_class) / classes[i]) + 0.5)
	print(lemmas_sentiment_freq['do'])
	print(classes)
	total_business_sentiment = sum(initial_business_sentiment_prob_dist)
	total_sentence_sentiment = sum(initial_sentence_sentiment_prob_dist)
	for i in range(len(initial_business_sentiment_prob_dist)):
		initial_business_sentiment_prob_dist[i] = float(initial_business_sentiment_prob_dist[i])/total_business_sentiment
		initial_sentence_sentiment_prob_dist[i] = float(initial_sentence_sentiment_prob_dist[i])/total_sentence_sentiment
	print(initial_business_sentiment_prob_dist, initial_sentence_sentiment_prob_dist)
	return lemmas_sentiment_freq, lemmas_document_freq, N_documents, initial_business_sentiment_prob_dist, initial_sentence_sentiment_prob_dist

def process_business_lemmas_freq(business):
	business_lemmas_freq = {}
	for review_id in business:
		for sentence in business[review_id]['lemmas']:
			for lemma in sentence:
				if(lemma in business_lemmas_freq):
					business_lemmas_freq[lemma] = business_lemmas_freq[lemma] + 1
				else:
					business_lemmas_freq[lemma] = 1
	return business_lemmas_freq # return lemmas frequency

def process_max_tfidf(n, business_lemmas_freq, lemmas_document_freq, N_documents):
	max_tfidf_lemmas = []
	N_documents = N_documents + 1
	max_tfidf_lemmas = [None for i in range(n)]
	tfidfs = [0. for i in range(n)]
	min_tfidf_id = 0
	for lemma in business_lemmas_freq:
		if lemma not in lemmas_document_freq:
			lemmas_document_freq[lemma] = 1
		lemma_tfidf = aspect.tf_idf(business_lemmas_freq[lemma], lemmas_document_freq[lemma], N_documents)
		if(lemma_tfidf > tfidfs[min_tfidf_id]):
			tfidfs[min_tfidf_id] = lemma_tfidf
			max_tfidf_lemmas[min_tfidf_id] = lemma
			min_tfidf_id = min((val, idx) for (idx, val) in enumerate(tfidfs))[1]
	return max_tfidf_lemmas # return max tf-idf words list (max to min) len([]) = n

def process_mean_aspect_sentiment(max_tfidf_lemmas, business, sentiment_model, sentence_sentiment_prob_distribution):
	max_tfidf_mean_sentiment = dict.fromkeys(max_tfidf_lemmas, sentence_sentiment_prob_distribution)
	for review in business:
		for sentence in business[review]['lemmas']:
			for max_tfidf_lemma in max_tfidf_lemmas:
				if(max_tfidf_lemma in sentence):
					for lemma in sentence:
						if(lemma in sentiment_model):
							max_tfidf_mean_sentiment[max_tfidf_lemma] = sentiment.compute_adj_posteriors(sentiment_model[lemma], max_tfidf_mean_sentiment[max_tfidf_lemma])
	return max_tfidf_mean_sentiment # return dict with mean sentiment for max tf-idf lemmas

def process_sentence_aspect_sentiment_simmilarity(sent, plain_text, max_tfidf_mean_sentiment, summary_lemmas, summary_text, min_error, sentiment_model):
	for max_tfidf_lemma in max_tfidf_mean_sentiment:
		if(max_tfidf_lemma in sent):
			sentence_aspect_sentiment = [1./SENTIMENT_LEVELS]*SENTIMENT_LEVELS #[1./3, 1./3, 1./3]
			for lemma in sent:
				if lemma in sentiment_model:
					sentence_aspect_sentiment = sentiment.compute_adj_posteriors(sentiment_model[lemma], sentence_aspect_sentiment)
			new_error = sentiment.compute_error(max_tfidf_mean_sentiment[max_tfidf_lemma], sentence_aspect_sentiment)
			if(new_error < min_error[max_tfidf_lemma]):
				min_error[max_tfidf_lemma] = new_error
				summary_lemmas[max_tfidf_lemma] = sent
				summary_text[max_tfidf_lemma] = plain_text
	return summary_lemmas, summary_text, min_error # return new summary and minimal error

def evaluate(confusion_mat):
	total = sum(sum(confusion_mat,[]))
	acc = 1
	pres = 1
	rec = 1
	if(SENTIMENT_LEVELS==3):
		TP_0 = confusion_mat[0][0]
		TP_1 = confusion_mat[1][1]
		TP_2 = confusion_mat[2][2]
		FP_0 = confusion_mat[0][1] + confusion_mat[0][2]
		FP_1 = confusion_mat[1][0] + confusion_mat[1][2]
		FN_0 = confusion_mat[1][0] + confusion_mat[2][0]
		FN_1 = confusion_mat[0][1] + confusion_mat[2][1]
		acc = float(TP_0 + TP_1 + TP_2)/total
		pres = float(TP_0 + TP_1)/(TP_0 + TP_1 + FP_0 + FP_1)
		rec = float(TP_0 + TP_1)/(TP_0 + TP_1 + FN_0 + FN_1)
	elif(SENTIMENT_LEVELS==2):
		TP = confusion_mat[0][0]
		TN = confusion_mat[1][1]
		FP = confusion_mat[0][1]
		FN = confusion_mat[1][0]
		acc = float(TP + TN)/total
		pres = float(TP)/(TP + FP)
		rec = float(TP)/(TP + FN)
	else:
		acc = 0
		for i in range(len(confusion_mat)):
			for j in range(len(confusion_mat[i])):
				if i == j:
					acc = acc + confusion_mat[i][j]
		acc = float(acc)/total
	F1 = (2*pres*rec)/(pres + rec)
	return acc, pres, rec, F1

def repport_results(acc, pres, rec, F1, mat):
	print('Accuracy : ' + str(acc))
	print('Precision : ' + str(pres))
	print('Recall : ' + str(rec))
	print('F1 score: ' + str(F1))
	print_mat(mat)

def print_mat(mat):
	for line in mat:
		stream = '  |'
		for val in line:
			stream = stream + ' ' + format(val, '6d') 
		stream = stream + ' |'
		print(stream)

if __name__ == '__main__':
	max_h = 1.0
	step = 0.02
	N = [1, 5, 10] #[i + 1 for i in range(10)]
	start_h_2_levels = 0.5
	start_h_3_levels = 0.3
	start_h_5_levels = 0.2
	'''
	# 2 sentiments
	SENTIMENT_LEVELS = 2
	highest_probs = [start_h_2_levels + i*step for i in range(int((max_h - start_h_2_levels)/step + 0.5))]
	print('================== 2 SENTIMENTS')
	best_model, lemmas_document_freq, N_documents, mean_sentiment_prob_distribution, sentence_sentiment_prob_distribution = train_sentiment_classifier(TRAIN_SET, VALIDATION_SET, highest_probs, balance=True)
	test_confusion_mat = test_sentiment(TEST_SET, best_model, lemmas_document_freq, N_documents, mean_sentiment_prob_distribution, sentence_sentiment_prob_distribution)
	#loader.serialize_structure(best_model, 'best_model')
	print('Test set results: ')
	acc, pres, rec, F1 = evaluate(test_confusion_mat)
	repport_results(acc, pres, rec, F1, test_confusion_mat)
	print('============ UNBALANCED SENTIMENTS')
	best_models, lemmas_document_freq, N_documents, mean_sentiment_prob_distribution, sentence_sentiment_prob_distribution = train(TRAIN_SET, VALIDATION_SET, N, highest_probs, balance=False)
	for i in range(len(N)):
		test_confusion_mat = test(TEST_SET, N[i], best_models[i], lemmas_document_freq, N_documents, mean_sentiment_prob_distribution, sentence_sentiment_prob_distribution)
		#loader.serialize_structure(best_model, 'best_model')
		print('Test set results -> n = ' + str(N[i]))
		acc, pres, rec, F1 = evaluate(test_confusion_mat)
		repport_results(acc, pres, rec, F1, test_confusion_mat)
	print('============ BALANCED SENTIMENTS')
	best_models, lemmas_document_freq, N_documents, mean_sentiment_prob_distribution, sentence_sentiment_prob_distribution = train(TRAIN_SET, VALIDATION_SET, N, highest_probs, balance=True)
	for i in range(len(N)):
		test_confusion_mat = test(TEST_SET, N[i], best_models[i], lemmas_document_freq, N_documents, mean_sentiment_prob_distribution, sentence_sentiment_prob_distribution)
		#loader.serialize_structure(best_model, 'best_model')
		print('Test set results -> n = ' + str(N[i]))
		acc, pres, rec, F1 = evaluate(test_confusion_mat)
		repport_results(acc, pres, rec, F1, test_confusion_mat)
	# 3 sentiments
	print('================== 5 SENTIMENTS')
	SENTIMENT_LEVELS = 5
	highest_probs = [start_h_5_levels + i*step for i in range(int((max_h - start_h_5_levels)/step + 0.5))]
	print('============ UNBALANCED SENTIMENTS')
	best_models, lemmas_document_freq, N_documents, mean_sentiment_prob_distribution, sentence_sentiment_prob_distribution = train(TRAIN_SET, VALIDATION_SET, N, highest_probs, balance=False)
	for i in range(len(N)):
		test_confusion_mat = test(TEST_SET, N[i], best_models[i], lemmas_document_freq, N_documents, mean_sentiment_prob_distribution, sentence_sentiment_prob_distribution)
		#loader.serialize_structure(best_model, 'best_model')
		print('Test set results -> n = ' + str(N[i]))
		acc, pres, rec, F1 = evaluate(test_confusion_mat)
		repport_results(acc, pres, rec, F1, test_confusion_mat)
	print('============ BALANCED SENTIMENTS')
	best_models, lemmas_document_freq, N_documents, mean_sentiment_prob_distribution, sentence_sentiment_prob_distribution = train(TRAIN_SET, VALIDATION_SET, N, highest_probs, balance=True)
	for i in range(len(N)):
		test_confusion_mat = test(TEST_SET, N[i], best_models[i], lemmas_document_freq, N_documents, mean_sentiment_prob_distribution, sentence_sentiment_prob_distribution)
		#loader.serialize_structure(best_model, 'best_model')
		print('Test set results -> n = ' + str(N[i]))
		acc, pres, rec, F1 = evaluate(test_confusion_mat)
		repport_results(acc, pres, rec, F1, test_confusion_mat)
	'''
	highest_probs = [start_h_2_levels + i*step for i in range(int((max_h - start_h_2_levels)/step + 0.5))]
	best_model, lemmas_document_freq, N_documents, mean_sentiment_prob_distribution, sentence_sentiment_prob_distribution, best_h = train_sentiment_classifier (TRAIN_SET, VALIDATION_SET, highest_probs, balance=False)
	print('Best h is: ' + str(best_h))
	for i in range(len(N)):
		test_confusion_mat = test(TEST_SET, N[i], best_model, lemmas_document_freq, N_documents, mean_sentiment_prob_distribution, sentence_sentiment_prob_distribution)
		#loader.serialize_structure(best_model, 'best_model')
		print('Test set results -> n = ' + str(N[i]))
		acc, pres, rec, F1 = evaluate(test_confusion_mat)
		repport_results(acc, pres, rec, F1, test_confusion_mat)
