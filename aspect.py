import loader
import copy
import math
import nltk
import operator

def tf_idf(fwd, fwD, D):
	return fwd*math.log(D/fwD, 10)

def compute_vocab_tf_idf(in_file, serialize=True):
	Fwd = loader.load('yelp/' + in_file)
	FwD = {}
	D = len(Fwd)
	print("Processing lemma frequencies per document")
	for business in Fwd:
		for lemma in Fwd[business]:
			if lemma in FwD:
				FwD[lemma] = FwD[lemma] + 1
			else:
				FwD[lemma] = 1	
	print("Processing tf-idf")
	for business in Fwd:
		for lemma in Fwd[business]:
			# print(lemma)
			Fwd[business][lemma] = tf_idf(Fwd[business][lemma], FwD[lemma], D)
	if(serialize):
		loader.serialize_structure(Fwd, 'yelp/tf-idf')
	return Fwd


def compute_max_tfidf(Wd, limit, serialize=True):
	for business in Wd:
		max_tfidf = []
		words = []
		for lemma in Wd[business]:
			if len(max_tfidf) < limit:
				max_tfidf.append(Wd[business][lemma])
				words.append(lemma)
			else:
				index, value = min(enumerate(max_tfidf))
				#print(index, value)
				if Wd[business][lemma] > value:
					max_tfidf[index] = Wd[business][lemma]
					words[index] = lemma
		Wd[business] = words
	if(serialize):
		loader.serialize_structure(Wd, 'yelp/max_tfidf')
	return Wd

if __name__ == '__main__':
	tfidf = compute_vocab_tf_idf('BUSINESSLEMMASFREQ')
	#tfidf = loader.load_add_to_dict('tf-idfNN')
	tfidf = compute_max_tfidf(tfidf, limit=5, serialize=True)
	i = 0
	for business in tfidf:
		print('============' + business)
		print(tfidf[business])
		i = i + 1
		if i > 10:
			break