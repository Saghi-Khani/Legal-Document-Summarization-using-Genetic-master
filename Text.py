
from nlp_utils import *

import math
import numpy as np

from nltk.util import ngrams
from nltk.stem import WordNetLemmatizer
wordnet_lemmatizer = WordNetLemmatizer()

def lemmatize_content(sentences, N=1):
	words = []
	for s in sentences:
		words.extend([wordnet_lemmatizer.lemmatize(r) for r in tokenizer.tokenize(s)])
	if N == 1:
		content_words = [w for w in words if w not in stopset]
	normalized_content_words = map(normalize_word, content_words)
	if N > 1:
		return [G for G in ngrams(normalized_content_words, N)]
	return normalized_content_words

def stem_content(sentences, N=1):
	def is_ngram(g):
		for a in g:
			if not(a in stopset):
				return True
		return False

	words = []
	for s in sentences:
		words.extend([stemmer.stem(r) for r in tokenizer.tokenize(s)])

	if N == 1:
		content_words = [w for w in words if w not in stopset]
	else:
		content_words = words

	normalized_content = map(normalize_word, content_words)
	if N > 1:
		return [gram for gram in ngrams(normalized_content, N) if is_ngram(gram)]
	return normalized_content

def get_content(sentences, N=1):
	words = []
	for s in sentences:
		words.extend(tokenizer.tokenize(s))
	content = [w for w in words if w not in stopset]
	normalized_content = map(normalize_word, content)
	if N > 1:
		return [G for G in ngrams(normalized_content, N)]
	return normalized_content

def get_words_sentence(sentence):
	words = tokenizer.tokenize(sentence)
	return [w for w in words if w not in stopset]

def KL_DIVERGENCE(frequency_summary, frequency_doc):
	sum_val = 0
	# print(frequency_summary)
	for w, f in frequency_summary.items():
		if w in frequency_doc:
			sum_val += f * math.log(f / float(frequency_doc[w]))
	
	return sum_val

# def compute_tf_doc(docs, N=1):
# 	sentences = []
# 	for title, doc in docs:
# 		sentences.append(title)
# 		sentences.extend(doc)

# 	content_words = list(set(stem_content(sentences, N)))
# 	docs_words = []
# 	for title, doc in docs:
# 		s_tmp = [title]
# 		s_tmp.extend(doc)
# 		docs_words.append(stem_content(s_tmp, N))

# 	word_freq = {}
# 	for w in content_words:
# 		w_score = 0
# 		for d in docs_words:
# 			if w in d:
# 				w_score += 1
# 		if w_score != 0:
# 			word_freq[w] = w_score

# 	content_word_tf = dict((w, f / float(len(word_freq.keys()))) for w, f in word_freq.items())
# 	return content_word_tf

def get_tf(words):
	word_freq = {}
	for w in words:
		word_freq[w] = word_freq.get(w, 0) + 1
	return word_freq

def compute_tf(sentences, N=1):
	content = list(stem_content(sentences, N))
	content_count = len(content)
	content_freq = get_tf(content)

	content_word_tf = dict((w, f / float(content_count)) for w, f in content_freq.items())
	return content_word_tf

def find_avg_freq(freq_1, freq_2):
	average_freq = {}

	keys = set(freq_1.keys()) | set(freq_2.keys())

	for k in keys:
		s_1 = freq_1.get(k, 0)
		s_2 = freq_2.get(k, 0)

		average_freq[k] = (s_1 + s_2) / 2.

	return average_freq

def JS_DIVERGENCE(sys_summary, doc_freq):
	summary_freq = compute_tf(sys_summary)
	average_freq = find_avg_freq(summary_freq, doc_freq)

	jsd = KL_DIVERGENCE(summary_freq, average_freq) + KL_DIVERGENCE(doc_freq, average_freq)
	return jsd / 2.

