from GeneticOptimizer import GeneticOptimizer
from nltk.tokenize import sent_tokenize
from Text import JS_DIVERGENCE, compute_tf

sentences1 = []
sentences2 = []
sentences3 = []
sentences4 = []
sentences5 = []


def get_Data():
	global sentences1
	global sentences2
	global sentences3
	global sentences4
	global sentences5
	f = open('data/One')
	text = f.read()
	sentences1.append(sent_tokenize(text))
	f = open('data/Two')
	text = f.read()
	sentences2.append(sent_tokenize(text))
	f = open('data/Three')
	text = f.read()
	sentences3.append(sent_tokenize(text))
	f = open('data/Four')
	text = f.read()
	sentences4.append(sent_tokenize(text))
	f = open('data/Five')
	text = f.read()
	sentences5.append(sent_tokenize(text))

	sentences1 = [y for x in sentences1 for y in x]
	sentences2 = [y for x in sentences2 for y in x]
	sentences3 = [y for x in sentences3 for y in x]
	sentences4 = [y for x in sentences4 for y in x]
	sentences5 = [y for x in sentences5 for y in x]

def GenAlgo(docs, length_max, epoch, population_size=1000):
	sentences = []
	for title, doc in docs:
		# sentences.append(title)
		sentences.extend(doc)

	doc_freq = compute_tf(sentences)
	
	gen_optimizer = GeneticOptimizer(func_fit=JS_DIVERGENCE,
                                     docs=docs,
                                     docs_representation = doc_freq,
                                     max_length=length_max,
                                     population_size=population_size,
                                     survival_rate=0.4,
                                     mutation_rate=0.2,
                                     reproduction_rate=0.4,
                                     maximization=False)

	return gen_optimizer.evolve(epoch)

if __name__ == '__main__':
	get_Data()
	doc_1 = ("First", sentences1)
	doc_2 = ("Second", sentences2)
	doc_3 = ("Third", sentences3)
	doc_4 = ("Fourth", sentences4)
	doc_5 = ("Fifth", sentences5)
	docs = [doc_1, doc_2, doc_3, doc_4, doc_5]

	length_max = 100
	epoch = 10
	# length_max = 100
	# epoch = 1000
	population_size = 50
	print ("Genetic Algorithm example:")
	print (GenAlgo(docs, length_max, epoch, population_size))