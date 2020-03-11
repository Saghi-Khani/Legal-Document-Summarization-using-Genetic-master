import numpy as np
import random
from copy import deepcopy

from greedy import greedy_optimizer

from nltk.tokenize import RegexpTokenizer
tokenizer = RegexpTokenizer(r'\w+')

#func_fit: fitness function
#documents: documents
#rep_documents: rep_documents
#max_lenghth: len_max
#size_pop: size_pop
#rate_of_survival: rate_of_survival
#mutation_rate: rate_of_mutation
#is_max: is_max

class GeneticOptimizer(object):
    def __init__(self, func_fit, documents, rep_documents, len_max, size_pop, rate_of_survival, mutation_rate, rate_of_mutation, is_max=False):
        np.random.seed(123)

        self._fitness_fun = func_fit
        self._size_pop = size_pop
        self._rate_of_survival = rate_of_survival
        self._mutation_rate = mutation_rate
        self._rate_of_mutation = rate_of_mutation
        self._is_max = is_max

        self._documents = documents
        self._rep_documents = rep_documents
        self._len_max = len_max
        
        self._sentences = []
        self._sentence_tokens = []
        for title, doc in documents:
            self._sentences.append(title)
            self._sentence_tokens.append(tokenizer.tokenize(title))
            self._sentences.extend(doc)
            for s in doc:
                self._sentence_tokens.append(tokenizer.tokenize(s))

    def create_individual(self):
        random_scores = np.random.rand(len(self._sentences))
        scored_sentences = zip(self._sentences, random_scores)
        sorted_sentences = sorted(scored_sentences, key=lambda tup: tup[1], reverse=True)
        return greedy_optimizer(sorted_sentences, self._len_max)

    def create_population(self, n):
        population = []
        for i in range(n):
            population.append(self.create_individual())
        return population

    def scoring_population(self, population):
        scored_population = []
        for individual in population:
            # score = self._fitness_fun(individual, self._documents)
          
            score = self._fitness_fun(individual, self._rep_documents)
            scored_population.append((individual, score))

        return scored_population

    def survivors_selection(self, scored_population):
        sorted_population = sorted(scored_population, key=lambda tup: tup[1], reverse=self._is_max)

        percentage_winner = 0.5

        to_keep = int(self._rate_of_survival * self._size_pop)
        number_winners = int(percentage_winner * to_keep)
        winners = [tup[0] for tup in sorted_population[:number_winners]]
        
        losers = sorted_population[number_winners:]

        number_losers = int((1 - percentage_winner) * to_keep) 

        survivors = deepcopy(winners)
        random_scores = np.random.rand(len(losers))

        sorted_losers = sorted(zip(losers, random_scores), key=lambda tup: tup[1])
        loser_survivors = [tup[0][0] for tup in sorted_losers[:number_losers]]

        survivors.extend(loser_survivors)
        return survivors, winners

    def create_new_generation(self, scored_population):
        new_generation, winners = self.survivors_selection(scored_population)
        new_generation = self.mutation(new_generation)
        new_generation.extend(self.reproduction(winners, len(new_generation)))
        individuals_to_create = self._size_pop - len(new_generation)
        new_generation.extend(self.create_population(individuals_to_create))
        
        return new_generation

    def individual_length_calculation(self, individual):
        len_ = 0
        for sentence in individual:
            len_ += len(tokenizer.tokenize(sentence))
        return len_

    def mutation(self, population, mutation_rate="auto"):
        if mutation_rate == "auto":
            mutation_rate = self._mutation_rate

        nb_mutant = int(mutation_rate * len(population))

        random_scores = np.random.rand(len(population))
        sorted_population = sorted(zip(population, random_scores), key=lambda tup: tup[1])
        mutants = [tup[0] for tup in sorted_population[:nb_mutant]]

        mutated = []
        i = 0
        for mutant in mutants:
            tomutation = deepcopy(mutant)

            sentence_to_remove = random.choice(tomutation)
            idx = tomutation.index(sentence_to_remove)
            del tomutation[idx]

            available_size = self._len_max - self.individual_length_calculation(tomutation)
        
            available_sentences = [s[0] for s in zip(self._sentences, self._sentence_tokens) if len(s[1]) <= available_size]
            if available_sentences != []:
                i += 1
                sentence_to_add = random.choice(available_sentences)
                tomutation.append(sentence_to_add)
                
                mutated.append(tomutation)
        
        population.extend(mutated)
        return population

    def reproduction(self, population_winners, size_pop, rate_of_mutation="auto"):
        if rate_of_mutation == "auto":
            rate_of_mutation = self._rate_of_mutation

        parents = []
        number_families = int(rate_of_mutation * size_pop)
    
        for i in range(number_families):
            parents.append(random.sample(population_winners, 2))

        children = []
        for father, mother in parents:
            genetic_pool = [s for s in self._sentences if s in father]
            genetic_pool.extend([s for s in self._sentences if s in mother])

            random_scores = np.random.rand(len(genetic_pool))

            scored_sentences = zip(self._sentences, random_scores)
            sorted_sentences = sorted(scored_sentences, key=lambda tup: tup[1], reverse=True)
            child = greedy_optimizer(sorted_sentences, self._len_max)
            children.append(child)

        return children

    def population_intitialization(self):
        population_intitialization = self.create_population(self._size_pop)
        # print ("initial population len:", len(population_intitialization))
        return population_intitialization

    def compare(self, scored_individual, best_scored_individual):
        if self._is_max:
            return scored_individual[1] > best_scored_individual[1]
        return scored_individual[1] < best_scored_individual[1]

    def implementation(self, epoch):
        population = self.population_intitialization()
        if self._is_max:
            best_individual = (None, -10000)
        else:
            best_individual = (None, 10000)
        for i in range(epoch):
            # print ("Iteration: ", i, " -- best individual: ", best_individual[0])
            scored_population = self.scoring_population(population)
            sorted_population = sorted(scored_population, key=lambda tup: tup[1], reverse=self._is_max)
            best_individual_in_generation = sorted_population[0]

            if self.compare(best_individual_in_generation, best_individual):
                best_individual = best_individual_in_generation

            population = self.create_new_generation(scored_population)
        
        return best_individual




