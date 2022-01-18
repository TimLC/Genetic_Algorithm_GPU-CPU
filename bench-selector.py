import math
from time import time, time_ns

import numpy as np
from numba import cuda
from numba.cuda import random
import random as rd

from CPU.genetic.evaluation import evaluate
from GPU.genetic.init_population import init_population
from GPU.processing_dataset.dataset import distance_enter_points

import numpy as np
from numba import jit, cuda, prange
import random

@jit(nopython=True)
def selector1(population, score_population, size_population_to_keep):
    list_sort = sorted(zip(population.copy(), score_population.copy()), key=lambda pair: pair[1])

    for index, value in enumerate(list_sort):
        population[index] = value[0]
        score_population[index] = value[1]

    population[size_population_to_keep] = population[0]

@jit(nopython=True, parallel=True, fastmath=True)
def selector1_1(population, score_population, size_population_to_keep):
    list_sort = sorted(zip(population.copy(), score_population.copy()), key=lambda pair: pair[1])

    for index in prange(len(population)):
        population[index] = list_sort[index][0]
        score_population[index] = list_sort[index][1]

    population[size_population_to_keep] = population[0]

@jit(nopython=True)
def selector3(population, score_population):
    list_sort = sorted(zip(population.copy(), score_population.copy()), key=lambda pair: pair[1])
    selector3_plus(population, score_population, list_sort)

@jit(nopython=True, parallel=True)
def selector3_plus(population, score_population, list_sort):
    for index in prange(len(population)):
        population[index] = list_sort[index][0]
        score_population[index] = list_sort[index][1]

def selector2(population, score_population):
    population_and_score = list(zip(population, score_population))
    population_and_score_sorted = sorted(population_and_score, key=lambda individual_and_score: individual_and_score[1])
    population_sorted_by_score, score = list(zip(*population_and_score_sorted))
    return list(population_sorted_by_score), list(score)


with open('./GPU/dataset/dataset2.txt', 'r') as f:
    dataset = [tuple(map(int, i.replace('(','').replace(')','').split(','))) for i in f]

size_population = 8192
size_population_to_keep = round(size_population/2)
dataset_score = distance_enter_points(dataset)
size_individual = len(dataset_score)
population = np.array(init_population(size_individual, size_population))
score_population = np.array(evaluate(population, dataset_score))

time_iteration_start = time()
selector1(population, score_population, size_population_to_keep)
print('Algo 1 :' + str(time() - time_iteration_start))

time_iteration_start = time()
selector1(population, score_population, size_population_to_keep)
print('Algo 1 V2:' + str(time() - time_iteration_start))

time_iteration_start = time()
selector1_1(population, score_population, size_population_to_keep)
print('Algo 1 1:' + str(time() - time_iteration_start))

time_iteration_start = time()
selector1_1(population, score_population, size_population_to_keep)
print('Algo 1 1 V2:' + str(time() - time_iteration_start))

time_iteration_start = time()
selector3(population, score_population)
print('Algo 3 :' + str(time() - time_iteration_start))

time_iteration_start = time()
selector3(population, score_population)
print('Algo 3 V2:' + str(time() - time_iteration_start))

population = init_population(size_individual, size_population)
score_population = evaluate(population, dataset_score)

time_iteration_start = time()
selector2(population, score_population)
print('Algo 2:' + str(time() - time_iteration_start))

time_iteration_start = time()
selector2(population, score_population)
print('Algo 2 V2:' + str(time() - time_iteration_start))