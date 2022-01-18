import math
from time import time

import numpy as np
from numba import cuda

from GPU.genetic.init_population import init_population
from GPU.processing_dataset.dataset import distance_enter_points


@cuda.jit
def evaluate1(population, dataset_score, score_population, size_individual):
    for index, individual in enumerate(population):
        score_population[index] = loss_function(individual, dataset_score, size_individual)

@cuda.jit
def evaluate2(population, dataset_score, score_population, size_individual):
    index = cuda.grid(1)
    score_population[index] = loss_function2(population[index], dataset_score, size_individual)

def evaluate3(population, dataset_score):
    score_population = [None] * len(population)
    for index, individual in enumerate(population):
        score_population[index] = loss_function1(individual, dataset_score)
    return score_population

@cuda.jit(device=True)
def loss_function(individual, dataset_score, size_individual):
    score = 0
    for index in range(size_individual):
        if index < size_individual - 1:
            score += dataset_score[individual[index]][individual[index+1]]
        else:
            score += dataset_score[individual[0]][individual[index]]
    return score

@cuda.jit(device=True)
def loss_function2(individual, dataset_score, size_individual):
    score = 0
    for index in range(size_individual):
        if index < size_individual - 1:
            score += dataset_score[individual[index]][individual[index+1]]
        else:
            score += dataset_score[individual[0]][individual[index]]
    return score

def loss_function1(individual, dataset_score):
    score = 0
    for index in range(len(individual)):
        if index < len(individual) - 1:
            score += dataset_score[individual[index]][individual[index+1]]
        else:
            score += dataset_score[individual[0]][individual[index]]
    return score

with open('./GPU/dataset/dataset2.txt', 'r') as f:
    dataset = [tuple(map(int, i.replace('(','').replace(')','').split(','))) for i in f]

size_population = 10000
dataset_score = distance_enter_points(dataset)
size_individual = len(dataset_score)
population = np.array(init_population(size_individual, size_population))

population_cuda = cuda.to_device(population)
dataset_score = cuda.to_device(dataset_score)

threadsperblock = 256
blockspergrid = math.ceil(size_population / threadsperblock)

score_population = cuda.to_device(np.array([0.] * size_population))
time_iteration_start = time()
evaluate1[blockspergrid, threadsperblock](population, dataset_score, score_population, size_individual)
print('Algo 1 :' + str(time() - time_iteration_start))
print(np.array(score_population)[:20])

score_population = cuda.to_device(np.array([0.] * size_population))
time_iteration_start = time()
evaluate1[blockspergrid, threadsperblock](population, dataset_score, score_population, size_individual)
print('Algo 1 V2:' + str(time() - time_iteration_start))
print(np.array(score_population)[:20])

score_population = cuda.to_device(np.array([0.] * size_population))
time_iteration_start = time()
evaluate2[blockspergrid, threadsperblock](population, dataset_score, score_population, size_individual)
print('Algo 2 :' + str(time() - time_iteration_start))
print(np.array(score_population)[:20])

score_population = cuda.to_device(np.array([0.] * size_population))
time_iteration_start = time()
evaluate2[blockspergrid, threadsperblock](population, dataset_score, score_population, size_individual)
print('Algo 2 V2:' + str(time() - time_iteration_start))
print(np.array(score_population)[:20])

score_population = cuda.to_device(np.array([0.] * size_population))
time_iteration_start = time()
evaluate1[blockspergrid, threadsperblock](population, dataset_score, score_population, size_individual)
print('Algo 1 V3:' + str(time() - time_iteration_start))
print(np.array(score_population)[:20])

score_population = cuda.to_device(np.array([0.] * size_population))
time_iteration_start = time()
evaluate2[blockspergrid, threadsperblock](population, dataset_score, score_population, size_individual)
print('Algo 3 V2:' + str(time() - time_iteration_start))
print(np.array(score_population)[:20])

population = np.array(init_population(size_individual, size_population))
dataset_score = distance_enter_points(dataset)

time_iteration_start = time()
evaluate3(population, dataset_score)
print('Algo 3:' + str(time() - time_iteration_start))

time_iteration_start = time()
evaluate3(population, dataset_score)
print('Algo 3 V2:' + str(time() - time_iteration_start))