import math
from time import time, time_ns

import numpy as np
from numba import cuda, jit
import numba
from numba.cuda import random
import random as rd

from GPU.genetic.init_population import init_population
from GPU.processing_dataset.dataset import distance_enter_points


@jit(nopython=True)
def regenerate_population1(population):
    for individual in population:
        np.random.shuffle(individual)

@jit(nopython=True, parallel=True, fastmath=True)
def regenerate_population1_1(population, size_population):
    size_population_1 = len(population)
    for i in numba.prange(size_population_1):
        np.random.shuffle(population[i])


@jit(nopython=True)
def regenerate_population2(population, size_individual):
    population[:] = generate_list_individual2(len(population), size_individual)

def regenerate_population3(population, size_individual, blockspergrid, threadsperblock):
    new_population = generate_list_individual2(len(population), size_individual)
    population_cuda = cuda.to_device(population)
    copy_population[blockspergrid, threadsperblock](population_cuda, cuda.to_device(new_population))
    population[:] = population_cuda.copy_to_host()


@jit(nopython=True, parallel=True)
def generate_list_individual2(size_population, size_individual):
    standard_individual = np.arange(size_individual)
    population = standard_individual.repeat(size_population).reshape(-1, size_population).T
    [np.random.shuffle(x) for x in population]
    return population


@cuda.jit
def copy_population(population, new_population):
    x, y = cuda.grid(2)
    population[x, y] = new_population[x, y]

with open('./GPU/dataset/dataset2.txt', 'r') as f:
    dataset = [tuple(map(int, i.replace('(', '').replace(')', '').split(','))) for i in f]

size_population = 100000
dataset_score = distance_enter_points(dataset)
size_individual = len(dataset_score)
population = np.array(init_population(size_individual, size_population))

threadsperblock = 256
blockspergrid_x = math.ceil(size_population / threadsperblock)
blockspergrid = blockspergrid_x
nthreads = threadsperblock

# print(population)
time_iteration_start = time()
regenerate_population1(population)
print('Algo 1 :' + str(time() - time_iteration_start))
#print(population)

time_iteration_start = time()
regenerate_population1(population)
print('Algo 1 V2:' + str(time() - time_iteration_start))
# print(population)

time_iteration_start = time()
regenerate_population1_1(population, size_population)
print('Algo 1 1 :' + str(time() - time_iteration_start))
#print(population)

time_iteration_start = time()
regenerate_population1_1(population, size_population)
print('Algo 1 1 V2:' + str(time() - time_iteration_start))
# print(population)

time_iteration_start = time()
regenerate_population2(population, size_individual)
print('Algo 2 :' + str(time() - time_iteration_start))
#print(population)

time_iteration_start = time()
regenerate_population2(population, size_individual)
print('Algo 2 V2:' + str(time() - time_iteration_start))
#print(population)

time_iteration_start = time()
regenerate_population3(population, size_individual, blockspergrid, threadsperblock)
print('Algo 3 :' + str(time() - time_iteration_start))
#print(population)

time_iteration_start = time()
regenerate_population3(population, size_individual, blockspergrid, threadsperblock)
print('Algo 3 V2:' + str(time() - time_iteration_start))
#print(population)
