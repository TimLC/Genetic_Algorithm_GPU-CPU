import math
from time import time, time_ns

import numpy as np
from numba import cuda
from numba.cuda import random
import random as rd

from GPU.genetic.init_population import init_population
from GPU.processing_dataset.dataset import distance_enter_points

import numpy as np
from numba import jit, cuda, prange
import random


@jit(nopython=True)
def crossover1(population, size_individual):
    for i in range(0, len(population), 2):
        size_crossover = round(random.random() * (size_individual-2))
        crossover_mutate1(population[i], population[i+1], size_individual, size_crossover)

@jit(nopython=True, parallel=True, fastmath=True)
def crossover3(population, size_individual):
    for i in prange(int(len(population)/2)):
        size_crossover = round(random.random() * (size_individual-2))
        crossover_mutate1(population[i*2], population[(i*2)+1], size_individual, size_crossover)


@jit(nopython=True)
def crossover_mutate1(individual_a, individual_b, size_individual, size_crossover):
    index_to_split = round(random.random() * (size_individual-1))

    if size_individual >= (index_to_split + size_crossover):
        new_individual_a = np.concatenate((individual_a[:index_to_split], individual_b[index_to_split:(index_to_split + size_crossover)], individual_a[(index_to_split + size_crossover):]))
        new_individual_b = np.concatenate((individual_b[:index_to_split], individual_a[index_to_split:(index_to_split + size_crossover)], individual_b[(index_to_split + size_crossover):]))

        mapping_individual_a = individual_a[index_to_split:(index_to_split + size_crossover)]
        mapping_individual_b = individual_b[index_to_split:(index_to_split + size_crossover)]

        matrix_mapping = generate_matrix_mapping1(mapping_individual_a, mapping_individual_b)

        list_gene_mapping = list(range(index_to_split)) + list(range(index_to_split + size_crossover, size_individual))

    else:
        new_individual_a = np.concatenate((individual_b[:(index_to_split + size_crossover - size_individual)], individual_a[(index_to_split + size_crossover - len(individual_a)):index_to_split], individual_b[index_to_split:]))
        new_individual_b = np.concatenate((individual_a[:(index_to_split + size_crossover - size_individual)], individual_b[(index_to_split + size_crossover - len(individual_a)):index_to_split], individual_a[index_to_split:]))

        mapping_individual_a = np.concatenate((individual_a[:(index_to_split + size_crossover - size_individual)], individual_a[index_to_split:]))
        mapping_individual_b = np.concatenate((individual_b[:(index_to_split + size_crossover - size_individual)], individual_b[index_to_split:]))

        matrix_mapping = generate_matrix_mapping1(mapping_individual_a, mapping_individual_b)

        list_gene_mapping = list(range((index_to_split + size_crossover - size_individual), index_to_split))

    if matrix_mapping:
        for i in list_gene_mapping:
            if np.count_nonzero(new_individual_a == new_individual_a[i]) == 2:
                new_individual_a[i] = [value_matrix[1] for value_matrix in matrix_mapping if new_individual_a[i] == value_matrix[0]][0]
            if np.count_nonzero(new_individual_b == new_individual_b[i]) == 2:
                new_individual_b[i] = [value_matrix[1] for value_matrix in matrix_mapping if new_individual_b[i] == value_matrix[0]][0]
    individual_a[:] = new_individual_a
    individual_b[:] = new_individual_b


@jit(nopython=True)
def generate_matrix_mapping1(mapping_individual_a, mapping_individual_b):
    matrix_mapping = list(zip(mapping_individual_a, mapping_individual_b))
    while True:
        list_del = []
        for value_matrix in matrix_mapping:
            index = matrix_mapping.index(value_matrix)
            mapper_b = [j for _, j in matrix_mapping]
            if value_matrix[0] in mapper_b:
                if not index in list_del:
                    position = mapper_b.index(value_matrix[0])
                    matrix_mapping[index] = (matrix_mapping[position][0], matrix_mapping[index][1])
                    list_del.append(position)
        if list_del:
            list_del = sorted(list_del, reverse=True)
            for idx in list_del:
                matrix_mapping.pop(idx)
        else:
            break
    matrix_mapping = matrix_mapping + list(map(lambda x: (x[1], x[0]), matrix_mapping))
    return matrix_mapping



import random


def crossover2(population, size_individual):
    for i in range(0, len(population), 2):
        size_crossover = round(random.random() * (size_individual-2))
        population[i:(i+2)].append(crossover_mutate2(population[i], population[i+1], size_crossover))
    return population

def crossover_mutate2(individual_a, individual_b, size_crossover):
    index_to_split = round(random.random() * len(individual_a))

    if len(individual_a) >= (index_to_split + size_crossover):
        new_individual_a = individual_a[:index_to_split] + individual_b[index_to_split:(index_to_split + size_crossover)] + individual_a[(index_to_split + size_crossover):]
        new_individual_b = individual_b[:index_to_split] + individual_a[index_to_split:(index_to_split + size_crossover)] + individual_b[(index_to_split + size_crossover):]

        mapping_individual_a = individual_b[index_to_split:(index_to_split + size_crossover)]
        mapping_individual_b = individual_a[index_to_split:(index_to_split + size_crossover)]

        matrix_mapping = generate_matrix_mapping2(mapping_individual_a, mapping_individual_b)

        list_gene_mapping = list(range(index_to_split)) + list(range(index_to_split + size_crossover, len(individual_a)))

    else:
        new_individual_a = individual_b[:(index_to_split + size_crossover - len(individual_a))] + individual_a[(index_to_split + size_crossover - len(individual_a)):index_to_split] + individual_b[index_to_split:]
        new_individual_b = individual_a[:(index_to_split + size_crossover - len(individual_a))] + individual_b[(index_to_split + size_crossover - len(individual_a)):index_to_split] + individual_a[index_to_split:]

        mapping_individual_a = individual_a[:(index_to_split + size_crossover - len(individual_a))] + individual_a[index_to_split:]
        mapping_individual_b = individual_b[:(index_to_split + size_crossover - len(individual_a))] + individual_b[index_to_split:]

        matrix_mapping = generate_matrix_mapping2(mapping_individual_a, mapping_individual_b)

        list_gene_mapping = list(range((index_to_split + size_crossover - len(individual_a)),index_to_split))

    for i in list_gene_mapping:
        if new_individual_a.count(new_individual_a[i]) == 2:
            new_individual_a[i] = [value_matrix[1] for value_matrix in matrix_mapping if new_individual_a[i] == value_matrix[0]][0]
        if new_individual_b.count(new_individual_b[i]) == 2:
            new_individual_b[i] = [value_matrix[1] for value_matrix in matrix_mapping if new_individual_b[i] == value_matrix[0]][0]
    return new_individual_a, new_individual_b


def generate_matrix_mapping2(mapping_individual_a, mapping_individual_b):
    matrix_mapping = list(zip(mapping_individual_a, mapping_individual_b))
    while True:
        list_del = []
        for value_matrix in matrix_mapping:
            index = matrix_mapping.index(value_matrix)
            mapper_b = [j for _, j in matrix_mapping]
            if value_matrix[0] in mapper_b:
                if not index in list_del:
                    position = mapper_b.index(value_matrix[0])
                    matrix_mapping[index] = (matrix_mapping[position][0], matrix_mapping[index][1])
                    list_del.append(position)
        if list_del:
            list_del = sorted(list_del, reverse=True)
            for idx in list_del:
                matrix_mapping.pop(idx)
        else:
            break
    matrix_mapping = matrix_mapping + list(map(lambda x: (x[1], x[0]), matrix_mapping))
    return matrix_mapping






with open('./GPU/dataset/dataset2.txt', 'r') as f:
    dataset = [tuple(map(int, i.replace('(','').replace(')','').split(','))) for i in f]

size_population = 8192
dataset_score = distance_enter_points(dataset)
size_individual = len(dataset_score)
population = np.array(init_population(size_individual, size_population))

population_cuda = cuda.to_device(population)
dataset_score = cuda.to_device(dataset_score)

# threadsperblock = (32, 32)
# blockspergrid_x = math.ceil(size_population / threadsperblock[0])
# blockspergrid_y = math.ceil(size_individual / threadsperblock[1])
# blockspergrid = (blockspergrid_x, blockspergrid_y)
# nthreads = threadsperblock[0] * threadsperblock[1]
# rng_states = random.create_xoroshiro128p_states(nthreads, seed=time_ns())

time_iteration_start = time()
crossover1(population, size_individual)
print('Algo 1 :' + str(time() - time_iteration_start))

time_iteration_start = time()
crossover1(population, size_individual)
print('Algo 1 V2:' + str(time() - time_iteration_start))

time_iteration_start = time()
crossover3(population, size_individual)
print('Algo 3 :' + str(time() - time_iteration_start))

time_iteration_start = time()
crossover3(population, size_individual)
print('Algo 3 V2:' + str(time() - time_iteration_start))
print('Test')
population = init_population(size_individual, size_population)

time_iteration_start = time()
crossover2(population, size_individual)
print('Algo 2:' + str(time() - time_iteration_start))

time_iteration_start = time()
crossover2(population, size_individual)
print('Algo 2 V2:' + str(time() - time_iteration_start))