import math
import threading
from time import *
import numpy as np
from numba import cuda, jit
from multiprocessing import Process

from GPU.genetic.evaluation import evaluate
from GPU.genetic.mutation import mutate
from GPU.genetic.selection import selector
from GPU.utils.utils import shuffle_list

import numpy as np
from numba import jit, cuda
import random

# @jit(nopython=True)
def regenerate_population(population, size_individual):
    initial_individual = np.arange(size_individual)
    for individual in population:
        initial_individual = initial_individual
        np.random.shuffle(individual)

# @jit(nopython=True)
def crossover(population, size_individual):
    for i in range(0, len(population), 2):
        size_crossover = 1 + round(random.random() * (size_individual-2))
        crossover_mutate(population[i], population[i+1], size_individual, size_crossover)


@jit(nopython=True)
def crossover_mutate(individual_a, individual_b, size_individual, size_crossover):
    index_to_split = round(random.random() * (size_individual-1))
    #print("index_to_split : " + str(index_to_split) + " size_crossover : " + str(size_crossover))

    if size_individual >= (index_to_split + size_crossover):
        new_individual_a = np.concatenate((individual_a[:index_to_split], individual_b[index_to_split:(index_to_split + size_crossover)], individual_a[(index_to_split + size_crossover):]))
        new_individual_b = np.concatenate((individual_b[:index_to_split], individual_a[index_to_split:(index_to_split + size_crossover)], individual_b[(index_to_split + size_crossover):]))

        mapping_individual_a = individual_a[index_to_split:(index_to_split + size_crossover)]
        mapping_individual_b = individual_b[index_to_split:(index_to_split + size_crossover)]

        matrix_mapping = generate_matrix_mapping(mapping_individual_a, mapping_individual_b)

        list_gene_mapping = list(range(index_to_split)) + list(range(index_to_split + size_crossover, size_individual))

    else:
        new_individual_a = np.concatenate((individual_b[:(index_to_split + size_crossover - size_individual)], individual_a[(index_to_split + size_crossover - len(individual_a)):index_to_split], individual_b[index_to_split:]))
        new_individual_b = np.concatenate((individual_a[:(index_to_split + size_crossover - size_individual)], individual_b[(index_to_split + size_crossover - len(individual_a)):index_to_split], individual_a[index_to_split:]))

        mapping_individual_a = np.concatenate((individual_a[:(index_to_split + size_crossover - size_individual)], individual_a[index_to_split:]))
        mapping_individual_b = np.concatenate((individual_b[:(index_to_split + size_crossover - size_individual)], individual_b[index_to_split:]))

        matrix_mapping = generate_matrix_mapping(mapping_individual_a, mapping_individual_b)

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
def generate_matrix_mapping(mapping_individual_a, mapping_individual_b):
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



def evolution(population, dataset_score, score_population, size_individual, mutation_rate, spliter, number_population_to_keep, result, iteration, rng_states, threadsperblock, blockspergrid):
    population_cuda = cuda.to_device(population)
    score_population_cuda = cuda.to_device(score_population)
    dataset_score_cuda = cuda.to_device(dataset_score)

    evaluate[blockspergrid, threadsperblock](population_cuda, dataset_score_cuda, score_population_cuda, size_individual)
    score_population[:] = score_population_cuda.copy_to_host()

    selector(population, score_population, number_population_to_keep)

    shuffle_list(population[1:(number_population_to_keep + 1)])

    population_cuda = cuda.to_device(population[(spliter+1):(number_population_to_keep+1)])

    threading.Thread(target=crossover, args=(population[1:(spliter+1)], size_individual,)).start()
    threading.Thread(target=mutate2, args=(blockspergrid, threadsperblock, population_cuda, size_individual, mutation_rate, rng_states,)).start()
    threading.Thread(target=regenerate_population, args=(population[(number_population_to_keep + 1):], size_individual,)).start()
    # threading.Thread(target=test, args=(population,)).start()

    population[(spliter+1):(number_population_to_keep+1)] = population_cuda.copy_to_host()

    if not np.any(result):
        result = np.array([[iteration, score_population[0], population[0]]])
        new_best_individual = True
    elif np.any(result) and result[-1][1] > score_population[0]:
        result = np.concatenate((result, np.array([[iteration, score_population[0], population[0]]])))
        new_best_individual = True
    else:
        new_best_individual = False

    cuda.current_context().deallocations.clear()

    return result, new_best_individual

# @jit(nopython=True)
def test(population):
    print('test')
    print(population)
    population[0][0] = 99
    print(population)



def mutate2(blockspergrid, threadsperblock, population_cuda, size_individual, mutation_rate, rng_states):
    mutate[blockspergrid, threadsperblock](population_cuda, size_individual, mutation_rate, rng_states)

import numpy as np
from numba import cuda
from numba.cuda import random as ra

from GPU.processing_dataset.dataset import dataset_of_traveling_salesman, distance_enter_points
from GPU.genetic.init_population import init_population
from GPU.utils.utils import display

if __name__ == '__main__':
    size_population = 10
    population_rate_to_keep = 0.5
    mutation_rate = 0.1
    cycle = 10000

    duration = 0

    with open('./GPU/dataset/dataset1.txt', 'r') as f:
        dataset = [tuple(map(int, i.replace('(', '').replace(')', '').split(','))) for i in f]
    dataset_score = distance_enter_points(dataset)
    size_individual = len(dataset_score)
    number_population_to_keep = int(round(population_rate_to_keep * size_population))
    spliter = int(round(number_population_to_keep)/2) if int(round(number_population_to_keep)/2)%2 == 0 else int(round(number_population_to_keep)/2)+1

    population = np.array(init_population(size_individual, size_population))
    score_population = np.array([0.] * size_population)
    result = np.array([[],[],[]])

    threadsperblock =10
    blockspergrid = math.ceil(size_population / threadsperblock)
    nthreads = threadsperblock
    rng_states = ra.create_xoroshiro128p_states(nthreads, seed=time_ns())

    for iteration in range(cycle):
        time_iteration_start = time()
        print("---" + str(iteration) + "---")
        print(population)
        print(id(population))
        result, new_best_individual = evolution(population, dataset_score, score_population, size_individual, mutation_rate, spliter, number_population_to_keep, result, iteration, rng_states, threadsperblock, blockspergrid)
        duration += time() - time_iteration_start
        print(population)
        if iteration == 3:
            exit()
        print(time() - time_iteration_start)
        display(dataset, result, iteration, cycle, new_best_individual)
    print(population)
    print(duration)