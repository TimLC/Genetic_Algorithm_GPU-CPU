import math
from time import time, time_ns

import numpy as np
from numba import cuda
from numba.cuda import random
import random as rd

from GPU.genetic.init_population import init_population
from GPU.processing_dataset.dataset import distance_enter_points


@cuda.jit
def mutate1(population, size_individual, mutation_rate, rng_states):
    index = cuda.grid(1)
    individual_mutate1(population[index], size_individual, mutation_rate, rng_states)

@cuda.jit(device=True)
def individual_mutate1(individual, size_individual, mutation_rate, rng_states):
    thread_id = cuda.grid(1)
    for position_1 in range(size_individual):
        if random.xoroshiro128p_uniform_float32(rng_states, thread_id) < mutation_rate:
            position_2 = round(random.xoroshiro128p_uniform_float32(rng_states, thread_id) * (size_individual - 1))
            if not position_1 == position_2:
                swap_value = individual[position_1]
                individual[position_1] = individual[position_2]
                individual[position_2] = swap_value

@cuda.jit
def mutate2(population, size_individual, mutation_rate, rng_states):
    index = cuda.grid(1)
    individual_mutate2(population[index], size_individual, mutation_rate, rng_states, index)

@cuda.jit(device=True)
def individual_mutate2(individual, size_individual, mutation_rate, rng_states, thread_id):
    for position_1 in range(size_individual):
        if random.xoroshiro128p_uniform_float32(rng_states, thread_id) < mutation_rate:
            position_2 = round(random.xoroshiro128p_uniform_float32(rng_states, thread_id) * (size_individual - 1))
            if not position_1 == position_2:
                swap_value = individual[position_1]
                individual[position_1] = individual[position_2]
                individual[position_2] = swap_value

def mutate3(population, size_individual, mutation_rate):
    return list(map(lambda individual: individual_mutate3(individual, mutation_rate), population))


def individual_mutate3(individual, mutation_rate):
    for position_1 in range(len(individual)):
        if rd.random() < mutation_rate:
            position_2 = round(rd.random() * (len(individual)-1))
            if not position_1 == position_2:
                swap_value = individual[position_1]
                individual[position_1] = individual[position_2]
                individual[position_2] = swap_value
    return individual


with open('./GPU/dataset/dataset2.txt', 'r') as f:
    dataset = [tuple(map(int, i.replace('(','').replace(')','').split(','))) for i in f]

size_population = 100000
mutation_rate = 0.1
dataset_score = distance_enter_points(dataset)
size_individual = len(dataset_score)
population = np.array(init_population(size_individual, size_population))

population_cuda = cuda.to_device(population)
dataset_score = cuda.to_device(dataset_score)

threadsperblock = 256
blockspergrid = math.ceil(size_population / threadsperblock)
nthreads = threadsperblock
rng_states = random.create_xoroshiro128p_states(nthreads, seed=time_ns())

score_population = cuda.to_device(np.array([0.] * size_population))
time_iteration_start = time()
mutate1[blockspergrid, threadsperblock](population, size_individual, mutation_rate, rng_states)
print('Algo 1 :' + str(time() - time_iteration_start))

score_population = cuda.to_device(np.array([0.] * size_population))
time_iteration_start = time()
mutate1[blockspergrid, threadsperblock](population, size_individual, mutation_rate, rng_states)
print('Algo 1 V2:' + str(time() - time_iteration_start))

score_population = cuda.to_device(np.array([0.] * size_population))
time_iteration_start = time()
mutate2[blockspergrid, threadsperblock](population, size_individual, mutation_rate, rng_states)
print('Algo 2 :' + str(time() - time_iteration_start))

score_population = cuda.to_device(np.array([0.] * size_population))
time_iteration_start = time()
mutate2[blockspergrid, threadsperblock](population, size_individual, mutation_rate, rng_states)
print('Algo 2 V2:' + str(time() - time_iteration_start))



population = np.array(init_population(size_individual, size_population))
dataset_score = distance_enter_points(dataset)

time_iteration_start = time()
mutate3(population, size_individual, mutation_rate)
print('Algo 3:' + str(time() - time_iteration_start))

time_iteration_start = time()
mutate3(population, size_individual, mutation_rate)
print('Algo 3 V2:' + str(time() - time_iteration_start))