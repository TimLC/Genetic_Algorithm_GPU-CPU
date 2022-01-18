import math
from time import time, time_ns

import numpy as np
from numba import cuda
from numba.cuda import random

from processing_dataset.dataset import dataset_of_traveling_salesman, distance_enter_points
from genetic.evolution import evolution
from genetic.init_population import init_population
from utils.utils import display

if __name__ == '__main__':
    size_population = 8192
    population_rate_to_keep = 0.5
    mutation_rate = 0.1
    number_generation = 10000
    number_step_to_actualize_view = 100

    duration = 0

    dataset = dataset_of_traveling_salesman('dataset2')
    dataset_score = distance_enter_points(dataset)
    size_individual = len(dataset_score)
    number_population_to_keep = int(round(population_rate_to_keep * size_population))
    spliter = int(round(number_population_to_keep)/2) if int(round(number_population_to_keep)/2)%2 == 0 else int(round(number_population_to_keep)/2)+1

    population = np.array(init_population(size_individual, size_population), dtype=np.min_scalar_type(size_individual))

    score_population = np.array([0.] * size_population)
    result = []
    dataset_score_cuda = cuda.to_device(dataset_score)

    threadsperblock = 1024
    blockspergrid = math.ceil(size_population / threadsperblock)
    nthreads = threadsperblock
    rng_states = random.create_xoroshiro128p_states(nthreads, seed=time_ns())

    for iteration in range(number_generation):
        time_iteration_start = time()
        new_best_individual = evolution(population, dataset_score_cuda, score_population, size_individual, mutation_rate, spliter, number_population_to_keep, result, iteration, rng_states, threadsperblock, blockspergrid)
        duration += time() - time_iteration_start
        display(dataset, result, iteration, number_generation, new_best_individual, number_step_to_actualize_view)
    print(population)
    print(duration)

