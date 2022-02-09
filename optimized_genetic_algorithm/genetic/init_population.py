import random

import numpy as np
from numba import jit, prange


def init_population(size_individual, size_population):
    return [generate_individual(size_individual) for _ in range(size_population)]


def generate_individual(size_individual):
    individual = [gene for gene in range(size_individual)]
    random.shuffle(individual)
    return individual


@jit(nopython=True, parallel=True, fastmath=True)
def regenerate_population(population):
    for index_population in prange(len(population)):
        np.random.shuffle(population[index_population])
