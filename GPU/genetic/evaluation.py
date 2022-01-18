import numpy as np
from numba import cuda


@cuda.jit
def evaluate(population, dataset_score, score_population, size_individual):
    index = cuda.grid(1)
    if index < population.shape[0]:
        score_population[index] = loss_function(population[index], dataset_score, size_individual)

@cuda.jit(device=True)
def loss_function(individual, dataset_score, size_individual):
    score = 0
    for index_individual in range(size_individual):
        if index_individual < size_individual - 1:
            score += dataset_score[individual[index_individual]][individual[index_individual+1]]
        else:
            score += dataset_score[individual[0]][individual[index_individual]]
    return score
