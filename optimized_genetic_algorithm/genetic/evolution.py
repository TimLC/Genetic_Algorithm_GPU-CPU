import numpy as np
from numba import cuda

from optimized_genetic_algorithm.genetic.crossover import crossover
from optimized_genetic_algorithm.genetic.evaluation import evaluate
from optimized_genetic_algorithm.genetic.init_population import regenerate_population
from optimized_genetic_algorithm.genetic.mutation import mutate
from optimized_genetic_algorithm.genetic.selection import selector
from optimized_genetic_algorithm.utils.utils import shuffle_list


def evolution(population, dataset_score_cuda, population_scores, size_individual, mutation_rate, spliter, number_population_to_keep, result, iteration, threads_per_block, blocks_per_grid):
    population_cuda = cuda.to_device(population)
    population_scores_cuda = cuda.to_device(population_scores)

    evaluate[blocks_per_grid, threads_per_block](population_cuda, dataset_score_cuda, population_scores_cuda, size_individual)
    population_scores[:] = population_scores_cuda.copy_to_host()

    selector(population, population_scores, number_population_to_keep)

    shuffle_list(population[1:(number_population_to_keep + 1)])

    crossover(population[1:(spliter + 1)], size_individual)

    random_values = np.random.rand((number_population_to_keep - spliter), 2)
    random_values_cuda = cuda.to_device(random_values)

    population_cuda_mutation = cuda.to_device(population[(spliter + 1):(number_population_to_keep + 1)])
    mutate[blocks_per_grid, threads_per_block](population_cuda_mutation, random_values_cuda, size_individual, mutation_rate)

    population[(spliter + 1):(number_population_to_keep + 1)] = population_cuda_mutation.copy_to_host()

    regenerate_population(population[(number_population_to_keep + 1):])

    if not result:
        result.append((iteration, population_scores[0], population[0]))
        new_best_individual = True
    elif result and result[-1][1] > population_scores[0]:
        result.append((iteration, population_scores[0], population[0]))
        new_best_individual = True
    else:
        new_best_individual = False

    return new_best_individual
