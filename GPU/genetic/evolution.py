from numba import cuda

from GPU.genetic.crossover import crossover
from GPU.genetic.evaluation import evaluate
from GPU.genetic.init_population import regenerate_population
from GPU.genetic.mutation import mutate
from GPU.genetic.selection import selector
from GPU.utils.utils import shuffle_list

def evolution(population, dataset_score_cuda, score_population, size_individual, mutation_rate, spliter, number_population_to_keep, result, iteration, rng_states, threadsperblock, blockspergrid):
    population_cuda = cuda.to_device(population)
    score_population_cuda = cuda.to_device(score_population)

    evaluate[blockspergrid, threadsperblock](population_cuda, dataset_score_cuda, score_population_cuda, size_individual)
    score_population[:] = score_population_cuda.copy_to_host()

    selector(population, score_population, number_population_to_keep)

    shuffle_list(population[1:(number_population_to_keep+1)])

    crossover(population[1:(spliter+1)], size_individual)

    population_cuda = cuda.to_device(population[(spliter+1):(number_population_to_keep+1)])
    mutate[blockspergrid, threadsperblock](population_cuda, size_individual, mutation_rate, rng_states)
    population[(spliter+1):(number_population_to_keep+1)] = population_cuda.copy_to_host()

    regenerate_population(population[(number_population_to_keep+1):])

    if not result:
        result.append((iteration, score_population[0], population[0]))
        new_best_individual = True
    elif result and result[-1][1] > score_population[0]:
        result.append((iteration, score_population[0], population[0]))
        new_best_individual = True
    else:
        new_best_individual = False

    return new_best_individual
