from numba import jit, prange


@jit(nopython=True, parallel=True, fastmath=True)
def selector(population, population_scores, number_population_to_keep):
    list_sort = sorted(zip(population.copy(), population_scores.copy()), key=lambda pair: pair[1])

    for index in prange(len(population)):
        population[index] = list_sort[index][0]
        population_scores[index] = list_sort[index][1]

    population[number_population_to_keep] = population[0]