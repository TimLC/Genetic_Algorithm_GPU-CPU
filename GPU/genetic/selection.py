from numba import jit, prange

@jit(nopython=True, parallel=True, fastmath=True)
def selector(population, score_population, number_population_to_keep):
    list_sort = sorted(zip(population.copy(), score_population.copy()), key=lambda pair: pair[1])

    for index in prange(len(population)):
        population[index] = list_sort[index][0]
        score_population[index] = list_sort[index][1]

    population[number_population_to_keep] = population[0]