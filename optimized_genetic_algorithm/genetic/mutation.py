from numba import cuda


@cuda.jit
def mutate(population, random_values, size_individual, mutation_rate):
    index = cuda.grid(1)
    if index < population.shape[0]:
        individual_mutate(population[index], random_values[index], size_individual, mutation_rate)


@cuda.jit(device=True)
def individual_mutate(individual, random_values, size_individual, mutation_rate):
    for position_1 in range(size_individual):
        if random_values[0] < mutation_rate:
            position_2 = round(random_values[1] * (size_individual - 1))
            if not position_1 == position_2:
                swap_value = individual[position_1]
                individual[position_1] = individual[position_2]
                individual[position_2] = swap_value
