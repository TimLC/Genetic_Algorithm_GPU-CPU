import random


def mutate(population, size_individual, mutation_rate):
    return list(map(lambda individual: individual_mutate(individual, mutation_rate), population))


def individual_mutate(individual, mutation_rate):
    for position_1 in range(len(individual)):
        if random.random() < mutation_rate:
            position_2 = round(random.random() * (len(individual)-1))
            if not position_1 == position_2:
                swap_value = individual[position_1]
                individual[position_1] = individual[position_2]
                individual[position_2] = swap_value
    return individual