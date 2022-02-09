import random


def mutate(population, size_individual, mutation_rate):
    for index in range(len(population)):
        individual_mutate(population[index], size_individual, mutation_rate)


def individual_mutate(individual, size_individual, mutation_rate):
    for position_1 in range(size_individual):
        if random.random() < mutation_rate:
            position_2 = round(random.random() * (size_individual - 1))
            if not position_1 == position_2:
                swap_value = individual[position_1]
                individual[position_1] = individual[position_2]
                individual[position_2] = swap_value