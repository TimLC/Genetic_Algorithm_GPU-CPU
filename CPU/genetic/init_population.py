import random


def init_population(size_individual, size_population):
    return [generate_individual(size_individual) for i in range(size_population)]


def generate_individual(size_individual):
    individual = [i for i in range(size_individual)]
    random.shuffle(individual)
    return individual