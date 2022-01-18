import random
from copy import deepcopy

from CPU.genetic.crossover import crossover
from CPU.genetic.evaluation import evaluate
from CPU.genetic.init_population import generate_individual
from CPU.genetic.mutation import mutate
from CPU.genetic.selection import selector


def evolution(population, dataset_score, size_individual, mutation_rate, population_rate_to_keep, result, iteration):
    score_population = evaluate(population, dataset_score)
    population_to_keep, score_population = selector(population, score_population, population_rate_to_keep)

    best_individual = deepcopy([population_to_keep[0]])
    random.shuffle(population_to_keep)
    spliter = int(round(len(population_to_keep))/2) if int(round(len(population_to_keep))/2)%2 == 0 else int(round(len(population_to_keep))/2)+1

    crossover_population = crossover(population_to_keep[:spliter], size_individual)
    mutate_population = mutate(population_to_keep[spliter:], size_individual, mutation_rate)

    generate_population = [generate_individual(size_individual) for i in range(len(population_to_keep)+1, len(population))]

    new_population = [*best_individual, *mutate_population, *crossover_population, *generate_population]

    if not result:
        result.append((iteration, score_population[0], best_individual[0]))
        new_best_individual = True
    elif result and result[-1][1] > score_population[0]:
        result.append((iteration, score_population[0], best_individual[0]))
        new_best_individual = True
    else:
        new_best_individual = False

    return new_population, result, new_best_individual



