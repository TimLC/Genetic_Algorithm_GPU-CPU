from classic_genetic_algorithm.genetic.crossover import crossover
from classic_genetic_algorithm.genetic.evaluation import evaluate
from classic_genetic_algorithm.genetic.init_population import regenerate_population
from classic_genetic_algorithm.genetic.mutation import mutate
from classic_genetic_algorithm.genetic.selection import selector
from classic_genetic_algorithm.utils.utils import shuffle_list


def evolution(population, dataset_score, population_scores, size_individual, mutation_rate, spliter, number_population_to_keep, result, iteration):

    evaluate(population, dataset_score, population_scores, size_individual)

    selector(population, population_scores, number_population_to_keep)

    shuffle_list(population[1:(number_population_to_keep + 1)])

    crossover(population[1:(spliter + 1)], size_individual)

    mutate(population[(spliter + 1):(number_population_to_keep + 1)], size_individual, mutation_rate)

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
