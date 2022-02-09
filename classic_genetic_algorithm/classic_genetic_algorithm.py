from time import time

import numpy as np

from display.score_board import display
from processing_dataset.processing_dataset import dataset_of_traveling_salesman, distance_between_points
from classic_genetic_algorithm.genetic.evolution import evolution
from classic_genetic_algorithm.genetic.init_population import init_population


def run_classic_genetic_algorithm(dataset_name='dantzig42', size_population=8192, number_generation=10000,
                                  population_rate_to_keep=0.5, mutation_rate=0.1, number_step_to_actualize_view=100):

    duration = 0
    dataset = dataset_of_traveling_salesman(dataset_name)
    dataset_score = distance_between_points(dataset)
    size_individual = len(dataset_score)
    number_population_to_keep = int(round(population_rate_to_keep * size_population))
    spliter_index = int(round(number_population_to_keep) / 2) if int(round(number_population_to_keep) / 2) % 2 == 0 else int(round(number_population_to_keep)/2)+1

    population = np.array(init_population(size_individual, size_population), dtype=np.min_scalar_type(size_individual))

    population_scores = np.array([0.] * size_population)
    result = []

    for iteration in range(number_generation):
        time_iteration_start = time()
        new_best_individual = evolution(population, dataset_score, population_scores, size_individual, mutation_rate,
                                        spliter_index, number_population_to_keep, result, iteration)
        duration += time() - time_iteration_start
        display(dataset, result, iteration, number_generation, new_best_individual, number_step_to_actualize_view)

    return duration, population[0]
