import warnings
from time import time

from processing_dataset.dataset import dataset_of_traveling_salesman, distance_enter_points
from genetic.evolution import evolution
from genetic.init_population import init_population
from utils.utils import display

if __name__ == '__main__':
    # warnings.filterwarnings("ignore")

    size_population = 16384
    population_rate_to_keep = 0.5
    cycle = 10000
    mutation_rate = 0.1

    duration = 0

    dataset = dataset_of_traveling_salesman('dataset2')
    dataset_score = distance_enter_points(dataset)
    size_individual = len(dataset_score)
    population = init_population(size_individual, size_population)
    result = []
    for iteration in range(cycle):
        time_iteration_start = time()
        population, result, new_best_individual = evolution(population, dataset_score, size_individual, mutation_rate, population_rate_to_keep, result, iteration)
        duration += time() - time_iteration_start
        print(time() - time_iteration_start)
        display(dataset, result, iteration, cycle, new_best_individual)
    print(duration)