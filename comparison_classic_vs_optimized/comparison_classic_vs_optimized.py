from classic_genetic_algorithm.classic_genetic_algorithm import run_classic_genetic_algorithm
from display.display_comparison import display_comparison
from optimized_genetic_algorithm.optimized_genetic_algorithm import run_optimized_genetic_algorithm


def run_comparison_classic_vs_optimized_for_size_population(dataset_name='dantzig42', list_size_population=[1024, 2048, 4096, 8192],
                                                            number_generation=10000, population_rate_to_keep=0.5,
                                                            mutation_rate=0.1, number_step_to_actualize_view=100,
                                                            threads_per_block=1024):
    result = []
    for size_population in list_size_population:
        classic_GA, _ = run_classic_genetic_algorithm(dataset_name, size_population, number_generation,
                                                      population_rate_to_keep, mutation_rate, number_step_to_actualize_view)
        optimized_GA, _ = run_optimized_genetic_algorithm(dataset_name, size_population, number_generation,
                                                          population_rate_to_keep, mutation_rate, number_step_to_actualize_view,
                                                          threads_per_block)
        result.append([classic_GA, optimized_GA])

    display_comparison(result, list_size_population, 'Size of the population')


def run_comparison_classic_vs_optimized_for_number_generation(dataset_name='dantzig42', size_population=8192,
                                                              list_number_generation=[10, 100, 1000, 10000],
                                                              population_rate_to_keep=0.5,
                                                              mutation_rate=0.1, number_step_to_actualize_view=100,
                                                              threads_per_block=1024):
    result = []
    for number_generation in list_number_generation:
        classic_GA, _ = run_classic_genetic_algorithm(dataset_name, size_population, number_generation,
                                                      population_rate_to_keep, mutation_rate, number_step_to_actualize_view)
        optimized_GA, _ = run_optimized_genetic_algorithm(dataset_name, size_population, number_generation,
                                                          population_rate_to_keep, mutation_rate, number_step_to_actualize_view,
                                                          threads_per_block)
        result.append([classic_GA, optimized_GA])

    display_comparison(result, list_number_generation, 'Number of generations')