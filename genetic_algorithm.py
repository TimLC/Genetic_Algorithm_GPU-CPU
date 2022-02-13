import argparse
import sys

from classic_genetic_algorithm.classic_genetic_algorithm import run_classic_genetic_algorithm
from comparison_classic_vs_optimized.comparison_classic_vs_optimized import \
    run_comparison_classic_vs_optimized_for_size_population, run_comparison_classic_vs_optimized_for_number_generation
from optimized_genetic_algorithm.optimized_genetic_algorithm import run_optimized_genetic_algorithm


def classic():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str)
    parser.add_argument('--size_population', type=int, default=8192)
    parser.add_argument('--number_generation', type=int, default=1000)
    parser.add_argument('--population_rate_to_keep', type=float, default=0.50, choices=[i / 100 for i in range(0, 100)])
    parser.add_argument('--mutation_rate', type=float, default=0.10, choices=[i / 100 for i in range(0, 100)])
    parser.add_argument('--number_step_to_actualize_view', type=str, default=100)

    args, _ = parser.parse_known_args()
    duration, individual = run_classic_genetic_algorithm(**vars(args))
    print('Duration : %.2f seconds' % duration)
    print('Individual best solution : %s' % individual)


def optimized():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str)
    parser.add_argument('--size_population', type=int, default=8192)
    parser.add_argument('--number_generation', type=int, default=1000)
    parser.add_argument('--population_rate_to_keep', type=float, default=0.50, choices=[i / 100 for i in range(0, 100)])
    parser.add_argument('--mutation_rate', type=float, default=0.10, choices=[i / 100 for i in range(0, 100)])
    parser.add_argument('--number_step_to_actualize_view', type=str, default=100)
    parser.add_argument('--threads_per_block', type=int, default=1024)

    args, _ = parser.parse_known_args()
    duration, individual = run_optimized_genetic_algorithm(**vars(args))
    print('Duration : %.2f seconds' % duration)
    print('Individual best solution : %s' % individual)


def comparison_size_population():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str)
    parser.add_argument('--size_population', type=int, nargs='*', default=[1024, 2048, 4096, 8192])
    parser.add_argument('--number_generation', type=int, default=1000)
    parser.add_argument('--population_rate_to_keep', type=float, default=0.50, choices=[i / 100 for i in range(0, 100)])
    parser.add_argument('--mutation_rate', type=float, default=0.10, choices=[i / 100 for i in range(0, 100)])
    parser.add_argument('--number_step_to_actualize_view', type=str, default=100)
    parser.add_argument('--threads_per_block', type=int, default=1024)

    args, _ = parser.parse_known_args()
    run_comparison_classic_vs_optimized_for_size_population(**vars(args))


def comparison_number_generation():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str)
    parser.add_argument('--size_population', type=int, default=8192)
    parser.add_argument('--number_generation', type=int, nargs='*', default=[10, 100, 1000,10000])
    parser.add_argument('--population_rate_to_keep', type=float, default=0.50, choices=[i / 100 for i in range(0, 100)])
    parser.add_argument('--mutation_rate', type=float, default=0.10, choices=[i / 100 for i in range(0, 100)])
    parser.add_argument('--number_step_to_actualize_view', type=str, default=100)
    parser.add_argument('--threads_per_block', type=int, default=1024)

    args, _ = parser.parse_known_args()
    run_comparison_classic_vs_optimized_for_number_generation(**vars(args))


def get_action_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument('--action', type=str)
    action_arg, _ = parser.parse_known_args()

    return action_arg


if __name__ == '__main__':
    if len(sys.argv) > 2:
        arg = get_action_arg()
        globals()[arg.action]()
    else:
        exit()
