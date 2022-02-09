from numba.cuda import random
from numba import cuda


@cuda.jit
def mutate(population, size_individual, mutation_rate, rng_states):
    index = cuda.grid(1)
    individual_mutate(population[index], size_individual, mutation_rate, rng_states, index)


@cuda.jit(device=True)
def individual_mutate(individual, size_individual, mutation_rate, rng_states, thread_id):
    for position_1 in range(size_individual):
        if random.xoroshiro128p_uniform_float32(rng_states, thread_id) < mutation_rate:
            position_2 = round(random.xoroshiro128p_uniform_float32(rng_states, thread_id) * (size_individual - 1))
            if not position_1 == position_2:
                swap_value = individual[position_1]
                individual[position_1] = individual[position_2]
                individual[position_2] = swap_value