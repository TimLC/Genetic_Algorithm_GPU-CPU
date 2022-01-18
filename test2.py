import math
import os
import time
from copy import deepcopy

import numba
import numpy as np
from matplotlib import pyplot as plt
from numba import cuda, jit
from numba.cuda import random
from numba.types import int32
from GPU.genetic.crossover import crossover
from GPU.genetic.init_population import init_population

@cuda.jit
def crossover1(population, size_individual, rng_states):
    thread_id = cuda.grid(1)
    for i in range(0, len(population), 2):
        size_crossover = round(random.xoroshiro128p_uniform_float32(rng_states, thread_id) * (size_individual-1))
        crossover_mutate(population[i], population[i+1], size_individual, rng_states)


@cuda.jit(device=True)
def crossover_mutate(individual_a, individual_b, size_crossover, rng_states):
    thread_id = cuda.grid(1)
    index_to_split = round(random.xoroshiro128p_uniform_float32(rng_states, thread_id) * (len(individual_a)-1))

    if size_individual >= (index_to_split + size_crossover):
        new_individual_a = cuda.local.array(size_individual, int32)
        new_individual_b = cuda.local.array(size_individual, int32)
        vec_concatenate_3(individual_a[:index_to_split], individual_b[index_to_split:(index_to_split + size_crossover)], individual_a[(index_to_split + size_crossover):], new_individual_a)
        vec_concatenate_3(individual_b[:index_to_split], individual_a[index_to_split:(index_to_split + size_crossover)], individual_b[(index_to_split + size_crossover):], new_individual_a)

        mapping_individual_a = individual_b[index_to_split:(index_to_split + size_crossover)]
        mapping_individual_b = individual_a[index_to_split:(index_to_split + size_crossover)]

        a1 = [3, 6, 4, 7, 5, 1]
        a2 = [2, 3, 5, 1, 8, 6]
        matrix_mapping = generate_matrix_mapping(a1, a2)

        list_gene_mapping = np.array(range(index_to_split)) + np.array(range(index_to_split + size_crossover, len(new_individual_a)))

    else:
        new_individual_a = individual_b[:(index_to_split + size_crossover - len(individual_a))] + individual_a[(index_to_split + size_crossover - len(individual_a)):index_to_split] + individual_b[index_to_split:]
        new_individual_b = individual_a[:(index_to_split + size_crossover - len(individual_a))] + individual_b[(index_to_split + size_crossover - len(individual_a)):index_to_split] + individual_a[index_to_split:]

        mapping_individual_a = individual_b[:(index_to_split + size_crossover - len(individual_a))] + individual_b[index_to_split:] + individual_b[:(index_to_split + size_crossover - len(individual_a))]
        mapping_individual_b = individual_a[:(index_to_split + size_crossover - len(individual_a))] + individual_a[index_to_split:] + individual_a[:(index_to_split + size_crossover - len(individual_a))]

        matrix_mapping = generate_matrix_mapping(mapping_individual_a, mapping_individual_b)

        list_gene_mapping = range((index_to_split + size_crossover - len(individual_a)),index_to_split)

    if matrix_mapping:
        for i in list_gene_mapping:
            if new_individual_a.count(new_individual_a[i]) == 2:
                new_individual_a[i] = [value_matrix[1] for value_matrix in matrix_mapping if new_individual_a[i] == value_matrix[0]][0]
            if new_individual_b.count(new_individual_b[i]) == 2:
                new_individual_b[i] = [value_matrix[1] for value_matrix in matrix_mapping if new_individual_b[i] == value_matrix[0]][0]
    individual_a = new_individual_a
    individual_b = new_individual_b

@jit(nopython=True)
def generate_matrix_mapping(mapping_individual_a, mapping_individual_b):
    matrix_mapping = list(zip(mapping_individual_a, mapping_individual_b))
    while True:
        list_del = []
        for value_matrix in matrix_mapping:
            index = matrix_mapping.index(value_matrix)
            mapper_b = [j for _, j in matrix_mapping]
            if value_matrix[0] in mapper_b:
                if not index in list_del:
                    position = mapper_b.index(value_matrix[0])
                    matrix_mapping[index] = (matrix_mapping[position][0], matrix_mapping[index][1])
                    list_del.append(position)
        if list_del:
            list_del = sorted(list_del, reverse=True)
            for idx in list_del:
                matrix_mapping.pop(idx)
        else:
            break
    matrix_mapping = matrix_mapping + list(map(lambda x: (x[1], x[0]), matrix_mapping))
    return matrix_mapping

@cuda.jit(device=True)
def vec_concatenate_2(vec, vec2, vec3):
    n = len(vec)
    n2 = len(vec2)
    for i in range(n):
        vec3[i] = vec[i]
    for i in range(n2):
        vec3[i + n] = vec2[i]

@cuda.jit(device=True)
def vec_concatenate_3(vec, vec2, vec3, vec4):
    n = len(vec)
    n2 = len(vec2)
    n3 = len(vec3)
    for i in range(n):
        vec4[i] = vec[i]
    for i in range(n2):
        vec4[i + n] = vec2[i]
    for i in range(n3):
        vec4[i + n + n2] = vec3[i]




@cuda.jit
def aaaa(a, b, c):
    matrix_mapping = cuda.local.array((10,2), int32)
    for index, value in enumerate(zip(a,b)):
        matrix_mapping[index,0] = value[0]
        matrix_mapping[index,1] = value[1]
    for i in range(10):
        for j in range(2):
            c[i,j] = matrix_mapping[i,j]


@jit(nopython=True)
def test2(mapping_individual_a, mapping_individual_b):
    matrix_mapping = list(zip(mapping_individual_a, mapping_individual_b))
    while True:
        list_del = []
        for value_matrix in matrix_mapping:
            index = matrix_mapping.index(value_matrix)
            mapper_b = [j for _, j in matrix_mapping]
            if value_matrix[0] in mapper_b:
                if not index in list_del:
                    position = mapper_b.index(value_matrix[0])
                    matrix_mapping[index] = (matrix_mapping[position][0], matrix_mapping[index][1])
                    list_del.append(position)
        if list_del:
            list_del = sorted(list_del, reverse=True)
            for idx in list_del:
                matrix_mapping.pop(idx)
        else:
            break
    matrix_mapping = matrix_mapping + list(map(lambda x: (x[1], x[0]), matrix_mapping))
    return matrix_mapping

@cuda.jit(device=True)
def test3(a, b, matrix_mapping_bool):
    matrix_mapping = cuda.shared.array(shape=(10,2), dtype=numba.int32)

    for index, value in enumerate(zip(a, b)):
        matrix_mapping[index, 0] = value[0]
        matrix_mapping[index, 1] = value[1]
    for index, value_matrix in enumerate(matrix_mapping):
        test_delete = False
        for b_index, b_value in enumerate(b):
            if value_matrix[0] == b_value:
                position = b_index
                test_delete = True
                break
        if test_delete:
            if matrix_mapping_bool[position]:
                matrix_mapping[index] = (matrix_mapping[position][0], matrix_mapping[index][1])
                matrix_mapping_bool[position] = False
    return matrix_mapping
    # matrix_mapping = matrix_mapping + map(lambda x: (x[1], x[0]), matrix_mapping)

@cuda.jit
def test1(a):
    a[0][1]=11


@jit(nopython=True)
def azertyuiop(population, size_individual):
    initial_individual = np.arange(size_individual)
    for individual in population:
        initial_individual = initial_individual
        np.random.shuffle(individual)



a = np.array([[0,1,2],[3,4,5],[6,7,8]])
print(a)
a[0] = a[1]
print(a)
a[0][0] = 9
print(a)
exit()

a = np.array([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],[9, 8, 7, 6, 5, 4, 3, 2, 1, 0]])
size=10

a1 = cuda.to_device(a)
a1[0][8]=44
test1[1,1](a1)

print(a)
print(np.array(a1))
exit()
azertyuiop(a, size)
print(a)
a[1][3] = 11
print(a)
exit()

size_population = 1000
nthreads = 32 * 32
mutation_rate = 0.2
size_individual = 10
# a1 = [3, 4, 5, 6]
# a2 = [6, 5, 4, 3]
# # a1 = [1, 4, 5, 6, 3]
# # a2 = [6, 5, 8, 3, 2]
a1 = [3, 6, 4, 7, 5, 1]
a2 = [2, 3, 5, 1, 8, 6]

matrix_mapping_bool = [True, True, True, True, True, True]

a1 = cuda.to_device(a1)
a2 = cuda.to_device(a2)
matrix_mapping_bool = cuda.to_device(matrix_mapping_bool)

test1[1,1](a1, a2, matrix_mapping_bool)
print(np.array(a1))
print(np.array(a2))
print(np.array(matrix_mapping_bool))


a = [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],[9, 8, 7, 6, 5, 4, 3, 2, 1, 0]]
print(np.array(a))
a = cuda.to_device(a)
rng_states = random.create_xoroshiro128p_states(nthreads, seed=time.time_ns())

crossover[32,32](a, size_individual, rng_states)

print(np.array(a))


