import random


def crossover(population, size_individual):
    for i in range(0, len(population), 2):
        size_crossover = 1 + round(random.random() * (size_individual-2))
        population[i:(i+2)].append(crossover_mutate(population[i], population[i+1], size_crossover))
    return population

def crossover_mutate(individual_a, individual_b, size_crossover):
    index_to_split = round(random.random() * (len(individual_a)-1))

    if len(individual_a) >= (index_to_split + size_crossover):
        new_individual_a = individual_a[:index_to_split] + individual_b[index_to_split:(index_to_split + size_crossover)] + individual_a[(index_to_split + size_crossover):]
        new_individual_b = individual_b[:index_to_split] + individual_a[index_to_split:(index_to_split + size_crossover)] + individual_b[(index_to_split + size_crossover):]

        mapping_individual_a = individual_b[index_to_split:(index_to_split + size_crossover)]
        mapping_individual_b = individual_a[index_to_split:(index_to_split + size_crossover)]

        matrix_mapping = generate_matrix_mapping(mapping_individual_a, mapping_individual_b)

        list_gene_mapping = list(range(index_to_split)) + list(range(index_to_split + size_crossover, len(individual_a)))

    else:
        new_individual_a = individual_b[:(index_to_split + size_crossover - len(individual_a))] + individual_a[(index_to_split + size_crossover - len(individual_a)):index_to_split] + individual_b[index_to_split:]
        new_individual_b = individual_a[:(index_to_split + size_crossover - len(individual_a))] + individual_b[(index_to_split + size_crossover - len(individual_a)):index_to_split] + individual_a[index_to_split:]

        mapping_individual_a = individual_a[:(index_to_split + size_crossover - len(individual_a))] + individual_a[index_to_split:]
        mapping_individual_b = individual_b[:(index_to_split + size_crossover - len(individual_a))] + individual_b[index_to_split:]

        matrix_mapping = generate_matrix_mapping(mapping_individual_a, mapping_individual_b)

        list_gene_mapping = list(range((index_to_split + size_crossover - len(individual_a)),index_to_split))

    for i in list_gene_mapping:
        if new_individual_a.count(new_individual_a[i]) == 2:
            new_individual_a[i] = [value_matrix[1] for value_matrix in matrix_mapping if new_individual_a[i] == value_matrix[0]][0]
        if new_individual_b.count(new_individual_b[i]) == 2:
            new_individual_b[i] = [value_matrix[1] for value_matrix in matrix_mapping if new_individual_b[i] == value_matrix[0]][0]
    return new_individual_a, new_individual_b


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
