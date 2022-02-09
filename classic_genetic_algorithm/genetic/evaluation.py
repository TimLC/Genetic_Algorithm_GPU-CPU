def evaluate(population, dataset_score, population_scores, size_individual):
    for index in range(len(population)):
        population_scores[index] = loss_function(population[index], dataset_score, size_individual)


def loss_function(individual, dataset_score, size_individual):
    score = 0
    for index_individual in range(size_individual):
        if index_individual < size_individual - 1:
            score += dataset_score[individual[index_individual]][individual[index_individual+1]]
        else:
            score += dataset_score[individual[0]][individual[index_individual]]
    return score
