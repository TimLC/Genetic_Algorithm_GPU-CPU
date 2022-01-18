
def evaluate(population, dataset_score):
    score_population = [None] * len(population)
    for index, individual in enumerate(population):
        score_population[index] = loss_function(individual, dataset_score)
    return score_population


def loss_function(individual, dataset_score):
    score = 0
    for index in range(len(individual)):
        if index < len(individual) - 1:
            score += dataset_score[individual[index]][individual[index+1]]
        else:
            score += dataset_score[individual[0]][individual[index]]
    return score
