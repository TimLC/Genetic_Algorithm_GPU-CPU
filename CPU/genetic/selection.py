def selector(population, score_population, population_rate_to_keep):
    population_and_score = list(zip(population, score_population))
    population_and_score_sorted = sorted(population_and_score, key=lambda individual_and_score: individual_and_score[1])
    population_sorted_by_score, score = list(zip(*population_and_score_sorted))
    return list(population_sorted_by_score[:round(len(population)*population_rate_to_keep)]), list(score)