import math


def euclidean_distance(p1x, p1y, p2x, p2y):
    return math.sqrt(((p1x - p2x)**2)+((p1y - p2y)**2))


def dataset_of_traveling_salesman(dataset_name):
    with open('./dataset/' + dataset_name + '.txt', 'r') as f:
        dataset = [tuple(map(float, i.replace('(','').replace(')','').split(','))) for i in f]
    return dataset


def distance_between_points(dataset):
    dataset_of_distance = [[-1 for _ in range(len(dataset))] for _ in range(len(dataset))]
    for i in range(len(dataset)):
        for j in range(len(dataset)):
            if not i == j:
                dataset_of_distance[i][j] = euclidean_distance(dataset[i][0], dataset[i][1], dataset[j][0], dataset[j][1])
    return dataset_of_distance
