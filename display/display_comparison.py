import os

from matplotlib import pyplot as plt


def display_comparison(results, step, name_comparison):
    path = './results/'
    image_name = 'graphic-'
    i = 0

    plt.figure()
    plt.plot(step, [x[0] for x in results])
    plt.plot(step, [x[1] for x in results])
    plt.legend(['classic', 'optimized'])
    plt.xlabel(name_comparison)
    plt.ylabel('Time (s)')

    while os.path.exists(path + image_name + '%s.png' % i):
        i += 1
    plt.savefig(path + image_name + '%s.png' % i)

    print("Find result in 'results' directory")
