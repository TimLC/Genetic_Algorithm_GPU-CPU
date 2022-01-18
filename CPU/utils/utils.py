import cv2
import matplotlib
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.ticker import LogFormatter


def display(dataset, result, iteration, cycle, new_best_individual):
    number_step_to_actualize_view = 100
    if not (new_best_individual or round(iteration%number_step_to_actualize_view) == 0):
        return
    elif new_best_individual:
        matplotlib.use('Agg')
        fig = plt.figure(figsize=(6, 4))

        new_width = 500
        font = cv2.FONT_HERSHEY_SIMPLEX
        bottom_left_corner_of_text = (100, 40)
        font_scale = 1
        font_color = (0,0,0)
        thickness = 1
        line_type = 0
        text_to_display = "Score : " + str(round(result[-1][1],2))
        best_individual = result[-1][2]

        x_values = []
        y_values = []

        for gene in best_individual:
            x_values.append(dataset[gene][0])
            y_values.append(dataset[gene][1])
        plt.scatter(x_values, y_values, color='blue', zorder=2)

        x_values.append(dataset[best_individual[0]][0])
        y_values.append(dataset[best_individual[0]][1])
        plt.plot(x_values, y_values, color='deepskyblue', zorder=1)

        for i, label in enumerate(dataset):
            plt.text(label[0], label[1], i)

        plt.xlim([0, round(max(x_values)*1.1)])
        plt.ylim([0, round(max(y_values)*1.1)])
        plt.gca().set_aspect('equal', adjustable='box')

        img = generate_image(fig, new_width)

        fig2 = plt.figure(figsize=(6,2.5))
        if len(result) > 1:
            x = [i[0] for i in result]
            y = [i[1] for i in result]
            plt.plot(x, y)

        plt.xscale('log')

        plt.xlim([1, cycle])
        plt.ylim([0, result[0][1]])

        img2 = generate_image(fig2, new_width)

        screen = np.concatenate((img, img2), axis=0)
        cv2.putText(screen, text_to_display, bottom_left_corner_of_text, font, font_scale, font_color, thickness, line_type)
        screen = cv2.cvtColor(screen, cv2.COLOR_RGB2BGR)

        plt.close('all')
        cv2.startWindowThread()
        cv2.imshow("Genetic algorithm", screen)
        cv2.waitKey(1)

    elif round(iteration%number_step_to_actualize_view) == 0:
        cv2.waitKey(1)


def generate_image(fig, new_width):
    fig.canvas.draw()
    img = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    percent_size = (new_width / img.shape[1])
    hsize = int(img.shape[0] * percent_size)
    img = cv2.resize(img, (new_width, hsize))
    return img


