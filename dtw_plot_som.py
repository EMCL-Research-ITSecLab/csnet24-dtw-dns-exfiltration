import math

import dtwsom
import matplotlib.pyplot as plt
from pyclustering.nnet.som import type_conn

from utils import TIME_INTERVAL_CONFIG, load_dataset

if __name__ == "__main__":
    for ti in TIME_INTERVAL_CONFIG:
        _, _, x_arr, y_arr = load_dataset(ti["time_interval_name"])

        structure = type_conn.grid_four

        som_x = som_y = math.ceil(math.sqrt(math.sqrt(len(x_arr))))
        network = dtwsom.DtwSom(som_x, som_y, structure)
        network.train(x_arr, 20)

        plt.tight_layout()
        network.show_distance_matrix()
        plt.savefig(f"figs/som_utmatrix_{ti['time_interval_name']}.pdf")
        plt.clf()
        
        plt.tight_layout()  
        network.show_winner_matrix()
        plt.savefig(f"figs/som_winner_{ti['time_interval_name']}.pdf")
        plt.clf()
