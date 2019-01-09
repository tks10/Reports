import matplotlib.pyplot as plt
import numpy as np
import math


class HopFieldNetwork:
    def __init__(self, size=9):
        self._size = size
        self._nodes = [0 for _ in range(self._size)]
        self._weight_adjacent = list()
        for _ in range(self._size):
            self._weight_adjacent.append([0 for _ in range(self._size)])

    def memorize(self, states):
        for state in states:
            if self._size != len(state):
                raise ValueError

        self.init_weights()

        for state in states:
            for i in range(self._size):
                alpha_i = state[i]
                for j in range(self._size):
                    alpha_j = state[j]
                    if i != j:
                        self._weight_adjacent[i][j] += (2 * alpha_i - 1) * (2 * alpha_j - 1)

    def recall(self, initial_state):
        if self._size != len(initial_state):
            raise ValueError

        for i in range(self._size):
            self._nodes[i] = initial_state[i]

    def update(self):
        pass

    def init_weights(self):
        for i in range(self._size):
            for j in range(self._size):
                self._weight_adjacent[i][j] = 0

    def show_nodes(self):
        nodes_np = np.asarray(self._nodes)
        size = int(math.sqrt(self._size))
        HopFieldNetwork.show_matrix(nodes_np.reshape((size, size)))

    @staticmethod
    def show_matrix(data):
        data = np.asarray(data)
        fig, ax = plt.subplots()
        ax.pcolor(data, cmap=plt.cm.Blues)

        ax.set_xticks(np.arange(data.shape[0]) + 0.5, minor=False)
        ax.set_yticks(np.arange(data.shape[1]) + 0.5, minor=False)

        ax.invert_yaxis()
        ax.xaxis.tick_top()

        plt.show()


def main():
    state = [[1, 0, 1, 0, 1, 0, 1, 0, 1]]

    hfn = HopFieldNetwork()
    hfn.recall(state[0])
    hfn.show_nodes()


if __name__ == "__main__":
    main()
