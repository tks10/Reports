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

    def recall(self, initial_state, show_progress=True):
        if self._size != len(initial_state):
            raise ValueError

        for i in range(self._size):
            self._nodes[i] = initial_state[i]

        self.show_nodes()

        while self.update_nodes(show_progress):
            pass

    def update_nodes(self, show_progress=False):
        is_changed = False

        for i in range(self._size):
            value_sum = 0
            for j in range(self._size):
                value_sum += self._weight_adjacent[j][i] * self._nodes[j]

            if value_sum > 0:
                updated_nodes = 1

            elif value_sum < 0:
                updated_nodes = 0

            else:
                updated_nodes = self._nodes[i]

            if self._nodes[i] != updated_nodes:
                self._nodes[i] = updated_nodes
                is_changed = True

                if show_progress:
                    self.show_nodes()

        return is_changed

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

        ax.set_xticks(np.arange(data.shape[0]), minor=False)
        ax.set_yticks(np.arange(data.shape[1]), minor=False)

        ax.invert_yaxis()
        ax.xaxis.tick_top()

        plt.show()


def main():
    hfn = HopFieldNetwork()

    state_to_memory = [
        [0, 1, 0,
         1, 1, 1,
         0, 1, 0]
    ]

    initial_state = [
        1, 0, 1,
        1, 0, 0,
        0, 1, 0
    ]

    hfn.memorize(state_to_memory)
    hfn.recall(initial_state)


if __name__ == "__main__":
    main()
