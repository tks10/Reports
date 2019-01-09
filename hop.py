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
        pass

    def init_weights(self):
        for i in range(self._size):
            for j in range(self._size):
                self._weight_adjacent[i][j] = 0


def main():
    pass


if __name__ == "__main__":
    main()
