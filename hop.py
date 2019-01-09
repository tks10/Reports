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

        

    def recall(self, initial_state):
        pass


def main():
    pass


if __name__ == "__main__":
    main()
