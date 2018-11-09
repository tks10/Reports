import numpy as np
import itertools


class TspSolver:
    def __init__(self, adjacent):
        self._adjacent = np.asarray(adjacent)

    def exhaustive_search(self):
        origin = range(len(self._adjacent))
        paths = list(itertools.permutations(origin))
        min_ = np.inf
        best_path = None

        for path in paths:
            cost = self.get_cost(path)
            if cost < min_:
                min_ = cost
                best_path = list(path) + [path[0]]
                print("[Cost]", min_)
                print("[Path]", " > ".join(map(str, best_path)), "\n")

        return min_, best_path

    def get_cost(self, path):
        sum_ = 0
        for i in range(len(path)-1):
            sum_ += self._adjacent[path[i]][path[i+1]]
        sum_ += self._adjacent[path[-1]][0]

        return sum_

    # 分枝わからん
    def solve(self, start, current_path=[]):
        if start in current_path:
            return
        current_path.append(start)
        print()
        print(current_path)

        # Get the potential destination
        dst = np.argmin(self._adjacent[start])

        # Check in case of use and unuse
        adjacent = np.copy(self._adjacent)
        adjacent_unuse = np.copy(self._adjacent)
        adjacent_unuse[start][dst] = np.inf

        # Get lowers
        lower_use = self.calc_lower_bound(adjacent)
        lower_unuse = self.calc_lower_bound(adjacent_unuse)
        print(start+1, dst+1)
        print(lower_use, lower_unuse)

        if lower_use <= lower_unuse:
            self._adjacent[start][dst] = 0
            print(self._adjacent)
            self.solve(dst, current_path)
        else:
            self._adjacent = adjacent_unuse
            current_path.remove(current_path[-1])
            self.solve(start, current_path)


    @staticmethod
    def calc_lower_bound(src):
        row_mins = src.min(axis=1)
        result = src - np.reshape(row_mins, (-1, 1))
        col_mins = result.min(axis=0)
        result = result - col_mins
        lower = row_mins.sum() + col_mins.sum()

        return lower

    @staticmethod
    def argmin2(src):
        assert len(src.shape) == 2
        row, col = np.unravel_index(np.argmin(src), src.shape)

        return row, col


def main():
    adjacent = [
        [np.inf, 21, 5, 15, 9],
        [17, np.inf, 12, 6, 24],
        [13, 5, np.inf, 20, 8],
        [9, 12, 7, np.inf, 23],
        [26, 7, 13, 8, np.inf]
    ]

    adjacent2 = [
        [np.inf, 21, 7, 13, 15],
        [11, np.inf, 19, 12, 25],
        [15, 24, np.inf, 13, 5],
        [6, 17, 9, np.inf, 22],
        [28, 6, 11, 5, np.inf]
    ]

    solver = TspSolver(adjacent2)
    solver.exhaustive_search()


if __name__ == '__main__':
    main()

