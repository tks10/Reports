import numpy as np
import itertools
np.seterr(divide='ignore', invalid='ignore')


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
                best_path = list(path)
                print("[Cost]", min_)
                print("[Path]", " > ".join(map(lambda p: str(p+1), best_path)), "\n")

        return min_, best_path

    def get_cost(self, path):
        sum_ = 0
        for i in range(len(path)-1):
            sum_ += self._adjacent[path[i]][path[i+1]]
        sum_ += self._adjacent[path[-1]][path[0]]

        return sum_

    def branch_and_bound_search(self, start, current_map=None, current_path=[]):
        current_path.append(start)
        if len(current_path) == len(self._adjacent):
            return self.get_cost(current_path), current_path

        # Init
        if current_map is None:
            current_map = self._adjacent

        # Get the potential destination
        dst = np.argmin(current_map[start])

        # Check in case of use and unuse
        adjacent = np.copy(current_map)
        adjacent_unuse = np.copy(current_map)

        # Update(Use)
        tmp = adjacent[start][dst]
        adjacent[start, :] = np.inf
        adjacent[:, dst] = np.inf
        adjacent[start][dst] = tmp

        # Update(Unuse)
        adjacent_unuse[start][dst] = np.inf

        # Get lowers
        lower_use = self.calc_lower_bound(adjacent)
        lower_unuse = self.calc_lower_bound(adjacent_unuse)

        print("[Determined]", " > ".join(map(str, current_path)))
        print("[Evaluating]", start+1, ">", dst+1)
        print("[Lower bounds(include, exclude)]", lower_use, lower_unuse, "\n")

        if lower_use <= lower_unuse:
            return self.branch_and_bound_search(dst, adjacent, current_path)

        else:
            current_path.remove(current_path[-1])
            return self.branch_and_bound_search(start, adjacent_unuse, current_path)


    @staticmethod
    def calc_lower_bound(src):
        row_mins = src.min(axis=1)
        result = src - np.reshape(row_mins, (-1, 1))
        col_mins = result.min(axis=0)
        lower = row_mins.sum() + col_mins.sum()

        if np.isnan(result).any():
            return np.inf

        return lower


def main():
    adjacent_report = [
        [np.inf, 21, 5, 15, 9],
        [17, np.inf, 12, 6, 24],
        [13, 5, np.inf, 20, 8],
        [9, 12, 7, np.inf, 23],
        [26, 7, 13, 8, np.inf]
    ]

    adjacent_sample = [
        [np.inf, 21, 7, 13, 15],
        [11, np.inf, 19, 12, 25],
        [15, 24, np.inf, 13, 5],
        [6, 17, 9, np.inf, 22],
        [28, 6, 11, 5, np.inf]
    ]

    solver = TspSolver(adjacent_report)

    cost, path = solver.exhaustive_search()
    print("************* Result(Exhaustive) *************")
    print("[Path]", " > ".join(map(lambda p: str(p+1), path+[path[0]])))
    print("[Cost]", cost)
    print("**********************************************\n\n")

    cost, path = solver.branch_and_bound_search(start=2)
    print("*********** Result(Branch and Bound) ***********")
    print("[Path]", " > ".join(map(lambda p: str(p+1), path+[path[0]])))
    print("[Cost]", cost)
    print("************************************************\n\n")


if __name__ == '__main__':
    main()

