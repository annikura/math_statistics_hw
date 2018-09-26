import math
import random

import matplotlib.pyplot as plt


def gen_uniform_distributed_sequence(length, a=0, b=1):
    return list([random.uniform(a, b) for _ in range(0, length)])


def gen_exponential_distributed_sequence(length, lam=1):
    return list([-math.log(el) / lam for el in gen_uniform_distributed_sequence(length)])


def uniform_distribution_method_of_moments(sample, k):
    return ((k + 1) * evaluate_kth_sample_moment(sample, k)) ** (1 / k)


def exponential_distribution_method_of_moments(sample, k):
    return (evaluate_kth_sample_moment(sample, k) / math.factorial(k)) ** (1 / k)


def evaluate_kth_sample_moment(sample, k):
    if len(sample) == 0:
        return 0
    return sum(map(lambda x: x ** k, sample)) / len(sample)


if __name__ == "__main__":
    theta = 10
    times = 200
    n = 200
    min_k = 1
    max_k = 55

    uniform_graph_x = []
    uniform_graph_y = []
    exp_graph_x = []
    exp_graph_y = []

    for k in range(min_k, max_k + 1):
        exp_delta = 0
        uniform_delta = 0
        for _ in range(0, times):
            exp_sample = list([el * theta for el in gen_exponential_distributed_sequence(n)])
            uniform_sample = list([el * theta for el in gen_uniform_distributed_sequence(n, 0, 1)])

            exp_delta += (theta - exponential_distribution_method_of_moments(exp_sample, k)) ** 2
            uniform_delta += (theta - uniform_distribution_method_of_moments(uniform_sample, k)) ** 2
        uniform_graph_x.append(k)
        uniform_graph_y.append(uniform_delta / times)

        exp_graph_x.append(k)
        exp_graph_y.append(exp_delta / times)

    plt.subplot(2, 1, 1)
    plt.title("Method of moments error.")
    plt.plot(uniform_graph_x, uniform_graph_y, '-', label="Uniform distribution")
    plt.legend()
    plt.ylabel("Standard deviation")

    plt.subplot(2, 1, 2)
    plt.plot(exp_graph_x, exp_graph_y, '-', label="Exponential distribution")
    plt.legend()
    plt.ylabel("Standard deviation")
    plt.xlabel("k")
    plt.savefig("Method of moments error")
    plt.show()
