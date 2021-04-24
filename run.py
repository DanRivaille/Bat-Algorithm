from BatAlgorithm import *
from cec2013lsgo.cec2013 import Benchmark
from numpy.random import rand

if __name__ == '__main__':
  bench = Benchmark()
  info = bench.get_info(1)
  print(info)

  ejecutions = 1

  for i in range(1, ejecutions + 1):
    BKS = info['best']
    D = info['dimension']
    NP = 40
    N_Gen = 30
    A = 0.95
    r = 0.1
    alpha = 0.95
    gamma = 0.95
    fmin = 0
    fmax = 1
    Lower = info['lower']
    Upper = info['upper']
    ObjetiveFunction = bench.get_function(1)

    bats = BatAlgorithm(i, BKS, D, NP, N_Gen, A, r, alpha, gamma, fmin, fmax, Lower, Upper, ObjetiveFunction)
    bats.move_bats(f'Funcion1_{i}.csv')
