from BatAlgorithm import *
from cec2013lsgo.cec2013 import Benchmark
from numpy.random import rand

def Function(D, x):
  value = 0.0
  for i in range(D):
    value = value + x[i] ** 2
  return value

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
  bench = Benchmark()
  print(bench.get_info(1))
  info = bench.get_info(1)
  dim = info['dimension']
  sol = info['lower'] + rand(dim) * (info['upper'] - info['lower'])

  fun_fitness = bench.get_function(1)
  fun_fitness(sol)


  BKS = 0.0

  D = dim
  NP = 40
  N_Gen = 1000
  A = 0.95
  r = 0.1
  alpha = 0.95
  gamma = 0.95
  fmin = 0
  fmax = 1
  Lower = -10
  Upper = 10

  ejecutions = 2

  for i in range(1, ejecutions + 1):
    bats = BatAlgorithm(i, BKS, D, NP, N_Gen, A, r, alpha, gamma, fmin, fmax, Lower, Upper, info)
    bats.move_bats()
