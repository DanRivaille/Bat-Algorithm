from BatAlgorithm import *
from cec2013lsgo.cec2013 import Benchmark
from numpy.random import rand

if __name__ == '__main__':
  bench = Benchmark()

  INITIAL_FUNCTION = 2
  CANT_FUNCTIONS = 1

  for num_function in range(INITIAL_FUNCTION, INITIAL_FUNCTION + CANT_FUNCTIONS):
    info = bench.get_info(num_function)
    print(f'\nFunction {num_function}: {info}')

    ejecutions = 1

    for i in range(1, ejecutions + 1):
      BKS = info['best']
      D = info['dimension']
      NP = 30
      N_Gen = 35
      A = 0.95
      r = 0.1
      alpha = 0.9
      gamma = 0.5
      fmin = 0
      fmax = 1
      Lower = info['lower']
      Upper = info['upper']
      ObjetiveFunction = bench.get_function(num_function)

      bats = BatAlgorithm(i, BKS, D, NP, N_Gen, A, r, alpha, gamma, fmin, fmax, Lower, Upper, ObjetiveFunction)
      bats.move_bats(num_function, f'Logs/Funcion{num_function}_{i}.csv', 10)
