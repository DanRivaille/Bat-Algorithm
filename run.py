from BatAlgorithm import *
from cec2013lsgo.cec2013 import Benchmark
from numpy.random import rand

if __name__ == '__main__':
  bench = Benchmark()

  CANT_FUNCTIONS = 15

  for num_function in range(1, CANT_FUNCTIONS + 1):
    info = bench.get_info(num_function)
    print(f'Function {num_function}: {info}')

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
      ObjetiveFunction = bench.get_function(num_function)

      bats = BatAlgorithm(i, BKS, D, NP, N_Gen, A, r, alpha, gamma, fmin, fmax, Lower, Upper, ObjetiveFunction)
      bats.move_bats(name_logs_file=f'Logs/Funcion{num_function}_{i}.csv', interval_logs=10)
