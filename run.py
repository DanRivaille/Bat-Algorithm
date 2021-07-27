import sys
from BatAlgorithm import *
from cec2013lsgo.cec2013 import Benchmark

INITIAL_FUNCTION = 1
LAST_FUNCTION = 15

INITIAL_EJECUTION = 1
LAST_EJECUTION = 31

def main():
  bench = Benchmark()

  for num_function in range(INITIAL_FUNCTION, LAST_FUNCTION + 1):
    info = bench.get_info(num_function)
    print(f'\nFunction {num_function}: {info}')


    for i in range(INITIAL_EJECUTION, LAST_EJECUTION + 1):
      BKS = info['best']
      D = info['dimension']
      NP = 30
      N_Gen = 5000
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
      bats.move_bats(num_function, f'Logs/Funcion{num_function}_{i}.csv', 100)


def handle_args():
  """
  Funcion que maneja los argumentos provenientes de la linea de comandos, las
  opciones validas son:
  -f, --function <number>              Ejecuta solo la funcion numero 'number'
  -F, --functions-range <init>:<last>  Ejecuta de la funcion 'init' hasta 'last'
  -e, --ejecution <number>             Ejecuta solo 'number' ejecuciones
  -E, --ejecutions-range <init>:<last> Ejecuta desde 'init' hasta 'last' ejecuciones
  """
  global INITIAL_FUNCTION
  global LAST_FUNCTION
  global INITIAL_EJECUTION
  global LAST_EJECUTION

  cant_args = len(sys.argv)

  if cant_args != 0:
    current_arg_index = 1
    
    while current_arg_index < cant_args:
      current_arg = sys.argv[current_arg_index]

      if '-f' == current_arg or '--function' == current_arg:
        current_arg_index += 1
        INITIAL_FUNCTION = int(sys.argv[current_arg_index])
        LAST_FUNCTION = INITIAL_FUNCTION

      elif '-F' == current_arg or '--functions-range' == current_arg:
        current_arg_index += 1
        range_functions = sys.argv[current_arg_index].split(':')
        INITIAL_FUNCTION = int(range_functions[0])
        LAST_FUNCTION = int(range_functions[1])

      elif '-e' == current_arg or '--ejecution' == current_arg:
        current_arg_index += 1
        INITIAL_EJECUTION = int(sys.argv[current_arg_index])
        LAST_EJECUTION = INITIAL_EJECUTION

      elif '-E' == current_arg or '--ejecution-range' == current_arg:
        current_arg_index += 1
        range_ejecutions = sys.argv[current_arg_index].split(':')
        INITIAL_EJECUTION = int(range_ejecutions[0])
        LAST_EJECUTION = int(range_ejecutions[1])

      elif '-h' == current_arg:
        help_text = "-f, --function <number>              Ejecuta solo la funcion numero 'number'"
        help_text += "\n-F, --functions-range <init>:<last>  Ejecuta de la funcion 'init' hasta 'last'"
        help_text += "\n-e, --ejecution <number>             Ejecuta solo 'number' ejecuciones"
        help_text += "\n-E, --ejecutions-range <init>:<last> Ejecuta desde 'init' hasta 'last' ejecuciones"
        print(help_text)

      current_arg_index += 1


if __name__ == '__main__':

  # Arguments handling
  handle_args()

  main()

