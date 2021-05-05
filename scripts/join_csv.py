logs_folder = '../Logs'

INITIAL_FUNCTION = 2
CANT_FUNCTIONS = 1
CANT_EJECUTIONS = 31

out_file_name = f'Functions.csv'

has_header = False

with open(out_file_name, mode='w') as out_file:

  for i in range(INITIAL_FUNCTION, INITIAL_FUNCTION + CANT_FUNCTIONS):
    for j in range(1, CANT_EJECUTIONS + 1):
      with open(f'{logs_folder}/Funcion{i}_{j}.csv', mode='r') as current_file:
        lines = current_file.readlines()

        if not has_header:
          out_file.write(lines[0])
          has_header = True

        for line in lines[1:]:
          out_file.write(line)
