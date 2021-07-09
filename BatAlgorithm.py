from sklearn.cluster import DBSCAN
import numpy as np
import time
import math
import csv
from scripts.utils import *

MAX_BATS = 100
MIN_BATS =  10
INCREMENTS_BATS = 2
INCREMENTS_BATS_PER_CLUSTER = 3

class BatAlgorithm():
  def __init__(self, ejecution, BKS, D, NP, N_Gen, A, r, alpha, gamma, fmin, fmax, Lower, Upper, function):
    self.ejecution = ejecution
    self.BKS = BKS
    self.seed = int(time.time())

    # MH params
    self.D = D
    self.NP = NP
    self.N_Gen = N_Gen
    self.alpha = alpha
    self.gamma = gamma
    self.fmin = fmin
    self.fmax = fmax
    self.Lower = Lower
    self.Upper = Upper
    self.function = function
    self.A0 = A
    self.r0 = r
    
    # Se inicializa A (loudness) y r (pulse rate)
    self.A = [A for i in range(self.NP)]
    self.r = [r for i in range(self.NP)]
    
    self.freq = [0.0] * NP
    self.v = [[0.0 for i in range(self.D)] for j in range(self.NP)]
    self.x = np.zeros((NP, D))
    self.fitness = [0.0] * NP
    self.F_min = 0.0
    self.best = [0.0] * D
  
  def init_bats(self):
    np.random.seed(self.seed)

    # Se generan soluciones aleatorias (entre el rango establecido por las bandas)
    for i in range(self.NP):
      self.freq[i] = 0
      for j in range(self.D):
        random = np.random.uniform(0,1)
        self.v[i][j] = 0.0
        self.x[i][j] = self.Lower + (self.Upper - self.Lower) * random
      self.fitness[i] = self.function(self.x[i])
    
    # Se busca el mejor murcialago
    self.best_bat()
  
  def best_bat(self):
    j = 0
    for i in range(self.NP):
      if self.fitness[i] < self.fitness[j]:
        j = i
            
    self.set_best_bat(self.x[j], self.fitness[j])

  def set_best_bat(self, bat, fitness):
    for i in range(self.D):
      self.best[i] = bat[i]
    
    self.F_min = fitness

  
  def simple_bounds(self, value, lower, upper):
    if(value > upper):
      value = upper
        
    if(value < lower):
      value = lower
    
    return value

  def checkImprove(self, past_best, Amean):
    global INCREMENTS_BATS
    global INCREMENTS_BATS_PER_CLUSTER

    # Se empaquetan los datos, para que cada murcielago tenga sus datos juntos al ordenarlos
    l = list(zip(self.x, self.A, self.r, self.freq, self.fitness, self.v))

    # Se ordenan los murcielagos, a partir del valor del fitness
    ol = sorted(l, key=lambda y: y[4])

    # Se desempaquetan las listas ordenadas (llegan como tuplas)
    self.x, A, r, freq, fitness, v = list(zip(*ol))

    # Si la solucion ha mejorado, y no se ha llegado al limite se decrementan los murcielagos
    if past_best > self.F_min:
      if self.NP - INCREMENTS_BATS >= MIN_BATS:
        # Se decrementan la cantidad de murcielagos
        self.NP -= INCREMENTS_BATS

        # Se eliminan los peores murcielagos con sus datos de cada lista
        self.A = list(A[:-INCREMENTS_BATS])
        self.r = list(r[:-INCREMENTS_BATS])
        self.freq = list(freq[:-INCREMENTS_BATS])
        self.fitness = list(fitness[:-INCREMENTS_BATS])
        self.v = list(v[:-INCREMENTS_BATS])
        self.x = np.array(self.x[:-INCREMENTS_BATS])
        #print(f'Decrement {past_best} -> {self.F_min}, {self.x[0]}: {self.function(self.x[0])}')
    else:
      # Se vuelven a pasar a listas (ya que al ordenar con zip, llegan como tuplas)
      self.A = list(A)
      self.r = list(r)
      self.freq = list(freq)
      self.fitness = list(fitness)
      self.v = list(v)
      self.x = np.array(self.x)

      clusters = clusterize_solutions(self.x, 3)

      cant_clusters = np.unique(clusters.labels_).shape[0]

      # Si todos los muercielagos estan muy juntos, se reemplaza la mitad
      # Se guarda ademas si se cambio o no los murcielagos (en la variable "x_is_modified")
      x_is_modified = self.replace_cluster(clusters)

      # Si se modificaron las posiciones de los muercielagos, se actualiza el mejor murcielago
      if x_is_modified:
        self.best_bat()
      else:
        if self.NP + (cant_clusters * INCREMENTS_BATS_PER_CLUSTER) < MAX_BATS:
          # Sino se modificaron los murcielagos, y no se alcanzo el limite, se incrementa la poblacion de murcielagos
          self.increment_cluster(clusters, Amean)

          # Se guarda la cantidad de murcielagos que se agregaron, para despues eliminar la misma cantidad
          INCREMENTS_BATS = cant_clusters * INCREMENTS_BATS_PER_CLUSTER
    
  def increment_cluster(self, clusters, Amean):
    x_is_modified = False
    best_bat_clusters = {l: {'index': [], 'cant': 0} for l in np.unique(clusters.labels_)}

    # Se guardan los indices de los 2 mejores murcielagos de cada cluster
    for index, label in enumerate(clusters.labels_):
      if best_bat_clusters[label]['cant'] < INCREMENTS_BATS_PER_CLUSTER:
        best_bat_clusters[label]['index'].append(index)
        best_bat_clusters[label]['cant'] += 1

    # Se incrementa la poblacion de todos los clusters en 2, agregando 2 soluciones locales
    # de los mejores murcielagos de cada cluster
    for label in best_bat_clusters:
      for index in best_bat_clusters[label]['index']:
        # Se encuentra una nueva solucion local
        new_solution = np.empty(self.D)
        new_solution = self.generate_local_solution(new_solution, self.x[index], Amean)

        # Se ingresa la nueva solucion a los muercielagos
        self.x = np.append(self.x, [new_solution], axis=0)

        # Se ingresan los datos del nuevo muercielago
        self.freq.append(self.freq[index])
        self.A.append(self.A[index])
        self.r.append(self.r[index])
        self.v.append(self.v[index])
        self.fitness.append(self.function(new_solution))
        self.NP += 1

        x_is_modified = True
        
      print(self.NP, self.fitness[:4], self.fitness[-1], label)

    # Si se modificaron las posiciones de los muercielagos, se actualiza el mejor murcielago
    if x_is_modified:
      self.best_bat()

  def replace_cluster(self, clusters):
    # Diccionario que contiene la informacion para calcular el promedio de cada cluster.
    # Para cada cluster se puede acceder a su informacion por su label,
    # como valor guarda un diccionario para organizar su informacion
    fitness_clusters = {l: {'sum':0, 'total':0} for l in np.unique(clusters.labels_)}

    # Se obtiene la suma de los fitness y el total de elementos en cada cluster
    for (index, label) in enumerate(clusters.labels_):
      fitness_clusters[label]['sum'] += self.fitness[index]
      fitness_clusters[label]['total'] += 1

    x_is_modified = False

    for label in fitness_clusters:
      # Se calcula el promedio
      suma = fitness_clusters[label]['sum']
      total = fitness_clusters[label]['total']
      mean_cluster = suma / total

      if -1 <= self.F_min - mean_cluster <= 1:
        # Se reemplaza la mitad mas mala del cluster con soluciones aleatorias usando la funcion de exploracion
        cant = total // 2

        for index in range(self.NP - 1, -1, -1):
          if cant <= 0:
            break

          # Si el elemento actual pertenece al cluster que queremos repoblar
          if clusters.labels_[index] == label:
            self.x[index], self.fitness[index] = self.generate_random_solution(self.x[index])
            cant -= 1

          print(self.x[index][:5], self.fitness[index], index, label)

        x_is_modified = True
        
      print(self.F_min - mean_cluster, self.F_min, mean_cluster, label)

    return x_is_modified

  
  def move_bats(self, n_fun=1, name_logs_file='logs.csv', interval_logs=100):
    self.init_bats()
    solutions = np.zeros(self.D)

    past_best = self.F_min

    with open(name_logs_file, mode='w') as logs_file:
      initial_time = time.perf_counter()
      logs_writter = csv.writer(logs_file, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)

      logs_writter.writerow('function,ejecution,iteration,D,NP,N_Gen,A,r,fmin,fmax,lower,upper,alpha,gamma,time_ms,seed,BKS,fitness'.split(','))

      # Metaheuristic
      for t in range(self.N_Gen + 1):
        Amean = np.mean(self.A)

        if (t % interval_logs) == 0:
          # For logs purposes, not metaheuristic
          MH_params = f'{self.D},{self.NP},{self.N_Gen},{self.A0},{self.r0},{self.fmin}'
          MH_params += f',{self.fmax},{self.Lower},{self.Upper},{self.alpha},{self.gamma}'
          current_time = parseSeconds(time.perf_counter() - initial_time)
          log = f'{n_fun},{self.ejecution},{t},{MH_params},{current_time},{self.seed},{self.BKS},"{self.F_min}"'
          logs_writter.writerow(log.split(','))
          print('\n' + log)

          # Se ajusta la cantidad de murcielagos dependiendo del desempeño
          if t != 0:
            # LUEGO DE ESTO, LAS LISTAS ESTAN ORDENADAS POR FITNESS
            self.checkImprove(past_best, Amean)

            past_best = self.F_min

        for i in range(self.NP):
          # Ecuacion (2)
          beta = np.random.uniform(0, 1)
          self.freq[i] = self.fmin + (self.fmax - self.fmin) * beta

          for j in range(self.D):
            # Ecuaciones (3) y (4)
            self.v[i][j] = self.v[i][j] + (self.x[i][j] - self.best[j]) * self.freq[i]
            solutions[j] = self.simple_bounds(self.x[i][j] + self.v[i][j], self.Lower, self.Upper)
          
          random = np.random.uniform(0, 1)

          if(random > self.r[i]):
            solutions = self.generate_local_solution(solutions, self.best, Amean)
          
          fitness = self.function(solutions)
          
          random = np.random.uniform(0, 1)
          
          # Se ve si se acepta la nueva solucion
          if(random < self.A[i] and fitness < self.fitness[i]):
            self.fitness[i] = fitness
            for j in range(self.D):
              self.x[i][j] = solutions[j]

          # Si se encontro un mejor fitness, se actualizan algunas variables
          if(self.fitness[i] < self.F_min):
            self.set_best_bat(self.x[i], self.fitness[i])
          
            # Se actualizan A y r
            self.A[i] = self.A[i] * self.alpha
            self.r[i] = self.r0 * (1 - math.exp(-self.gamma * t))
              


  def generate_local_solution(self, solution, bat, Amean):
    '''
    Genera una nueva solucion local alrededor del murcielago "bat"
    '''
    for j in range(self.D):
      random = np.random.uniform(-1.0, 1.0)
      solution[j] = self.simple_bounds(bat[j] + random * Amean, self.Lower, self.Upper)

    return solution


  def generate_random_solution(self, solution):
    '''
    Genera una nueva solucion aleatoria para explorar el expacio de busqueda
    '''
    for j in range(self.D):
      random = np.random.uniform(0,1)
      solution[j] = self.Lower + (self.Upper - self.Lower) * random

    fitness = self.function(solution)

    return solution, fitness
