from sklearn.cluster import DBSCAN
import numpy as np
import time
import math
import csv
from scripts.utils import parseSeconds

MAX_BATS = 100
MIN_BATS = 5
INCREMENTS_BATS = 5

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
            
    for i in range(self.D):
      self.best[i] = self.x[j][i]
    
    self.F_min = self.fitness[j]
  
  def simple_bounds(self, value, lower, upper):
    if(value > upper):
      value = upper
        
    if(value < lower):
      value = lower
    
    return value

  def checkImprove(self, past_best, solutions):
    # Se empaquetan los datos, para que cada murcielago tenga sus datos juntos al ordenarlos
    l = list(zip(self.x, self.A, self.r, self.freq, self.fitness, self.v, solutions))

    # Se ordenan los murcielagos, a partir del valor del fitness
    ol = sorted(l, key=lambda y: y[4])

    # Se desempaquetan las listas ordenadas (llegan como tuplas)
    self.x, A, r, freq, fitness, v, solutions = list(zip(*ol))

    # Si la solucion no ha mejorado, y no se ha llegado al limite se incrementan los murcielagos
    if past_best == self.F_min:
      if self.NP + INCREMENTS_BATS <= MAX_BATS:
        # Se incrementan la cantidad de murcielagos
        self.NP += INCREMENTS_BATS

        # Se concatenan los datos de los mejores fitness al final de cada lista
        self.A = list(A) + [A[0]] * INCREMENTS_BATS
        self.r = list(r) + [r[0]] * INCREMENTS_BATS
        self.freq = list(freq) + [freq[0]] * INCREMENTS_BATS
        self.fitness = list(fitness) + [fitness[0]] * INCREMENTS_BATS
        self.v = list(v) + [v[0]] * INCREMENTS_BATS
        self.x = np.array(self.x + (self.x[0], ) * INCREMENTS_BATS)
        solutions = np.array(solutions + (solutions[0], ) * INCREMENTS_BATS)
        #print(f'Increment {past_best} -> {self.F_min}, {self.x[-1]}: {self.function(self.x[-1])}')
    else:
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
        solutions = np.array(solutions[:-INCREMENTS_BATS])
        #print(f'Decrement {past_best} -> {self.F_min}, {self.x[0]}: {self.function(self.x[0])}')

    return solutions
  
  def move_bats(self, n_fun=1, name_logs_file='logs.csv', interval_logs=100):
    self.init_bats()
    solutions = np.zeros((self.NP, self.D))

    past_best = self.F_min

    with open(name_logs_file, mode='w') as logs_file:
      initial_time = time.perf_counter()
      logs_writter = csv.writer(logs_file, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)

      logs_writter.writerow('function,ejecution,iteration,D,NP,N_Gen,A,r,fmin,fmax,lower,upper,alpha,gamma,time_ms,seed,BKS,fitness'.split(','))

      # Metaheuristic
      for t in range(self.N_Gen + 1):
        Arata2 = np.mean(self.A)

        if (t % interval_logs) == 0:
          # For logs purposes, not metaheuristic
          MH_params = f'{self.D},{self.NP},{self.N_Gen},{self.A0},{self.r0},{self.fmin}'
          MH_params += f',{self.fmax},{self.Lower},{self.Upper},{self.alpha},{self.gamma}'
          current_time = parseSeconds(time.perf_counter() - initial_time)
          log = f'{n_fun},{self.ejecution},{t},{MH_params},{current_time},{self.seed},{self.BKS},"{self.F_min}"'
          logs_writter.writerow(log.split(','))
          print(log)

          # Se ajusta la cantidad de murcielagos dependiendo del desempeño
          if t != 0:
            solutions = self.checkImprove(past_best, solutions)
            past_best = self.F_min

        for i in range(self.NP):
          # Ecuacion (2)
          beta = np.random.uniform(0, 1)
          self.freq[i] = self.fmin + (self.fmax - self.fmin) * beta

          for j in range(self.D):
            # Ecuaciones (3) y (4)
            self.v[i][j] = self.v[i][j] + (self.x[i][j] - self.best[j]) * self.freq[i]
            solutions[i][j] = self.simple_bounds(self.x[i][j] + self.v[i][j], self.Lower, self.Upper)
          
          random = np.random.uniform(0, 1)

          if(random > self.r[i]):
            for j in range(self.D):
              random = np.random.uniform(-1.0, 1.0)
              solutions[i][j] = self.simple_bounds(self.best[j] + random * Arata2, self.Lower, self.Upper)
          
          fitness = self.function(solutions[i])
          
          random = np.random.uniform(0, 1)
          
          if(random < self.A[i] and fitness < self.fitness[i]):
            self.fitness[i] = fitness
            for j in range(self.D):
              self.x[i][j] = solutions[i][j]
          
          if(self.fitness[i] < self.F_min):
            self.F_min = self.function(solutions[i])
            for j in range(self.D):
                self.best[j] = self.x[i][j] 
              
            # Se actualizan A y r
            self.A[i] = self.A[i] * self.alpha
            self.r[i] = self.r[0] * (1 - math.exp(-self.gamma * t))
