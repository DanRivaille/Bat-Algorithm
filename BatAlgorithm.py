import numpy as np
import time
import math
import csv

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
    self.x = [[0.0 for i in range(self.D)] for j in range(self.NP)]
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
      self.fitness[i] = self.function(self.D, self.x[i])
    
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
  
  def simple_bounds(self, value):
    if(value > self.Upper):
      value = self.Upper
        
    if(value < self.Lower):
      value = self.Lower
    
    return value
  
  def move_bats(self):
    self.init_bats()
    solutions = [[0.0 for i in range(self.D)] for j in range(self.NP)]

    with open('logs.csv', mode='w') as logs_file:
      initial_time = time.perf_counter()
      logs_writter = csv.writer(logs_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

      logs_writter.writerow("ejecution,iteration,D,NP,N_Gen,A,r,fmin,fmax,lower,upper,alpha,gamma,time_ms,seed,BKS,fitness".split(','))

      # Metaheuristic
      for t in range(self.N_Gen + 1):
        Arata2 = np.mean(self.A)

        # For logs purposes, not metaheuristic
        if (t % 100) == 0:
          MH_params = f'{self.D},{self.NP},{self.N_Gen},{self.A0},{self.r0},{self.fmin}'
          MH_params += f',{self.fmax},{self.Lower},{self.Upper},{self.alpha},{self.gamma}'
          log = f'{self.ejecution},{t},{MH_params},{(time.perf_counter() - initial_time) * 1000},{self.seed},{self.BKS},{self.F_min}'
          logs_writter.writerow(log.split(','))
          print(log)

        for i in range(self.NP):
          # Ecuacion (2)
          beta = np.random.uniform(0, 1)
          self.freq[i] = self.fmin + (self.fmax - self.fmin) * beta

          for j in range(self.D):
            # Ecuaciones (3) y (4)
            self.v[i][j] = self.v[i][j] + (self.x[i][j] - self.best[j]) * self.freq[i]
            solutions[i][j] = self.simple_bounds(self.x[i][j] + self.v[i][j])
          
          random = np.random.uniform(0, 1)

          if(random > self.r[i]):
            for j in range(self.D):
              random = np.random.uniform(-1.0, 1.0)
              solutions[i][j] = self.simple_bounds(self.best[j] + random * Arata2)
          
          fitness = self.function(self.D, solutions[i])
          
          random = np.random.uniform(0, 1)
          
          if(random < self.A[i] and fitness < self.fitness[i]):
            self.fitness[i] = fitness
            for j in range(self.D):
              self.x[i][j] = solutions[i][j]
          
          if(self.fitness[i] < self.F_min):
            self.F_min = self.function(self.D, solutions[i])
            for j in range(self.D):
                self.best[j] = self.x[i][j] 
              
            # Se actualizan A y r
            self.A[i] = self.A[i] * self.alpha
            self.r[i] = self.r[0] * (1 - math.exp(-self.gamma * t))
