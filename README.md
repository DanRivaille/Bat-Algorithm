# Bat-Algorithm
Implementacion del Algoritmo de Murcielagos estandar propuesto por Xin-She Yang, 2010

Se agrega la habilidad para auto-ajustar sus parametros dependiendo del desempeño que tenga en la ejecucion.

Se uso como base el [repositorio](https://github.com/herukurniawan/bat-algorithm) escrito por Heru Purnomo Kurniawan, se realizaron algunas traducciones y otros cambios menores. Se aplanea agregar las funciones CEC 2021 para el benchmarking.

## Install
Para correr el algoritmo se tienen que instalar algunas dependencias entre otras cosas, los pasos serian los siguientes:

1. Crear el ambiente virtual:
```
$ python3 -m venv ./venv
```

2. Instalar las dependecias:
```
$ pip3 install -r requeriments.txt
```

3. Crear la carpeta de logs:
```
$ mkdir Logs
```
4. Probar que funcione:
```
$ python3 run.py

Function 2: {'lower': -5.0, 'upper': 5.0, 'threshold': 0, 'best': 0.0, 'dimension': 1000}

2,15,0,1000,30,5000,0.95,0.1,0,1,-5.0,5.0,0.9,0.5,0:0:0:0.000,1625840350,0.0,"125694.67361923706"
```
