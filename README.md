# Bat-Algorithm
Implementacion del Algoritmo de Murcielagos estandar propuesto por Xin-She Yang, 2010

Se agrega la habilidad para auto-ajustar sus parametros dependiendo del desempeño que tenga en la ejecucion logrando así una poblacion dinámica.
Se agregaron las funciones CEC 2021 (2013) para el benchmarking.

Se uso como base el [repositorio](https://github.com/herukurniawan/bat-algorithm) escrito por Heru Purnomo Kurniawan, se realizaron algunas traducciones y otros cambios menores.

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

3. Crear la carpeta de logs y los logs de los clusters:
```
$ mkdir Logs
$ mkdir Logs/clusters
```
4. Probar que funcione:
```
$ source venv/bin/activate
(venv) $ python3 run.py

Function 2: {'lower': -5.0, 'upper': 5.0, 'threshold': 0, 'best': 0.0, 'dimension': 1000}

2,15,0,1000,30,5000,0.95,0.1,0,1,-5.0,5.0,0.9,0.5,0:0:0:0.000,1625840350,0.0,"125694.67361923706"
```
## Use
Si se corre el archivo run.py sin argumentos ejecutará por defecto la funcion 1 hasta la 15, y en cada una realizará 31 ejecuciones, aunque esto se puede establacer con los siguientes argumentos:

### Arguments
```
-f, --function <number>              Ejecuta solo la funcion numero 'number'
-F, --functions-range <init>:<last>  Ejecuta de la funcion 'init' hasta 'last'
-e, --ejecution <number>             Ejecuta solo 'number' ejecuciones
-E, --ejecutions-range <init>:<last> Ejecuta desde 'init' hasta 'last' ejecuciones
-A, --autonomous                     Ejecuta la MH con el autonomo
-O, --original                       Ejecuta la MH original (opcion por defecto)
-h                                   Muestra los comandos disponibles
```
### Examples
* Ejecuta solo la funcion 7 con el autonomo (31 ejecuciones por defecto)
```
(venv) $ python3 run.py -f 7 -A
```

* Ejecuta la funcion 2 hasta la 10 (31 ejecuciones por defecto en cada una) con la metaheuristica original
```
(venv) $ python3 run.py -F 2:10
```

* Ejecuta solo la funcion 3, y desde la ejecucion 14 hasta la 31, con el autonomo
```
(venv) $ python3 run.py -f 3 -E 14:31 -A
```
