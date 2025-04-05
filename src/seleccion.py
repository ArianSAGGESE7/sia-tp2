"""
Este archivo contiene:
- funciones para seleccionar los diferentes individuos de la poblacion
- ponder los otros esquemas
"""

import numpy as np
import random

from src.individuo import Individuo

def _random(poblacion, **kwargs):
    len_poblacion = len(poblacion)
    return random.choices(poblacion[:len_poblacion//2], k=2)


def _elite(poblacion, K, **kwargs):
    return poblacion[:K]


def _ruleta(poblacion, K, **kwargs):
    costo = np.array([ind.costo for ind in poblacion])
    costo_acumulado = np.cumsum(costo) / sum(costo)
    
    seleccion = []
    numeros = np.random.uniform(size=K)
    for numero in numeros:
        index = sum(numero > costo_acumulado)
        seleccion.append(poblacion[index])
    return seleccion


def _universal(poblacion, K, **kwargs):
    costo = np.array([ind.costo for ind in poblacion])
    costo_acumulado = np.cumsum(costo) / sum(costo)
    
    seleccion = []
    numeros = np.random.uniform(size=K)
    for j, numero in enumerate(numeros):
        index = sum((numero + j)/K > costo_acumulado)
        seleccion.append(poblacion[index])
    return seleccion


def _ranking(poblacion, K, **kwargs):
    "en vez de ordenarlos por costo, los pone en orden segun su fitness"
    
    #       costo   ruleta   ranking
    # 1     1        1/20    (5-1)/5 = 16/20
    # 2     4        4/20    (5-2)/5 = 12/20
    # 3     5        5/20    (5-3)/5 =  8/20
    # 4     10      10/20    (5-4)/5 =  4/20
    # total 20      1.0      1.0
    
    ranking = np.array([i for i, _ in enumerate(poblacion)])
    pseudo_costo = (len(poblacion) - ranking) / len(poblacion)
    costo_acumulado = np.cumsum(pseudo_costo) / sum(pseudo_costo)
    
    seleccion = []
    numeros = np.random.uniform(size=K)
    for numero in numeros:
        index = sum(numero > costo_acumulado)
        seleccion.append(poblacion[index])
    return seleccion


def _boltzmann(poblacion, K, T, dT, gen, **kwargs):
    cT = T * (1 - dT)**gen
    costo_max = poblacion[-1].costo
    pseudo_costo = np.array([np.exp(ind.costo / (cT * costo_max)) for ind in poblacion])       
    costo_acumulado = np.cumsum(pseudo_costo) / sum(pseudo_costo)
    
    print(cT)
    seleccion = []
    numeros = np.random.uniform(size=K)
    for numero in numeros:
        index = sum(numero > costo_acumulado)
        seleccion.append(poblacion[index])

    return seleccion


def _torneo_deterministico(poblacion, num_ind_torneo, **kwargs):
    torneo = random.choices(poblacion, k=num_ind_torneo)
    torneo.sort(key=lambda ind: ind.costo)
    return torneo[0]


def _torneo_probabilistico(poblacion, num_ind_torneo, threshold_torneo, **kwargs):
    torneo = random.choices(poblacion, k=num_ind_torneo)
    r = np.random.uniform()
    if r < threshold_torneo:
        return min(torneo, key=lambda ind: ind.costo)
    else:
        return max(torneo, key=lambda ind: ind.costo)


funcion_seleccion = {
    "random": _random,
    "elite": _elite,
    "ruleta": _ruleta,
    "universal": _universal,
    "ranking": _ranking,
    "boltzmann": _boltzmann,
    "torneo_deterministico": _torneo_deterministico,
    "torneo_probabilistico": _torneo_probabilistico,
}   


def seleccionar(poblacion, metodo, **kwargs):
    "selecciona K individuos de acuerdo a un metodo. Se asume poblacion ya ordenado por fitness"
    seleccionados = []
    while len(seleccionados) <= kwargs["K"]:
        s = funcion_seleccion[metodo](poblacion, **kwargs)
        if isinstance(s, Individuo):
            seleccionados.append(s)
        elif len(s) > 1:
            seleccionados.extend(s)
    return seleccionados
    