from dataclasses import dataclass
from multiprocessing import Pool
import numpy as np
import cv2
from time import time

from .cruzar import cruzar
from .mutacion import mutar
from .individuo import generar_poblacion, calcular_costo, crear_imagen
from .seleccion import seleccionar


opciones = {
    "num_poligonos": 10,
    "num_lados": 3,
    "num_individuos": 10,
    "num_generaciones": 100,
    "mutacion": {"prob_mutacion": 0.1, "cant_mutacion": 0.1,},
    "cruza": {"metodo": "cruce_un_punto"},
    "seleccion": {"num_seleccion_elite": 15, "metodo": "random", "K": 5, "T":10, "dT":0.001, "num_ind_torneo": 5, "threshold_torneo": 0.5},
    "paso_para_resultados_detallados": 5
}


def genetic_algorithm(ref_img, opciones=opciones):
    # leo las opciones
    num_poligonos       = opciones["num_poligonos"]
    num_lados           = opciones["num_lados"]
    num_individuos      = opciones["num_individuos"]
    num_generaciones    = opciones["num_generaciones"]    
    kwargs_mutacion     = opciones["mutacion"]
    kwargs_cruza        = opciones["cruza"]
    num_seleccion_elite = opciones["seleccion"]["num_seleccion_elite"]
    kwargs_seleccion    = opciones["seleccion"]


    poblacion = generar_poblacion(ref_img, num_individuos, num_poligonos, num_lados)
    
    results = {
        "mejor_fitness_por_generacion": [],
        "peor_fitness_por_generacion": [],
        "tiempo_por_generacion": [],
        "mejor_individuo_por_generacion": []
    }
    
    for gen in range(num_generaciones):
        tiempo = {"total": 0, "evaluacion": 0, "cruza": 0}
        
        # evaluo el fitness de cada uno
        # TODO Esto se podria paralelizar para mayor rapidez
        t0 = time()
        for ind in poblacion:
            img = crear_imagen(ind)
            ind.costo = calcular_costo(img, ref_img)
            

        # ordeno en funcion de costo
        t1 = time()
        poblacion.sort(key=lambda ind: ind.costo)
        mejor = poblacion[0]


        # obtengo la siguiente generacion
        t2 = time()
        nueva_poblacion = poblacion[:num_seleccion_elite]
        kwargs_seleccion.update(gen=gen)
        posibles_padres = seleccionar(poblacion, **kwargs_seleccion)
        
        while len(nueva_poblacion) < len(poblacion):
            p1, p2 = np.random.choice(posibles_padres, 2)
            hijo = cruzar(p1, p2, **kwargs_cruza)
            hijo = mutar(hijo, **kwargs_mutacion)
            nueva_poblacion.append(hijo)
        poblacion = nueva_poblacion
        

        # Calculo metricas
        t3 = time()
        tiempo["total"] = t3-t0
        tiempo["cruza"] = t3-t2
        tiempo["evaluacion"] = t1-t0
        results["tiempo_por_generacion"].append(tiempo)
        results["mejor_fitness_por_generacion"].append(poblacion[0].costo)
        results["peor_fitness_por_generacion"].append(poblacion[-1].costo)
        results["mejor"] = mejor
           
        if gen % opciones["paso_para_resultados_detallados"]:
            results["mejor_individuo_por_generacion"].append((gen, mejor))
            
    return results
