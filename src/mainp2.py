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
    "seleccion": {"num_seleccion_elite": 15, "num_nuevos_individuos": 5, "metodo": "random", "K": 5, "T":10, "dT":0.001, "num_ind_torneo": 5, "threshold_torneo": 0.5},
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
    verbose             = opciones.get("verbose", False)

    # incializo
    poblacion = generar_poblacion(ref_img, num_individuos, num_poligonos, num_lados)
    for ind in poblacion:
        img = crear_imagen(ind)
        ind.costo = calcular_costo(img, ref_img)

    results = {
        "mejor_fitness_por_generacion": [],
        "peor_fitness_por_generacion": [],
        "tiempo_por_generacion": [],
        "mejor_individuo_por_generacion": []
    }
    
    tiempo_total = 0

    for gen in range(num_generaciones):
        tiempo = {"total": 0, "evaluacion": 0, "cruza": 0}
        
        # evaluo el fitness de cada uno
        # TODO Esto se podria paralelizar para mayor rapidez
        t0 = time()
        # for ind in poblacion:
        #     img = crear_imagen(ind)
        #     ind.costo = calcular_costo(img, ref_img)
            

        # ordeno en funcion de costo
        t1 = time()
        poblacion.sort(key=lambda ind: ind.costo)
        mejor = poblacion[0]
        peor = poblacion[-1]


        # obtengo la siguiente generacion
        t2 = time()
        nueva_poblacion = poblacion[:num_seleccion_elite]
        kwargs_seleccion.update(gen=gen) # para calculo de temperatura

        # Los posibles padres son una mezcla de los mejores,
        # los mejores seleccionados
        # y de nuevos individuos.
        posibles_padres = poblacion[:num_seleccion_elite]
        posibles_padres += seleccionar(poblacion, **kwargs_seleccion)
        nuevos_individuos = generar_poblacion(ref_img, kwargs_seleccion["num_nuevos_individuos"], num_poligonos, num_lados)
        for ind in nuevos_individuos:
            img = crear_imagen(ind)
            ind.costo = calcular_costo(img, ref_img)
        posibles_padres += nuevos_individuos

        while len(nueva_poblacion) < len(poblacion):        
            p1, p2 = np.random.choice(posibles_padres, 2)
            hijo = cruzar(p1, p2, **kwargs_cruza)
            hijo = mutar(hijo, **kwargs_mutacion)

            img = crear_imagen(hijo)
            hijo.costo = calcular_costo(img, ref_img)
            nueva_poblacion.append(hijo)#min(hijo, p1, p2, key=lambda x: x.costo))


        # Calculo metricas
        t3 = time()
        tiempo_total += t3-t0
        tiempo["total"] = t3-t0
        tiempo["cruza"] = t3-t2
        tiempo["evaluacion"] = t1-t0
        results["tiempo_por_generacion"].append(tiempo)
        results["mejor_fitness_por_generacion"].append(mejor.costo)
        results["peor_fitness_por_generacion"].append(peor.costo)
        results["mejor"] = mejor
           
        if gen % opciones["paso_para_resultados_detallados"] == 0:
            results["mejor_individuo_por_generacion"].append((gen, mejor))
            img = crear_imagen(mejor)
            cv2.imwrite("gen_{}.jpg".format(gen), img)
            
        if verbose:
            print("{}\t{:.4f}\t{:.4f}\t\t{:.2f}\t{:.2f}".format(gen, mejor.costo, peor.costo, tiempo["total"], tiempo_total))

        # Actualizo la nueva poblacion
        poblacion = nueva_poblacion

    return results
