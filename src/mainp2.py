from dataclasses import dataclass
from multiprocessing import Pool
from joblib import Parallel, delayed
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
    "num_seleccion_elite": 15,
    "prob_mutacion": 0.1,
    "cant_mutacion": 0.1,
    "metodo_de_cruza": "cruce_un_punto",
    "kwargs_cruza": {},
    "metodo_de_seleccion": "random",
    "kwargs_seleccion": {"K": 5, "T":10, "dT":0.001, "num_ind_torneo": 5, "threshold_torneo": 0.5},
}


def genetic_algorithm(ref_img, opciones=opciones):
    # leo las opciones
    num_poligonos       = opciones["num_poligonos"]
    num_lados           = opciones["num_lados"]
    num_individuos      = opciones["num_individuos"]
    num_generaciones    = opciones["num_generaciones"]
    num_seleccion_elite = opciones["num_seleccion_elite"]
    prob_mutacion       = opciones["prob_mutacion"]
    cant_mutacion       = opciones["cant_mutacion"]
    metodo_de_cruza     = opciones["metodo_de_cruza"]
    kwargs_cruza        = opciones["kwargs_cruza"]
    metodo_de_seleccion = opciones["metodo_de_seleccion"]
    kwargs_seleccion    = opciones["kwargs_seleccion"]


    poblacion = generar_poblacion(ref_img, num_individuos, num_poligonos)
    print("gen", "costo", "t.cost", "t.img", "t.eval", "t.cruzar", sep="\t")
    
    for gen in range(num_generaciones):
        
        # evaluo el fitness de cada uno
        # TODO Esto se podria paralelizar para mayor rapidez
        t0 = time()
        t_crea = 0
        t_eval = 0
        
        for ind in poblacion:
            ta = time()
            img = crear_imagen(ind)
            tb = time()
            t_crea += tb - ta
            
            ta = time()
            ind.costo = calcular_costo(img, ref_img)
            tb = time()
            t_eval += tb - ta

        # ordeno en funcion de costo
        t1 = time()
        poblacion.sort(key=lambda ind: ind.costo)
        mejor = poblacion[0]


        # obtengo la siguiente generacion
        t2 = time()
        nueva_poblacion = poblacion[:num_seleccion_elite]
        
        kwargs_seleccion.update(gen=gen)
        posibles_padres = seleccionar(poblacion, metodo_de_seleccion, **kwargs_seleccion)
        
        while len(nueva_poblacion) < len(poblacion):
            p1, p2 = np.random.choice(posibles_padres, 2)
            hijo = cruzar(p1, p2, metodo_de_cruza, **kwargs_cruza)
            hijo = mutar(hijo, prob_mutacion, cant_mutacion)
            nueva_poblacion.append(hijo)
        poblacion = nueva_poblacion
        
        t3 = time()
        print(gen,
            "{:.0f}".format(mejor.costo),
            "{:.3f}".format(t1-t0),
            "{:.2f}".format(t_crea),
            "{:.2f}".format(t_eval),
            "{:.3f}".format(t3-t2),
            sep="\t")
        
        if gen % 10 == 0:
            img = crear_imagen(mejor)
            cv2.imwrite("out-{}.jpg".format(gen), img)

    return mejor
