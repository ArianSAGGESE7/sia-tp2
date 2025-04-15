import cv2
from src.mainp2 import genetic_algorithm
from src.individuo import crear_imagen, crear_individuo


img = cv2.imread("Images/starrynight.jpg")
img = cv2.cvtColor(img, cv2.COLOR_RGB2RGBA)
img = cv2.resize(img, (75, 75)) 


opciones = {
    "num_poligonos": 5,
    "num_lados": 3,
    "num_individuos": 50,
    "num_generaciones": 301,
    "mutacion": {
        "prob_mutacion": 0.30,
        "cant_mutacion": 0.80
        },
    "cruza": {
        "metodo": "cruce_n_puntos",
        "n": 10,
        },
    "seleccion": {
        "num_seleccion_elite": 10,
        "num_nuevos_individuos": 15,
        "metodo": "ruleta",
        "K": 15,
        "T": 1,
        "dT": 0.1,
        "num_ind_torneo": 10,
        "threshold_torneo": 0.1},
    "paso_para_resultados_detallados": 50,
    "verbose": True
}

metricas = genetic_algorithm(img, opciones)
