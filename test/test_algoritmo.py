import cv2
from src.mainp2 import genetic_algorithm
from src.individuo import crear_imagen, crear_individuo

img = cv2.imread("Images/ucrania.jpg")
img = cv2.cvtColor(img, cv2.COLOR_RGB2RGBA)
img = cv2.resize(img, (100, 100)) 

opciones = {
    "num_poligonos": 25,
    "num_lados": 3,
    "num_individuos": 50,
    "num_generaciones": 40000,
    "num_seleccion_elite": 25,
    "prob_mutacion": 0.9,
    "cant_mutacion": 0.1,
    "metodo_de_cruza": "cruce_uniforme",
    "kwargs_cruza": {},
    "metodo_de_seleccion": "elite",
    "kwargs_seleccion": {"K": 5, "T":10, "dT":0.001, "num_ind_torneo": 5, "threshold_torneo": 0.5},
}


mejor,fit = genetic_algorithm(img, opciones)
mejor_img = crear_imagen(mejor)
cv2.imwrite("out.jpg", mejor_img)
cv2.imwrite("target.jpg", img)

