import cv2	import cv2
from src.mainp2 import genetic_algorithm
from src.individuo import crear_imagen, crear_individuo



def test_00():
    "ejemplo de uso"
    img = cv2.imread("Images/image_00.png")
    img = cv2.cvtColor(img, cv2.COLOR_RGB2RGBA)
    img = cv2.resize(img, (100, 100)) 


    opciones = {
        "num_poligonos": 1,
        "num_lados": 3,
        "num_individuos": 100,
        "num_generaciones": 200,
        "mutacion": {
            "prob_mutacion": 0.9,
            "cant_mutacion": 0.1
            },
        "cruza": {
            "metodo": "cruce_uniforme"
            },
        "seleccion": {
            "num_seleccion_elite": 3,
            "num_nuevos_individuos": 5,
            "metodo": "torneo_deterministico",
            "K": 35,
            "T": 1,
            "dT": 0.1,
            "num_ind_torneo": 5,
            "threshold_torneo": 0.5},
        "paso_para_resultados_detallados": 2,
        "verbose": True
    }

    metricas = genetic_algorithm(img, opciones)
    assert isinstance(metricas, dict)


def test_01():
    "ejemplo de uso"

    img = cv2.imread("Images/blacksquare.jpg")
    img = cv2.cvtColor(img, cv2.COLOR_RGB2RGBA)
    img = cv2.resize(img, (100, 100)) 

    opciones = {
        "num_poligonos": 2,
        "num_lados": 4,
        "num_individuos": 100,
        "num_generaciones": 200,
        "mutacion": {
            "prob_mutacion": 0.1,
            "cant_mutacion": 0.1
            },
        "cruza": {
            "metodo": "cruce_un_punto"
            },
        "seleccion": {
            "num_seleccion_elite": 3,
            "num_nuevos_individuos": 5,
            "metodo": "random",
            "K": 5,
            "T": 1,
            "dT": 0.1,
            "num_ind_torneo": 5,
            "threshold_torneo": 0.5},
        "paso_para_resultados_detallados": 5
    }

    metricas = genetic_algorithm(img, opciones)

    mejor = metricas["mejor"]
    img = crear_imagen(mejor)
    # cv2.imwrite("out.jpg", img)
    assert isinstance(metricas, dict)

def test_002():
    img = cv2.imread("Images/italia.jpg")
    img = cv2.cvtColor(img, cv2.COLOR_RGB2RGBA)
    img = cv2.resize(img, (50, 50)) 

    opciones = {
        "num_poligonos": 3,
        "num_lados": 4,
        "num_individuos": 30,
        "num_generaciones": 5000,
        "mutacion": {
            "prob_mutacion": 0.2,
            "cant_mutacion": 0.9
            },
        "cruza": {
            "metodo": "cruce_uniforme"
            },
        "seleccion": {
            "num_seleccion_elite": 5,
            "num_nuevos_individuos": 5,
            "metodo": "ranking",
            "K": 10,
            "T": 1,
            "dT": 0.001,
            "num_ind_torneo": 20,
            "threshold_torneo": 0.2},
        "paso_para_resultados_detallados": 100,
        "verbose": True
    }

    metricas = genetic_algorithm(img, opciones)
    mejor = metricas["mejor"]
    res_img = crear_imagen(mejor)
    cv2.imwrite("out.jpg", res_img)
    cv2.imwrite("target.jpg", img)
