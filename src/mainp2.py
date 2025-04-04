from dataclasses import dataclass
import numpy as np
from PIL import Image, ImageDraw
import random
import cv2

# Tamaño del lienzo
WIDTH, HEIGHT = 256, 256  



def fitness(img1, img2):
    """Calcula el Error Cuadrático Medio (MSE) entre dos imágenes"""
    """Lo usamos como función de fitness"""
    img1 = np.array(img1).astype(np.float32)  # Convertir a array numpy
    img2 = np.array(img2).astype(np.float32)
    
    mse = np.mean((img1 - img2) ** 2)  # Error cuadrático medio
    return mse


def draw(individuo):
    "Crea una imagen con el cromosoma de un individuo"
    colores = individuo.cromosoma[:,-4:]                    # Las 4 ultimas columnas son canales RGBA
    shape = (individuo.n_poligonos, individuo.n_lados, 2)   # De un array flatten a [[p1, p2, p3], [p1, p2, p3] ...]
    poligonos = individuo.cromosoma[:,:-4].reshape(shape)
    
    img = np.zeros(individuo.img_dims)                      # Lienzo vacio
    for i in range(poligonos.shape[0]):                     # Armo matriz con triangulos dentro
        poligono = poligonos[i]
        color = colores[i].tolist()
        img = cv2.fillPoly(img, [poligono], color)
    return img


@dataclass
class Individuo:
    cromosoma: np.array     # encoding [x1, y1, ... xn, yn, R, G, B, A]
    costo: float            # comparativa con imagen original
    img_dims: tuple         # dimensiones imagen de referencia
    n_poligonos: int        # numero de poligonos
    n_lados: int            # numero de lados


def poligono_aleatorio(img, lados=3):
    "genera vector de la forma [x1, y1, x2, y2, ... xi, yi, cR, cG, cB, cA]"
    l, w, num_color = img.shape
    puntos = []
    color = np.random.randint(0, 256, num_color)
    x = np.random.randint(0, w+1, lados)
    y = np.random.randint(0, l+1, lados)
    puntos = np.vstack((x, y)).T.reshape(-1)
    return np.hstack((puntos, color))


def crear_individuo(img, num_poligonos, num_lados):
    "genera un individuo de la poblacion"
    ind = Individuo(
        cromosoma=np.vstack([poligono_aleatorio(img, num_lados) for i in range(num_poligonos)]),
        costo=0,
        img_dims=img.shape,
        n_poligonos=num_poligonos,
        n_lados=num_lados
    )
    return ind


def generate_population(img, M, N):
    "Genera una población de M individuos con N triángulos cada uno"
    return [crear_individuo(img, N, 3) for _ in range(M)]    


# img1 = draw_population(poblacion,1);
# img2 = draw_population(poblacion,2);


# print(fitness(img_ref, img1))


def genetic_algorithm(M,N,img_size,img_ref,Generations):
    """M es la cantidad de triangulos por individuo
       N es la cantidad de individuos en la población inicial
       img_size es le tamaño de la imagen
       img_ref es la imagen de referencia
       K es la cantidad de generaciones (podría haber otro indicio de parada)"""
       
    population_0 = generate_population(img, M, N)
    population = population_0; 
    for gen in range(Generations):
        
        # Obtengo las imagenes de cada una de las representaciones 
        # TODO Esto se podria paralelizar para mayor rapidez
        fitness_gen = []
        for m in range(M):
            img = draw(population,m)
            fitness_gen.append(fitness(img_ref, img))
        new_population = []
        
        # Realizamos la selección
        num_elites = M;
        Seleccionados = seleccion_elite(population_0, fitness_gen, num_elites)
        Parents = np.array(Seleccionados[:2], dtype=object)
        child1, child2 = crossover(Parents[0], Parents[1],2)
        
        # Agregamos los hijos
        new_population.append(child1)
        new_population.append(child2)
        
        # Rellenamos con los padres más aptos
        for i in range(M-2): # Completamos con los padres mas aptos
            new_population.append(Seleccionados[i])
        
        # Mutamos
        for i in range(M):
            new_population[i] = mutate(new_population[i], mutation_rate=0.1, img_size=img_size)
        
        fitness_vec=[]
        population = new_population;
        
        for m in range(M):
            img = draw_population(population,m);
            fitness_vec.append(fitness(img_ref, img))
        
        print("Puntaje fitness mìnimo:\t{}".format(min(fitness_vec)))
    return seleccion_elite(population, fitness_vec, 1)

def seleccion_elite(poblacion, fitness_values, num_elites):
    """
    Realiza la selección por élite en una población.

    Args:
        poblacion (list): Lista de individuos.
        fitness_values (list): Lista de valores de fitness asociados a la población.
        num_elites (int): Cantidad de individuos élite que se seleccionarán.

    Returns:
        list: Lista con los individuos élite seleccionados.
    """
    # Ordenar la población por fitness (de mayor a menor)
    indices_ordenados = np.argsort(fitness_values)[::-1]  # Orden descendente
    elite = [poblacion[i] for i in indices_ordenados[:num_elites]]
    
    return elite

def crossover(parent1, parent2,P):
    """ Crossover de dos individuos: mezcla de triángulos, P es la cantidad de Genes tomados"""
    
    child1 = np.hstack((parent1[:,:P], parent2[:,P:len(parent1[1])]))
    child2 = np.hstack((parent2[:,:P], parent1[:,P:len(parent1[1])]))

        
    print("Puntaje fitness mìnimo:\t{}".format(mejor.costo))
    



if __name__ == "__main__":
    # Generalmente se pone __name__ == "__main__" para evitar
    # que al importar este archivo se corra las lineas debajo.
    # Si se ejecuta solo el archivo si las corre.
    M = 9   # Número de individuos
    N = 40  # Triángulos por individuo
    img_size = (256, 256)  # Tamaño de la imagen
    img_ref = Image.open("src/blacksquare.jpg").convert("RGBA")
    img_ref = img_ref.resize(img_size)
    Generations = 100;
    individual = genetic_algorithm(M,N,img_size,img_ref,Generations)
    img1 = draw(individual,0)
    img1.show()

