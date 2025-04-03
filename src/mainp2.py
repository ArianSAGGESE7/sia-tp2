import numpy as np
from PIL import Image, ImageDraw
import random

# Tamaño del lienzo
WIDTH, HEIGHT = 256, 256  



def fitness(img1, img2):
    """Calcula el Error Cuadrático Medio (MSE) entre dos imágenes"""
    """Lo usamos como función de fitness"""
    img1 = np.array(img1).astype(np.float32)  # Convertir a array numpy
    img2 = np.array(img2).astype(np.float32)
    
    mse = np.mean((img1 - img2) ** 2)  # Error cuadrático medio
    return mse


def draw_population(individuos,M):
    "Dibuja M individuos en una dada imagen"
    
    individuo = individuos[M]
    imagen_gen= Image.new("RGBA", (256, 256), (255, 255, 255, 255))  # Lienzo blanco
    draw = ImageDraw.Draw(imagen_gen, "RGBA")  # Permite transparencia

    for triangulo in range(len(individuo)):
        triangulo_color = individuo[triangulo]
        triangulo = tuple((triangulo_color[:3]))
        color = triangulo_color[3]
        draw.polygon(triangulo, fill=color) 
             
        # imagen_gen.show()
    
    return imagen_gen

def generate_population(M, N, img_size):
    """Genera una población de M individuos con N triángulos cada uno"""
    width, height = img_size
    poblacion = []
    
    for _ in range(M):  # Para cada individuo
        individuo = []
        for _ in range(N):  # Para cada triángulo
            triangulo = [
                (random.randint(0, width), random.randint(0, height)),  # Punto 1
                (random.randint(0, width), random.randint(0, height)),  # Punto 2
                (random.randint(0, width), random.randint(0, height)),  # Punto 3
                (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255), random.randint(50, 200))  # Color RGBA
            ]
            individuo.append(triangulo)
        poblacion.append(individuo)  # Guardar individuo en la población
    
    return np.array(poblacion, dtype=object)  # Convertir a arreglo numpy



# Generar población
# poblacion = generate_population(M, N, img_size)

# img1 = draw_population(poblacion,1);
# img2 = draw_population(poblacion,2);


# print(fitness(img_ref, img1))


def genetic_algorithm(M,N,img_size,img_ref,Generations):
    """M es la cantidad de triangulos por individuo
       N es la cantidad de individuos en la población inicial
       img_size es le tamaño de la imagen
       img_ref es la imagen de referencia
       K es la cantidad de generaciones (podría haber otro indicio de parada)"""
       
    population_0 = generate_population(M,N,img_size)
    population = population_0; 
    for gen in range(Generations):
        
        # Obtengo las imagenes de cada una de las representaciones 
        fitness_gen = []
        for m in range(M):
            img = draw_population(population,m);
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
            new_population[i] = mutate(new_population[i], mutation_rate=0.1)
        
        fitness_vec=[]
        population = new_population;
        
        for m in range(M):
            img = draw_population(population,m);
            fitness_vec.append(fitness(img_ref, img))
        
        print("Puntaje fitness mìnimo:{}".format(min(fitness_vec)))
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

    return child1, child2

def mutate(individual, mutation_rate=0.1):
    """ Aplica mutación a un individuo modificando sus triángulos """
    for i in range(len(individual)):
        if np.random.rand() < mutation_rate:
            individual[i] = generate_population(1, 1, img_size) # Genera un nuevo triángulo
    return individual




M = 9   # Número de individuos
N = 40  # Triángulos por individuo
img_size = (256, 256)  # Tamaño de la imagen
img_ref = Image.open("blacksquare.jpg").convert("RGBA")
img_ref = img_ref.resize(img_size)
Generations = 10000;
individual = genetic_algorithm(M,N,img_size,img_ref,Generations)
img1 = draw_population(individual,0)
img1.show()

