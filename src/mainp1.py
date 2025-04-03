import numpy as np
import cv2
import random
import string
import matplotlib.pyplot as plt
from pathlib import Path
import os
ASCII_CHARS = "@. "  # Caracteres que definen la composición del a imagen

# Convertir imagen 
def preprocess_image(image_path, size=32):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Cargar imagen en escala de grises
    img = cv2.resize(img, (size, size))  # Redimensionar a tamaño deseado
    img = img / 255.0  # Normalizar valores entre 0 y 1 (puede ser 256)
    return img

# Generar un individuo aleatorio como matriz de caracteres ASCII (esto da un aleatorío entre)
def generate_individual(size=32):
    return np.random.choice(list(ASCII_CHARS), (size, size)) 

# Función de aptitud: compara la imagen objetivo con el individuo
def fitness(individual, target_image):
    ascii_intensity = np.array([ASCII_CHARS.index(c) for c in individual.flat])
    ascii_intensity = ascii_intensity.reshape(individual.shape)
    ascii_intensity = ascii_intensity / (len(ASCII_CHARS) - 1)  # Normalizar
    error = np.sum((ascii_intensity - target_image) ** 2)  # Error cuadrático
    return -error  # Maximizar similitud (minimizar error)

# Selección de padres mediante torneo
def tournament_selection(population, fitness_scores, k=5):
    selected_indices = random.sample(range(len(population)), k)
    best_index = max(selected_indices, key=lambda i: fitness_scores[i])
    return population[best_index]

# Cruce de dos individuos (mezcla de caracteres ASCII)
def crossover(parent1, parent2):
    size = parent1.shape[0]
    mask = np.random.rand(size, size) > 0.5  # Matriz booleana aleatoria
    child = np.where(mask, parent1, parent2)  # Seleccionar de ambos padres
    return child

# Mutación: cambia algunos caracteres aleatoriamente
def mutate(individual, mutation_rate=0.05):
    size = individual.shape[0]
    num_mutations = int(size * size * mutation_rate)  # Cantidad de mutaciones
    for _ in range(num_mutations):
        i, j = np.random.randint(0, size, 2)  # Elegir posición aleatoria
        individual[i, j] = random.choice(ASCII_CHARS)  # Nuevo carácter aleatorio
    return individual

# Algoritmo genético principal
def genetic_algorithm(image_path, size=32, pop_size=50, generations=100, mutation_rate=0.05):
    target_image = preprocess_image(image_path, size)  # Imagen objetivo
    population = [generate_individual(size) for _ in range(pop_size)]  # Población inicial
    
    for gen in range(generations):
        fitness_scores = [fitness(ind, target_image) for ind in population]  # Evaluar aptitud
        new_population = []
        
        for _ in range(pop_size):
            parent1 = tournament_selection(population, fitness_scores)
            parent2 = tournament_selection(population, fitness_scores)
            child = crossover(parent1, parent2)
            child = mutate(child, mutation_rate)
            new_population.append(child)
        
        population = new_population  # Reemplazar con la nueva generación
        best_individual = population[np.argmax(fitness_scores)]  # Mejor individuo
        print(f"Generación {gen+1}: Mejor aptitud {max(fitness_scores):.5f}")
    
    return best_individual

# Función para visualizar la representación ASCII
def display_ascii_art(ascii_matrix):
    for row in ascii_matrix:
        print("".join(row))


best_ascii = genetic_algorithm('blacksquare.jpg', size=32, generations=10000)
display_ascii_art(best_ascii)
