import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import levy
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

population_size = 50
generations = 100
alpha = 0.5
beta = 0.1

def initialize_population(size, dimensions):
    population = np.random.rand(size, dimensions)
    opposition_population = 1 - population
    return np.vstack((population, opposition_population))

def fitness_function(X, y, individual):
    selected_features = np.where(individual >= 0.5)[0]
    if len(selected_features) == 0:
        return float('inf')
    X_selected = X[:, selected_features]
    knn = KNeighborsClassifier(n_neighbors=5, weights='distance')
    knn.fit(X_selected, y)
    accuracy = knn.score(X_selected, y)
    return 1 - accuracy + 0.01 * len(selected_features)

def update_position(current_position, leader_position, alpha):
    r1 = np.random.rand()
    r2 = np.random.rand()
    r3 = np.random.rand()
    r4 = np.random.rand()
    if r4 < 0.5:
        return current_position + r1 * np.sin(r2) * abs(r3 * leader_position - current_position)
    else:
        return current_position + r1 * np.cos(r2) * abs(r3 * leader_position - current_position)

def levy_flight(position):
    return position + levy.rvs(size=position.shape)

def separating_operator(population, fitness, worst_indices, best_individual):
    new_population = population.copy()
    for idx in worst_indices:
        if np.random.rand() > 0.5:
            new_population[idx] = best_individual + np.random.randn(*best_individual.shape) * 0.1
        else:
            new_population[idx] = np.random.rand(*best_individual.shape)
    return new_population

def elitism(population, fitness, elite_size):
    elite_indices = np.argsort(fitness)[:elite_size]
    return population[elite_indices]

def IEHO(X, y, population_size, generations, alpha, beta):
    dimensions = X.shape[1]
    population = initialize_population(population_size, dimensions)
    fitness = np.array([fitness_function(X, y, ind) for ind in population])
    
    for gen in range(generations):
        leader = population[np.argmin(fitness)]
        new_population = []
        for i in range(population.shape[0]):
            if i != np.argmin(fitness):
                new_population.append(update_position(population[i], leader, alpha))
            else:
                new_population.append(population[i])
        population = np.array(new_population)
        
        for i in range(population.shape[0]):
            if np.random.rand() < beta:
                population[i] = levy_flight(population[i])
        
        worst_indices = np.argsort(fitness)[-int(0.1 * population.shape[0]):]
        population = separating_operator(population, fitness, worst_indices, leader)
        
        fitness = np.array([fitness_function(X, y, ind) for ind in population])
        
        elite_population = elitism(population, fitness, int(0.1 * population_size))
        population[:len(elite_population)] = elite_population
    
    best_individual = population[np.argmin(fitness)]
    selected_features = np.where(best_individual >= 0.5)[0]
    return selected_features

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

selected_features = IEHO(X_scaled, y, population_size, generations, alpha, beta)
print("Selected features:", selected_features)
print("Number of selected features:", len(selected_features))

X_selected = X_scaled[:, selected_features]

knn = KNeighborsClassifier(n_neighbors=5, weights='distance')
X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, random_state=42)
knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print('Accuracy:', accuracy)
