import random

# Cities
cities = ['A', 'B', 'C', 'D', 'E']

# Distance matrix
distance = {
    'A': {'A': 0, 'B': 10, 'C': 15, 'D': 20, 'E': 25},
    'B': {'A': 10, 'B': 0, 'C': 35, 'D': 25, 'E': 30},
    'C': {'A': 15, 'B': 35, 'C': 0, 'D': 30, 'E': 20},
    'D': {'A': 20, 'B': 25, 'C': 30, 'D': 0, 'E': 15},
    'E': {'A': 25, 'B': 30, 'C': 20, 'D': 15, 'E': 0}
}

POPULATION_SIZE = 10
GENERATIONS = 100
MUTATION_RATE = 0.1
CROSSOVER_RATE = 0.8


# Calculate total distance of a route
def calculate_distance(route):
    total = 0

    for i in range(len(route) - 1):
        total += distance[route[i]][route[i + 1]]

    # Return to starting city
    total += distance[route[-1]][route[0]]

    return total


# Fitness function
def fitness(route):
    return 1 / calculate_distance(route)


# Create initial population
def create_population():
    population = []

    for _ in range(POPULATION_SIZE):
        route = cities.copy()
        random.shuffle(route)
        population.append(route)

    return population


# Selection
def selection(population):
    parent1 = random.choice(population)
    parent2 = random.choice(population)

    if fitness(parent1) > fitness(parent2):
        return parent1
    else:
        return parent2


# Order Crossover (OX)
def crossover(parent1, parent2):

    size = len(parent1)

    start, end = sorted(random.sample(range(size), 2))

    child = [None] * size

    # Copy part of parent1
    child[start:end] = parent1[start:end]

    # Fill remaining positions using parent2
    remaining = [city for city in parent2 if city not in child]

    j = 0

    for i in range(size):
        if child[i] is None:
            child[i] = remaining[j]
            j += 1

    return child


# Mutation: swap two cities
def mutation(route):

    if random.random() < MUTATION_RATE:

        i, j = random.sample(range(len(route)), 2)

        route[i], route[j] = route[j], route[i]


# Main Genetic Algorithm
population = create_population()

best_route = None
best_distance = float('inf')


for generation in range(GENERATIONS):

    new_population = []

    while len(new_population) < POPULATION_SIZE:

        # Select parents
        parent1 = selection(population)
        parent2 = selection(population)

        # Crossover
        if random.random() < CROSSOVER_RATE:
            child = crossover(parent1, parent2)
        else:
            child = parent1.copy()

        # Mutation
        mutation(child)

        # Add child
        new_population.append(child)

        # Update best solution
        child_distance = calculate_distance(child)

        if child_distance < best_distance:
            best_distance = child_distance
            best_route = child.copy()

    # Replace population
    population = new_population


# Output
print("Best Route:", best_route)
print("Minimum Distance:", best_distance)
