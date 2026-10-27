import math
import random


# -----------------------------------------
# 1. Problem data
# -----------------------------------------

# Node 0 = depot; nodes 1, 2, ... = customers
locations = [
    (0, 0),    # Depot
    (2, 3),    # Customer 1
    (5, 4),    # Customer 2
    (1, 6),    # Customer 3
    (7, 2),    # Customer 4
    (6, 6),    # Customer 5
    (3, 7),    # Customer 6
]

# Delivery demand of each node (depot demand = 0)
demands = [0, 2, 3, 2, 4, 2, 3]

vehicle_capacity = 7
num_vehicles = 3


# -----------------------------------------
# 2. Calculate distance matrix
# -----------------------------------------

n = len(locations)

distance = [[0.0] * n for _ in range(n)]

for i in range(n):
    for j in range(n):
        x1, y1 = locations[i]
        x2, y2 = locations[j]

        distance[i][j] = math.hypot(x2 - x1, y2 - y1)


# -----------------------------------------
# 3. ACO parameters
# -----------------------------------------

num_ants = 30
num_iterations = 200

alpha = 1.0       # Pheromone importance
beta = 3.0        # Distance importance
evaporation = 0.5
Q = 100.0         # Pheromone deposit constant

random.seed(42)

# Initial pheromone levels
pheromone = [
    [1.0 if i != j else 0.0 for j in range(n)]
    for i in range(n)
]


# -----------------------------------------
# 4. Calculate route distance
# -----------------------------------------

def route_distance(route):
    total = 0.0

    for i in range(len(route) - 1):
        total += distance[route[i]][route[i + 1]]

    return total


# -----------------------------------------
# 5. Construct a solution for one ant
# -----------------------------------------

def construct_solution():
    unvisited = set(range(1, n))
    routes = []

    for _ in range(num_vehicles):
        if not unvisited:
            break

        route = [0]
        current = 0
        remaining_capacity = vehicle_capacity

        while True:
            # Customers that fit in the current vehicle
            candidates = [
                c for c in unvisited
                if demands[c] <= remaining_capacity
            ]

            if not candidates:
                break

            # Calculate selection probabilities
            weights = []

            for customer in candidates:
                tau = pheromone[current][customer] ** alpha
                eta = 1.0 / max(distance[current][customer], 1e-10)
                weights.append(tau * (eta ** beta))

            total_weight = sum(weights)

            if total_weight <= 0:
                next_customer = random.choice(candidates)
            else:
                probabilities = [
                    w / total_weight for w in weights
                ]
                next_customer = random.choices(
                    candidates,
                    weights=probabilities,
                    k=1
                )[0]

            route.append(next_customer)
            remaining_capacity -= demands[next_customer]
            unvisited.remove(next_customer)
            current = next_customer

        # Return vehicle to depot
        route.append(0)
        routes.append(route)

    # Invalid if customers remain unserved
    if unvisited:
        return None, float("inf")

    total_distance = sum(route_distance(r) for r in routes)

    return routes, total_distance


# -----------------------------------------
# 6. Update pheromone trails
# -----------------------------------------

def update_pheromone(routes, total_distance):
    global pheromone

    # Evaporation
    for i in range(n):
        for j in range(n):
            pheromone[i][j] *= (1.0 - evaporation)

            # Prevent pheromone from vanishing completely
            pheromone[i][j] = max(pheromone[i][j], 1e-8)

    # Deposit pheromone on selected routes
    if total_distance == float("inf"):
        return

    deposit = Q / max(total_distance, 1e-10)

    for route in routes:
        for i in range(len(route) - 1):
            a = route[i]
            b = route[i + 1]

            pheromone[a][b] += deposit
            pheromone[b][a] += deposit


# -----------------------------------------
# 7. Main ACO optimization loop
# -----------------------------------------

best_routes = None
best_distance = float("inf")

for iteration in range(num_iterations):
    iteration_best_routes = None
    iteration_best_distance = float("inf")

    for ant in range(num_ants):
        routes, total_distance = construct_solution()

        if total_distance < iteration_best_distance:
            iteration_best_routes = routes
            iteration_best_distance = total_distance

        if total_distance < best_distance:
            best_routes = routes
            best_distance = total_distance

    # Reinforce the best solution of this iteration
    if iteration_best_routes is not None:
        update_pheromone(
            iteration_best_routes,
            iteration_best_distance
        )

    if (iteration + 1) % 20 == 0:
        if best_routes is None:
            print(
                f"Iteration {iteration + 1}: "
                "No feasible solution found yet"
            )
        else:
            print(
                f"Iteration {iteration + 1}: "
                f"Best distance = {best_distance:.2f}"
            )


# -----------------------------------------
# 8. Display optimized vehicle routes
# -----------------------------------------

print("\n========== OPTIMIZED ROUTES ==========")

if best_routes is None:
    print("No feasible solution found.")
    print("Try increasing the number of vehicles or capacity.")
else:
    for i, route in enumerate(best_routes, start=1):
        load = sum(demands[node] for node in route)

        print(f"\nVehicle {i}")
        print("Route:", " -> ".join(map(str, route)))
        print(f"Load: {load}/{vehicle_capacity}")
        print(f"Distance: {route_distance(route):.2f}")

    print(f"\nTotal distance: {best_distance:.2f}")
