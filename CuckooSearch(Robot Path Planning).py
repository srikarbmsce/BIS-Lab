import numpy as np
import math

# ==================================================
# 1. ROBOT ENVIRONMENT
# ==================================================

START = np.array([5.0, 5.0])
GOAL = np.array([95.0, 95.0])

X_MIN, X_MAX = 0, 100
Y_MIN, Y_MAX = 0, 100

# Circular obstacles (x, y, radius)
obstacles = [
    (25, 25, 9),
    (40, 50, 12),
    (60, 35, 10),
    (70, 70, 12),
    (30, 75, 8),
    (85, 45, 8)
]

ROBOT_RADIUS = 2.0


# ==================================================
# 2. CUCKOO SEARCH PARAMETERS
# ==================================================

N_WAYPOINTS = 8
N_NESTS = 30
MAX_ITER = 300

PA = 0.25
ALPHA = 0.5
BETA = 1.5

W_LENGTH = 1.0
W_COLLISION = 10000.0
W_SMOOTHNESS = 2.0

DIM = N_WAYPOINTS * 2

LOWER = np.tile([X_MIN, Y_MIN], N_WAYPOINTS)
UPPER = np.tile([X_MAX, Y_MAX], N_WAYPOINTS)


# ==================================================
# 3. GENERATE ROBOT PATH
# ==================================================

def create_path(solution):

    waypoints = solution.reshape(N_WAYPOINTS, 2)

    path = np.vstack((START, waypoints, GOAL))

    return path


# ==================================================
# 4. CALCULATE PATH LENGTH
# ==================================================

def path_length(path):

    total_length = 0.0

    for i in range(len(path) - 1):

        distance = np.linalg.norm(
            path[i + 1] - path[i]
        )

        total_length += distance

    return total_length


# ==================================================
# 5. POINT TO LINE SEGMENT DISTANCE
# ==================================================

def point_to_segment_distance(point, a, b):

    ab = b - a

    denominator = np.dot(ab, ab)

    if denominator == 0:

        return np.linalg.norm(point - a)

    t = np.dot(point - a, ab) / denominator

    t = np.clip(t, 0.0, 1.0)

    closest_point = a + t * ab

    return np.linalg.norm(point - closest_point)


# ==================================================
# 6. OBSTACLE COLLISION DETECTION
# ==================================================

def collision_penalty(path):

    penalty = 0.0

    for i in range(len(path) - 1):

        a = path[i]
        b = path[i + 1]

        for ox, oy, radius in obstacles:

            center = np.array([ox, oy])

            distance = point_to_segment_distance(
                center, a, b
            )

            clearance = radius + ROBOT_RADIUS

            if distance < clearance:

                penetration = clearance - distance

                penalty += 1000.0 + penetration ** 2

    return penalty


# ==================================================
# 7. PATH SMOOTHNESS
# ==================================================

def smoothness_penalty(path):

    smoothness = 0.0

    for i in range(1, len(path) - 1):

        v1 = path[i] - path[i - 1]

        v2 = path[i + 1] - path[i]

        n1 = np.linalg.norm(v1)
        n2 = np.linalg.norm(v2)

        if n1 > 0 and n2 > 0:

            cosine = np.dot(v1, v2) / (n1 * n2)

            cosine = np.clip(cosine, -1.0, 1.0)

            angle = np.arccos(cosine)

            smoothness += angle ** 2

    return smoothness


# ==================================================
# 8. FITNESS FUNCTION
# ==================================================

def fitness(solution):

    path = create_path(solution)

    length = path_length(path)

    collision = collision_penalty(path)

    smoothness = smoothness_penalty(path)

    cost = (
        W_LENGTH * length
        + W_COLLISION * collision
        + W_SMOOTHNESS * smoothness
    )

    return cost


# ==================================================
# 9. LEVY FLIGHT
# ==================================================

def levy_flight(size, beta=1.5):

    sigma = (
        (
            math.gamma(1 + beta)
            * math.sin(math.pi * beta / 2)
        )
        /
        (
            math.gamma((1 + beta) / 2)
            * beta
            * 2 ** ((beta - 1) / 2)
        )
    ) ** (1 / beta)

    u = np.random.normal(0, sigma, size)

    v = np.random.normal(0, 1, size)

    step = u / (np.abs(v) ** (1 / beta) + 1e-12)

    return step


# ==================================================
# 10. INITIALIZE CUCKOO NESTS
# ==================================================

def initialize_nests():

    nests = np.random.uniform(
        LOWER,
        UPPER,
        (N_NESTS, DIM)
    )

    return nests


# ==================================================
# 11. CUCKOO SEARCH ALGORITHM
# ==================================================

def cuckoo_search():

    nests = initialize_nests()

    fitness_values = np.array([
        fitness(nest) for nest in nests
    ])

    best_index = np.argmin(fitness_values)

    best_nest = nests[best_index].copy()

    best_fitness = fitness_values[best_index]

    print("\nCUCKOO SEARCH STARTED\n")

    for iteration in range(MAX_ITER):

        # Generate new solutions using Levy flight

        for i in range(N_NESTS):

            step = levy_flight(DIM, BETA)

            new_solution = (
                nests[i]
                + ALPHA * step * (nests[i] - best_nest)
            )

            new_solution += np.random.normal(
                0, 0.1, DIM
            )

            new_solution = np.clip(
                new_solution,
                LOWER,
                UPPER
            )

            new_fitness = fitness(new_solution)

            j = np.random.randint(N_NESTS)

            if new_fitness < fitness_values[j]:

                nests[j] = new_solution.copy()

                fitness_values[j] = new_fitness

        # Abandon nests with probability PA

        abandoned = np.random.rand(N_NESTS) < PA

        for i in range(N_NESTS):

            if abandoned[i]:

                nests[i] = np.random.uniform(
                    LOWER,
                    UPPER,
                    DIM
                )

                fitness_values[i] = fitness(nests[i])

        # Update best solution

        best_index = np.argmin(fitness_values)

        if fitness_values[best_index] < best_fitness:

            best_fitness = fitness_values[best_index]

            best_nest = nests[best_index].copy()

        # Display iteration results

        if (iteration + 1) % 50 == 0:

            print(
                "Iteration:",
                iteration + 1,
                "| Best Fitness:",
                round(best_fitness, 4)
            )

    print("\nCUCKOO SEARCH COMPLETED\n")

    return best_nest, best_fitness


# ==================================================
# 12. EXECUTE CUCKOO SEARCH
# ==================================================

best_solution, best_cost = cuckoo_search()

optimal_path = create_path(best_solution)

final_length = path_length(optimal_path)

final_collision = collision_penalty(optimal_path)

final_smoothness = smoothness_penalty(optimal_path)


# ==================================================
# 13. DISPLAY OPTIMIZATION RESULTS
# ==================================================

print("========== OPTIMIZATION RESULTS ==========")

print("\nOptimal Robot Path:\n")

for i, point in enumerate(optimal_path):

    print(
        "Point",
        i + 1,
        ":",
        np.round(point, 2)
    )

print("\nBest Fitness:", round(best_cost, 4))

print("Path Length:", round(final_length, 4))

print("Collision Penalty:", round(final_collision, 4))

print("Smoothness:", round(final_smoothness, 4))

if final_collision == 0:

    print("\nPath Status: COLLISION FREE")

else:

    print("\nPath Status: COLLISION DETECTED")
