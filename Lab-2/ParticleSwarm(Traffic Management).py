import random

vehicle_count = [120, 80, 150, 100]

MIN_GREEN = 10
MAX_GREEN = 60

NUM_PARTICLES = 30
NUM_ITERATIONS = 100
W = 0.7       # Inertia weight
C1 = 1.5      # Cognitive coefficient
C2 = 1.5      # Social coefficient


# Fitness function
# Lower value = better traffic signal timing
def fitness(green_times):
    total_waiting_time = 0

    for vehicles, green in zip(vehicle_count, green_times):
        # Approximate waiting time:
        # More green time -> less waiting time
        waiting_time = vehicles / green
        total_waiting_time += waiting_time

    return total_waiting_time


# Initialize particles
particles = []

for _ in range(NUM_PARTICLES):
    position = [
        random.uniform(MIN_GREEN, MAX_GREEN)
        for _ in range(4)
    ]

    velocity = [
        random.uniform(-2, 2)
        for _ in range(4)
    ]

    particles.append({
        "position": position,
        "velocity": velocity,
        "best_position": position.copy(),
        "best_fitness": fitness(position)
    })


# Find initial global best
global_best_particle = min(
    particles,
    key=lambda p: p["best_fitness"]
)

global_best_position = global_best_particle["best_position"].copy()
global_best_fitness = global_best_particle["best_fitness"]


for iteration in range(NUM_ITERATIONS):

    for particle in particles:

        for i in range(4):

            r1 = random.random()
            r2 = random.random()

            # Update velocity
            particle["velocity"][i] = (
                W * particle["velocity"][i]
                + C1 * r1 *
                (particle["best_position"][i]
                 - particle["position"][i])
                + C2 * r2 *
                (global_best_position[i]
                 - particle["position"][i])
            )

            # Update position
            particle["position"][i] += particle["velocity"][i]

            # Keep green time within limits
            particle["position"][i] = max(
                MIN_GREEN,
                min(MAX_GREEN, particle["position"][i])
            )

        # Calculate new fitness
        current_fitness = fitness(particle["position"])

        # Update personal best
        if current_fitness < particle["best_fitness"]:
            particle["best_fitness"] = current_fitness
            particle["best_position"] = particle["position"].copy()

        # Update global best
        if current_fitness < global_best_fitness:
            global_best_fitness = current_fitness
            global_best_position = particle["position"].copy()

print("Optimal Traffic Signal Timing")
print("--------------------------------")

for i, green_time in enumerate(global_best_position):
    print(
        f"Road {i + 1}: "
        f"{green_time:.2f} seconds green time"
    )

print("\nMinimum estimated waiting-time score:",
      round(global_best_fitness, 2))
