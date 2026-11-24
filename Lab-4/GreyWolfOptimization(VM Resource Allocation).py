import numpy as np

# ==================================================
# 1. CLOUD COMPUTING ENVIRONMENT
# ==================================================

np.random.seed(42)

# Number of Virtual Machines and Physical Machines
N_VMS = 10
N_PMS = 4

# Physical Machine CPU capacity (CPU units)
PM_CPU = np.array([100, 100, 100, 100])

# Physical Machine RAM capacity (GB)
PM_RAM = np.array([256, 256, 256, 256])

# Virtual Machine CPU requirements
VM_CPU = np.array([
    20, 30, 15, 40, 25,
    35, 20, 15, 30, 25
])

# Virtual Machine RAM requirements (GB)
VM_RAM = np.array([
    32, 64, 32, 96, 48,
    64, 32, 32, 64, 48
])


# ==================================================
# 2. GWO PARAMETERS
# ==================================================

N_WOLVES = 30
MAX_ITER = 200

# Objective function weights
W_RESOURCE = 0.4
W_LOAD = 0.3
W_ENERGY = 0.3

# Overloading penalty
OVERLOAD_PENALTY = 1000

DIM = N_VMS

LOWER_BOUND = 0
UPPER_BOUND = N_PMS - 1


# ==================================================
# 3. DECODE WOLF POSITION
# ==================================================

def decode_position(position):

    allocation = np.rint(position).astype(int)

    allocation = np.clip(
        allocation,
        LOWER_BOUND,
        UPPER_BOUND
    )

    return allocation


# ==================================================
# 4. CALCULATE RESOURCE UTILIZATION
# ==================================================

def calculate_resources(allocation):

    cpu_used = np.zeros(N_PMS)

    ram_used = np.zeros(N_PMS)

    for vm in range(N_VMS):

        pm = allocation[vm]

        cpu_used[pm] += VM_CPU[vm]

        ram_used[pm] += VM_RAM[vm]

    return cpu_used, ram_used


# ==================================================
# 5. RESOURCE WASTAGE
# ==================================================

def resource_wastage(cpu_used, ram_used):

    active = np.where(
        (cpu_used > 0) | (ram_used > 0)
    )[0]

    if len(active) == 0:
        return 1.0

    cpu_waste = (
        (PM_CPU[active] - cpu_used[active])
        / PM_CPU[active]
    )

    ram_waste = (
        (PM_RAM[active] - ram_used[active])
        / PM_RAM[active]
    )

    wastage = np.mean(
        (cpu_waste + ram_waste) / 2
    )

    return wastage


# ==================================================
# 6. LOAD IMBALANCE
# ==================================================

def load_imbalance(cpu_used, ram_used):

    active = np.where(
        (cpu_used > 0) | (ram_used > 0)
    )[0]

    if len(active) <= 1:
        return 0.0

    cpu_util = (
        cpu_used[active] / PM_CPU[active]
    )

    ram_util = (
        ram_used[active] / PM_RAM[active]
    )

    combined_util = (
        cpu_util + ram_util
    ) / 2

    imbalance = np.std(combined_util)

    return imbalance


# ==================================================
# 7. ENERGY CONSUMPTION
# ==================================================

def energy_consumption(cpu_used, ram_used):

    energy = 0.0

    for pm in range(N_PMS):

        if cpu_used[pm] > 0 or ram_used[pm] > 0:

            cpu_util = cpu_used[pm] / PM_CPU[pm]

            # Energy model:
            # Idle power = 100 W
            # Maximum additional power = 100 W

            pm_energy = 100 + 100 * cpu_util

            energy += pm_energy

    # Normalize energy
    max_energy = N_PMS * 200

    normalized_energy = energy / max_energy

    return normalized_energy


# ==================================================
# 8. FITNESS FUNCTION
# ==================================================

def fitness(position):

    allocation = decode_position(position)

    cpu_used, ram_used = calculate_resources(
        allocation
    )

    # Calculate capacity violations
    cpu_overload = np.maximum(
        cpu_used - PM_CPU, 0
    )

    ram_overload = np.maximum(
        ram_used - PM_RAM, 0
    )

    # Overloading penalty
    penalty = OVERLOAD_PENALTY * (
        np.sum(cpu_overload / PM_CPU)
        + np.sum(ram_overload / PM_RAM)
    )

    # Calculate objective values
    resource = resource_wastage(
        cpu_used, ram_used
    )

    load = load_imbalance(
        cpu_used, ram_used
    )

    energy = energy_consumption(
        cpu_used, ram_used
    )

    # Total fitness
    total_fitness = (
        W_RESOURCE * resource
        + W_LOAD * load
        + W_ENERGY * energy
        + penalty
    )

    return total_fitness


# ==================================================
# 9. INITIALIZE GREY WOLVES
# ==================================================

def initialize_wolves():

    wolves = np.random.uniform(
        LOWER_BOUND,
        UPPER_BOUND,
        (N_WOLVES, DIM)
    )

    return wolves


# ==================================================
# 10. GREY WOLF OPTIMIZATION ALGORITHM
# ==================================================

def grey_wolf_optimization():

    wolves = initialize_wolves()

    fitness_values = np.array([
        fitness(wolf) for wolf in wolves
    ])

    # Initialize Alpha, Beta and Delta wolves

    sorted_indices = np.argsort(fitness_values)

    alpha = wolves[sorted_indices[0]].copy()

    beta = wolves[sorted_indices[1]].copy()

    delta = wolves[sorted_indices[2]].copy()

    alpha_fitness = fitness_values[sorted_indices[0]]

    best_position = alpha.copy()

    best_fitness = alpha_fitness

    convergence = []

    print("\nGREY WOLF OPTIMIZATION STARTED\n")

    # Main GWO iterations

    for iteration in range(MAX_ITER):

        # Linearly decreasing parameter
        a = 2 - 2 * (iteration / MAX_ITER)

        for i in range(N_WOLVES):

            # --------------------------------------
            # Alpha wolf position update
            # --------------------------------------

            r1 = np.random.rand(DIM)
            r2 = np.random.rand(DIM)

            A1 = 2 * a * r1 - a
            C1 = 2 * r2

            D_alpha = np.abs(
                C1 * alpha - wolves[i]
            )

            X1 = alpha - A1 * D_alpha

            # --------------------------------------
            # Beta wolf position update
            # --------------------------------------

            r1 = np.random.rand(DIM)
            r2 = np.random.rand(DIM)

            A2 = 2 * a * r1 - a
            C2 = 2 * r2

            D_beta = np.abs(
                C2 * beta - wolves[i]
            )

            X2 = beta - A2 * D_beta

            # --------------------------------------
            # Delta wolf position update
            # --------------------------------------

            r1 = np.random.rand(DIM)
            r2 = np.random.rand(DIM)

            A3 = 2 * a * r1 - a
            C3 = 2 * r2

            D_delta = np.abs(
                C3 * delta - wolves[i]
            )

            X3 = delta - A3 * D_delta

            # --------------------------------------
            # Update wolf position
            # --------------------------------------

            new_position = (X1 + X2 + X3) / 3

            new_position = np.clip(
                new_position,
                LOWER_BOUND,
                UPPER_BOUND
            )

            wolves[i] = new_position

        # ------------------------------------------
        # Evaluate all wolves
        # ------------------------------------------

        fitness_values = np.array([
            fitness(wolf) for wolf in wolves
        ])

        sorted_indices = np.argsort(fitness_values)

        alpha = wolves[sorted_indices[0]].copy()

        beta = wolves[sorted_indices[1]].copy()

        delta = wolves[sorted_indices[2]].copy()

        alpha_fitness = fitness_values[sorted_indices[0]]

        # Maintain global best solution
        if alpha_fitness < best_fitness:

            best_fitness = alpha_fitness

            best_position = alpha.copy()

        convergence.append(best_fitness)

        # Display iteration results
        if (iteration + 1) % 50 == 0:

            print(
                "Iteration:",
                iteration + 1,
                "| Best Fitness:",
                round(best_fitness, 4)
            )

    print("\nGREY WOLF OPTIMIZATION COMPLETED\n")

    return best_position, best_fitness, convergence


# ==================================================
# 11. EXECUTE GWO
# ==================================================

best_position, best_fitness, convergence = (
    grey_wolf_optimization()
)

# Get final VM allocation
best_allocation = decode_position(best_position)

cpu_used, ram_used = calculate_resources(
    best_allocation
)


# ==================================================
# 12. DISPLAY VM RESOURCE ALLOCATION
# ==================================================

print("========== OPTIMIZATION RESULTS ==========")

print("\nOPTIMAL VM RESOURCE ALLOCATION:\n")

for vm in range(N_VMS):

    pm = best_allocation[vm]

    print(
        f"VM {vm + 1} --> PM {pm + 1}"
    )


# ==================================================
# 13. PHYSICAL MACHINE RESOURCE DETAILS
# ==================================================

print("\n========== PHYSICAL MACHINE DETAILS ==========")

for pm in range(N_PMS):

    cpu_util = (
        cpu_used[pm] / PM_CPU[pm]
    ) * 100

    ram_util = (
        ram_used[pm] / PM_RAM[pm]
    ) * 100

    print(f"\nPhysical Machine {pm + 1}")

    print(
        f"CPU Allocation: {cpu_used[pm]} / {PM_CPU[pm]}"
    )

    print(
        f"RAM Allocation: {ram_used[pm]} / {PM_RAM[pm]} GB"
    )

    print(
        f"CPU Utilization: {cpu_util:.2f}%"
    )

    print(
        f"RAM Utilization: {ram_util:.2f}%"
    )


# ==================================================
# 14. FINAL PERFORMANCE METRICS
# ==================================================

final_resource = resource_wastage(
    cpu_used, ram_used
)

final_load = load_imbalance(
    cpu_used, ram_used
)

final_energy = energy_consumption(
    cpu_used, ram_used
)

active_pms = np.sum(
    (cpu_used > 0) | (ram_used > 0)
)

cpu_overload = np.sum(
    np.maximum(cpu_used - PM_CPU, 0)
)

ram_overload = np.sum(
    np.maximum(ram_used - PM_RAM, 0)
)

print("\n========== PERFORMANCE METRICS ==========")

print(
    "\nBest Fitness:",
    round(best_fitness, 4)
)

print(
    "Resource Wastage:",
    round(final_resource * 100, 2), "%"
)

print(
    "Load Imbalance:",
    round(final_load, 4)
)

print(
    "Normalized Energy Consumption:",
    round(final_energy, 4)
)

print(
    "Active Physical Machines:",
    active_pms
)

if cpu_overload == 0 and ram_overload == 0:

    print("\nAllocation Status: SUCCESSFUL")

else:

    print("\nAllocation Status: CAPACITY VIOLATION")
