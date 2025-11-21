import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#location = input("ingrese la ubicacion del nuevo empleado seleccione \n1. Buenos Aires, Argentina \n2. Santiago de Chile, Chile  \n3. Bogota, Colombia \n4. Medellin, Colombia \n5. Ciudad de Mexico, México \n6. Guadalajara, México")
database = input("Ingrese si el nuevo empleado será líder o empleado (L), (E), o (NA): ").strip().upper()

if database == "L":
    data = pd.read_csv('./data/data_leaders.csv')
elif database == "E":
    data = pd.read_csv('./data/data_no_leaders.csv')
elif database == "NA":
    data = pd.read_csv('./data/data_global_clean.csv')
else:
    print("Entrada inválida. Se usará data_global_clean.csv por defecto.")
    data = pd.read_csv('./data/data_global_clean.csv')
    
"""def pedir_location(df):
    locations = sorted(df["Location"].dropna().unique())
    
    print("Ingrese la ubicación del nuevo integrante. Seleccione entre:\n")
    for i, loc in enumerate(locations, start=1):
        print(f"{i}. {loc}")
    
    while True:
        try:
            opcion = int(input("\nEscriba el número de la ubicación: "))
            if 1 <= opcion <= len(locations):
                return locations[opcion - 1]
            else:
                print("Número fuera de rango, intenta de nuevo.")
        except ValueError:
            print("Ingresa un número válido, por favor.")

# Uso:
loc_nuevo = pedir_location(data)
print("Ubicación seleccionada:", loc_nuevo)
"""



data = data[[ 'Month', 'Day', 'Engagement', "ID" ]]

i = 0
# Múltiplos de n/10
n = 100

for value in data['Engagement']:
    if value < .1 * n and value > 0:
        data['Engagement'][i] = .1 * n
    else:
        data['Engagement'][i] = round(value / n, 1) * n
    i += 1



data['Engagement'].value_counts().shape


data['Engagement'].value_counts().sort_index()



data = data.sort_values(["ID", "Month", "Day"]).reset_index(drop=True)



data.head()
data.to_csv("../data/data_test.csv", index=False)



states = data["Engagement"].values


unique_states = np.sort(data["Engagement"].unique())
n_states = data['Engagement'].value_counts().shape[0]


# Mapeo de estados
state_to_idx = {state: i for i, state in enumerate(unique_states)}



# Initialize transition count matrix
transition_counts = np.zeros((n_states, n_states), dtype=int)


transition_counts



for i in range(len(states) - 1):
    if data["ID"].values[i] != data["ID"].values[i + 1]:
        continue  # Skip transitions between different IDs
    s_current = state_to_idx[states[i]]
    s_next = state_to_idx[states[i + 1]]
    transition_counts[s_current, s_next] += 1



transition_counts = transition_counts / transition_counts.sum(axis=1, keepdims=True)



# Plot the transition matrix
plt.figure(figsize=(10, 8))
plt.imshow(transition_counts, cmap='Blues', interpolation='nearest')
plt.colorbar(label='Transition Probability')
plt.xticks(ticks=np.arange(n_states), labels=unique_states, rotation=45)
plt.yticks(ticks=np.arange(n_states), labels=unique_states)
plt.xlabel('Next State')
plt.ylabel('Current State')
plt.title('Markov Chain Transition Matrix')
# number labels
for i in range(n_states):
    for j in range(n_states):
        plt.text(j, i, f"{transition_counts[i, j]:.2f}", ha='center', va='center', color='black')
plt.show()



# Simulate the markov chain
n_steps = 365
current_state = np.random.choice(unique_states)
simulated_states = [current_state]
for _ in range(n_steps):
    current_idx = state_to_idx[current_state]
    next_state = np.random.choice(
        unique_states, p=transition_counts[current_idx])
    simulated_states.append(next_state)
    current_state = next_state

# Plot simulated states
plt.figure(figsize=(12, 6))
plt.plot(simulated_states, marker='o')
plt.xticks(ticks=np.arange(n_steps + 1))
plt.xlabel('Step')
plt.ylabel('Engagement Level')
plt.title('Simulated Engagement Levels using Markov Chain')
plt.grid()
plt.show()