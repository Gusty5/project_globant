"""
Análisis de Cadena de Markov para Employee Engagement
Version limpia y modular antes de implementación en Streamlit
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import networkx as nx
from functools import reduce
from math import gcd


class MarkovChainAnalyzer:
    """
    Clase para análisis de cadenas de Markov en datos de engagement de empleados
    """
    
    def __init__(self, database_type="NA"):
        """
        Inicializa el analizador con el tipo de base de datos
        
        Args:
            database_type: "L" (leaders), "E" (employees), "NA" (all)
        """
        self.database_type = database_type
        self.data = None
        self.original_data = None
        self.transition_matrix = None
        self.unique_states = None
        self.state_to_idx = None
        
    def load_data(self, file_path=None):
        """
        Carga los datos según el tipo de base de datos
        
        Args:
            file_path: Ruta personalizada al archivo CSV (opcional)
        """
        if file_path:
            self.data = pd.read_csv(file_path)
        else:
            if self.database_type == "L":
                self.data = pd.read_csv('../data/data_leaders.csv')
            elif self.database_type == "E":
                self.data = pd.read_csv('../data/data_no_leaders.csv')
            else:  # NA o default
                self.data = pd.read_csv('../data/data_global_clean.csv')
        
        self.original_data = self.data.copy()
        print(f"✅ Datos cargados: {len(self.data)} registros")
        return self.data
    
    def get_available_options(self, column):
        """
        Obtiene las opciones disponibles para una columna específica
        
        Args:
            column: Nombre de la columna
            
        Returns:
            Lista ordenada de valores únicos
        """
        if column not in self.data.columns:
            return []
        return sorted(self.data[column].dropna().unique())
    
    def filter_by_column(self, column, selected_values):
        """
        Filtra los datos por valores seleccionados en una columna
        
        Args:
            column: Nombre de la columna
            selected_values: Lista de valores a mantener (None para todos)
            
        Returns:
            DataFrame filtrado
        """
        if selected_values is None or len(selected_values) == 0:
            return self.data
        
        if column not in self.data.columns:
            print(f"⚠️  La columna '{column}' no existe en los datos")
            return self.data
        
        self.data = self.data[self.data[column].isin(selected_values)]
        print(f"✔️  Filtrado por {column}: {len(self.data)} registros restantes")
        return self.data
    
    def apply_filters(self, filters):
        """
        Aplica múltiples filtros de manera secuencial
        
        Args:
            filters: Diccionario con {columna: [valores]}
            
        Example:
            filters = {
                'Location': ['MX/CDMX/CDMX', 'MX/JALISCO/GDL'],
                'Team Name': ['Breaking Badger'],
                'Seniority': ['Sr Level 1', 'Sr Level 2']
            }
        """
        for column, values in filters.items():
            if values:  # Solo filtrar si hay valores seleccionados
                self.filter_by_column(column, values)
        
        print(f"\n📊 Registros finales después de filtros: {len(self.data)}")
        return self.data
    
    def preprocess_engagement(self):
        """
        Preprocesa los datos de engagement:
        - Selecciona columnas relevantes
        - Redondea valores de engagement
        - Ordena por ID, Month, Day
        """
        # Seleccionar columnas relevantes
        self.data = self.data[['Month', 'Day', 'Engagement', 'ID']]
        
        # Redondear engagement al entero más cercano
        self.data['Engagement'] = self.data['Engagement'].apply(math.ceil)
        
        # Ordenar por ID, Month, Day
        self.data = self.data.sort_values(["ID", "Month", "Day"]).reset_index(drop=True)
        
        print(f"✅ Datos preprocesados")
        print(f"   Estados de engagement únicos: {sorted(self.data['Engagement'].unique())}")
        print(f"   Distribución:")
        print(self.data['Engagement'].value_counts().sort_index())
        
        return self.data
    
    def build_transition_matrix(self):
        """
        Construye la matriz de transición de la cadena de Markov
        
        Returns:
            Matriz de transición normalizada
        """
        states = self.data["Engagement"].values
        self.unique_states = np.sort(self.data["Engagement"].unique())
        n_states = len(self.unique_states)
        
        # Mapeo de estados a índices (convertir a float para consistencia)
        self.state_to_idx = {float(state): i for i, state in enumerate(self.unique_states)}
        
        # Matriz de conteo de transiciones
        transition_counts = np.zeros((n_states, n_states), dtype=int)
        
        # Contar transiciones
        for i in range(len(states) - 1):
            # Saltar transiciones entre diferentes IDs
            if self.data["ID"].values[i] != self.data["ID"].values[i + 1]:
                continue
            
            s_current = self.state_to_idx[float(states[i])]
            s_next = self.state_to_idx[float(states[i + 1])]
            transition_counts[s_current, s_next] += 1
        
        # Normalizar (convertir a probabilidades)
        # Evitar división por cero
        row_sums = transition_counts.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1  # Para evitar NaN
        
        self.transition_matrix = transition_counts / row_sums
        
        print(f"\n✅ Matriz de transición construida ({n_states}x{n_states})")
        
        return self.transition_matrix
    
    def is_irreducible(self):
        """
        Verifica si la cadena de Markov es irreducible
        
        Returns:
            True si es irreducible, False en caso contrario
        """
        n = self.transition_matrix.shape[0]
        G = nx.DiGraph()
        G.add_nodes_from(range(n))
        
        for i in range(n):
            for j in range(n):
                if self.transition_matrix[i, j] > 0:
                    G.add_edge(i, j)
        
        return nx.is_strongly_connected(G)
    
    def period_of_state(self, state, max_power=200):
        """
        Calcula el período de un estado específico
        
        Args:
            state: Índice del estado
            max_power: Máxima potencia a calcular
            
        Returns:
            Período del estado (None si no se encuentra)
        """
        n = self.transition_matrix.shape[0]
        A = np.eye(n)
        hits = []
        
        for power in range(1, max_power + 1):
            A = A.dot(self.transition_matrix)
            if A[state, state] > 0:
                hits.append(power)
            
            # Optimización: si ya tenemos varios hits y gcd==1, terminamos
            if len(hits) >= 2:
                current_gcd = reduce(gcd, hits)
                if current_gcd == 1:
                    return 1
        
        if not hits:
            return None
        
        return reduce(gcd, hits)
    
    def is_aperiodic(self, max_power=200):
        """
        Verifica si la cadena de Markov es aperiódica
        
        Returns:
            True si es aperiódica, False en caso contrario
        """
        # Si existe P_ii > 0 en algún i => aperiódica
        if np.any(np.diag(self.transition_matrix) > 0):
            return True
        
        # Calcular período del primer estado
        per = self.period_of_state(0, max_power=max_power)
        return per == 1
    
    def is_ergodic(self):
        """
        Verifica si la cadena es ergódica (irreducible y aperiódica)
        
        Returns:
            Tupla (irreducible, aperiodic)
        """
        irreducible = self.is_irreducible()
        aperiodic = self.is_aperiodic()
        
        print(f"\n🔍 Propiedades de la cadena:")
        print(f"   Irreducible: {irreducible}")
        print(f"   Aperiódica: {aperiodic}")
        print(f"   Ergódica: {irreducible and aperiodic}")
        
        return irreducible, aperiodic
    
    def compute_stationary_distribution(self):
        """
        Calcula la distribución estacionaria (si existe)
        
        Returns:
            Vector de probabilidades estacionarias
        """
        # Calcular eigenvalores y eigenvectores de P^T
        vals, vectors = np.linalg.eig(self.transition_matrix.T)
        
        # Encontrar eigenvector con eigenvalor = 1
        stationary = vectors[:, np.isclose(vals, 1)]
        
        if stationary.shape[1] == 0:
            print("⚠️  No se encontró distribución estacionaria")
            return None
        
        # Normalizar para que sume 1
        pi = stationary[:, 0].real
        pi = pi / pi.sum()
        
        print(f"\n✅ Distribución estacionaria:")
        for state, prob in zip(self.unique_states, pi):
            print(f"   Estado {state}: {prob:.4f}")
        
        return pi
    
    def compute_mean_recurrence_times(self, pi):
        """
        Calcula los tiempos medios de recurrencia
        
        Args:
            pi: Distribución estacionaria
            
        Returns:
            Array con tiempos medios de recurrencia
        """
        mean_times = 1 / pi
        
        print(f"\n⏱️  Tiempos medios de recurrencia:")
        for state, time in zip(self.unique_states, mean_times):
            print(f"   Estado {state}: {time:.2f} pasos")
        
        return mean_times
    
    def simulate(self, initial_state, n_steps=60):
        """
        Simula la cadena de Markov
        
        Args:
            initial_state: Estado inicial
            n_steps: Número de pasos a simular
            
        Returns:
            Lista de estados simulados
        """
        # Convertir estado inicial a float para consistencia
        current_state = float(initial_state)
        
        # Verificar que el estado existe
        if current_state not in self.state_to_idx:
            available_states = list(self.state_to_idx.keys())
            raise ValueError(
                f"Estado inicial {current_state} no existe en los datos. "
                f"Estados válidos: {available_states}"
            )
        
        simulated_states = [current_state]
        
        for _ in range(n_steps):
            current_idx = self.state_to_idx[current_state]
            next_state = np.random.choice(
                self.unique_states, 
                p=self.transition_matrix[current_idx]
            )
            # Convertir a float para consistencia
            next_state = float(next_state)
            simulated_states.append(next_state)
            current_state = next_state
        
        return simulated_states
    
    def plot_transition_matrix(self, figsize=(10, 8)):
        """
        Visualiza la matriz de transición
        """
        plt.figure(figsize=figsize)
        plt.imshow(self.transition_matrix, cmap='Blues', interpolation='nearest')
        plt.colorbar(label='Transition Probability')
        
        n_states = len(self.unique_states)
        plt.xticks(ticks=np.arange(n_states), labels=self.unique_states, rotation=45)
        plt.yticks(ticks=np.arange(n_states), labels=self.unique_states)
        plt.xlabel('Next State')
        plt.ylabel('Current State')
        plt.title('Markov Chain Transition Matrix')
        
        # Añadir valores numéricos
        for i in range(n_states):
            for j in range(n_states):
                plt.text(j, i, f"{self.transition_matrix[i, j]:.2f}", 
                        ha='center', va='center', color='black')
        
        plt.tight_layout()
        plt.show()
    
    def plot_simulation(self, simulated_states, figsize=(12, 6)):
        """
        Visualiza una simulación de la cadena
        
        Args:
            simulated_states: Lista de estados simulados
        """
        plt.figure(figsize=figsize)
        plt.plot(simulated_states, marker='o')
        plt.xlabel('Step')
        plt.ylabel('Engagement Level')
        plt.title('Simulated Engagement Levels using Markov Chain')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    def run_complete_analysis(self, filters=None, visualize=True):
        """
        Ejecuta el análisis completo de la cadena de Markov
        
        Args:
            filters: Diccionario con filtros a aplicar
            visualize: Si True, genera gráficas
            
        Returns:
            Diccionario con todos los resultados
        """
        results = {}
        
        # 1. Aplicar filtros
        if filters:
            self.apply_filters(filters)
        
        # 2. Preprocesar engagement
        self.preprocess_engagement()
        
        # 3. Construir matriz de transición
        self.build_transition_matrix()
        results['transition_matrix'] = self.transition_matrix
        results['states'] = self.unique_states
        
        # 4. Verificar si es ergódica
        irreducible, aperiodic = self.is_ergodic()
        results['irreducible'] = irreducible
        results['aperiodic'] = aperiodic
        results['ergodic'] = irreducible and aperiodic
        
        # 5. Si es ergódica, calcular distribución estacionaria
        if results['ergodic']:
            pi = self.compute_stationary_distribution()
            results['stationary_distribution'] = pi
            
            if pi is not None:
                mean_times = self.compute_mean_recurrence_times(pi)
                results['mean_recurrence_times'] = mean_times
        
        # 6. Visualización
        if visualize:
            print("\n" + "="*60)
            print("VISUALIZACIÓN")
            print("="*60)
            
            self.plot_transition_matrix()
            
            # Simular desde estado 3.0 (o el estado medio)
            initial_state = self.unique_states[len(self.unique_states)//2]
            simulated = self.simulate(initial_state, n_steps=60)
            self.plot_simulation(simulated)
        
        return results


def main():
    """
    Función principal para ejecutar un ejemplo completo
    """
    print("="*60)
    print("ANÁLISIS DE CADENA DE MARKOV - EMPLOYEE ENGAGEMENT")
    print("="*60)
    
    # Crear analizador
    analyzer = MarkovChainAnalyzer(database_type="NA")
    
    # Cargar datos
    analyzer.load_data()
    
    # Definir filtros (ejemplo)
    filters = {
        'Location': ['MX/CDMX/CDMX', 'MX/JALISCO/GDL'],  # Solo México
        # 'Team Name': ['Breaking Badger'],  # Comentado para usar todos
        # 'Seniority': ['Sr Level 1', 'Sr Level 2'],  # Comentado para usar todos
        # 'Studio': ['Engineering'],  # Comentado para usar todos
        # 'Position': ['Software Developer']  # Comentado para usar todos
    }
    
    # Ejecutar análisis completo
    results = analyzer.run_complete_analysis(filters=filters, visualize=True)
    
    # Imprimir resumen final
    print("\n" + "="*60)
    print("RESUMEN DE RESULTADOS")
    print("="*60)
    print(f"Estados: {results['states']}")
    print(f"Ergódica: {results['ergodic']}")
    
    if results['ergodic']:
        print(f"\nDistribución estacionaria encontrada ✅")
        print(f"Tiempos medios de recurrencia calculados ✅")
    else:
        print(f"\nLa cadena NO es ergódica ⚠️")
        if not results['irreducible']:
            print("  - No es irreducible")
        if not results['aperiodic']:
            print("  - No es aperiódica")


if __name__ == "__main__":
    main()