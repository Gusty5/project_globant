"""
Ejemplos de uso del MarkovChainAnalyzer
Diferentes escenarios de análisis
"""

from markov_chain_clean import MarkovChainAnalyzer
import pandas as pd


def example_1_no_filters():
    """
    Ejemplo 1: Análisis sin filtros (todos los datos)
    """
    print("\n" + "="*70)
    print("EJEMPLO 1: Análisis sin filtros - Todos los empleados")
    print("="*70)
    
    analyzer = MarkovChainAnalyzer(database_type="NA")
    analyzer.load_data()
    
    # Sin filtros
    results = analyzer.run_complete_analysis(filters=None, visualize=True)
    
    return results


def example_2_location_only():
    """
    Ejemplo 2: Filtrar solo por ubicación
    """
    print("\n" + "="*70)
    print("EJEMPLO 2: Filtrado por ubicación - Solo México")
    print("="*70)
    
    analyzer = MarkovChainAnalyzer(database_type="NA")
    analyzer.load_data()
    
    filters = {
        'Location': ['MX/CDMX/CDMX', 'MX/JALISCO/GDL']
    }
    
    results = analyzer.run_complete_analysis(filters=filters, visualize=True)
    
    return results


def example_3_multiple_filters():
    """
    Ejemplo 3: Múltiples filtros combinados
    """
    print("\n" + "="*70)
    print("EJEMPLO 3: Múltiples filtros - Engineering en México, Senior")
    print("="*70)
    
    analyzer = MarkovChainAnalyzer(database_type="NA")
    analyzer.load_data()
    
    filters = {
        'Location': ['MX/CDMX/CDMX', 'MX/JALISCO/GDL'],
        'Studio': ['Engineering'],
        'Seniority': ['Sr Level 1', 'Sr Level 2', 'Sr Level 3']
    }
    
    results = analyzer.run_complete_analysis(filters=filters, visualize=True)
    
    return results


def example_4_leaders_only():
    """
    Ejemplo 4: Solo líderes
    """
    print("\n" + "="*70)
    print("EJEMPLO 4: Solo líderes")
    print("="*70)
    
    analyzer = MarkovChainAnalyzer(database_type="L")
    analyzer.load_data()
    
    results = analyzer.run_complete_analysis(filters=None, visualize=True)
    
    return results


def example_5_step_by_step():
    """
    Ejemplo 5: Análisis paso a paso (más control)
    """
    print("\n" + "="*70)
    print("EJEMPLO 5: Análisis paso a paso con control manual")
    print("="*70)
    
    analyzer = MarkovChainAnalyzer(database_type="NA")
    analyzer.load_data()
    
    # Paso 1: Ver opciones disponibles antes de filtrar
    print("\n📍 Ubicaciones disponibles:")
    locations = analyzer.get_available_options('Location')
    for i, loc in enumerate(locations, 1):
        print(f"   {i}. {loc}")
    
    # Paso 2: Filtrar por ubicación
    analyzer.filter_by_column('Location', ['MX/CDMX/CDMX'])
    
    # Paso 3: Ver equipos disponibles después del filtro de ubicación
    print("\n👥 Equipos disponibles después de filtrar por ubicación:")
    teams = analyzer.get_available_options('Team Name')
    for i, team in enumerate(teams, 1):
        print(f"   {i}. {team}")
    
    # Paso 4: Filtrar por equipo
    analyzer.filter_by_column('Team Name', ['Breaking Badger', 'Fight Club Penguin'])
    
    # Paso 5: Preprocesar
    analyzer.preprocess_engagement()
    
    # Paso 6: Construir matriz
    analyzer.build_transition_matrix()
    
    # Paso 7: Verificar propiedades
    irreducible, aperiodic = analyzer.is_ergodic()
    
    # Paso 8: Si es ergódica, calcular distribución estacionaria
    if irreducible and aperiodic:
        pi = analyzer.compute_stationary_distribution()
        mean_times = analyzer.compute_mean_recurrence_times(pi)
    
    # Paso 9: Visualizar
    analyzer.plot_transition_matrix()
    
    # Paso 10: Simular
    initial_state = 3.0
    simulated = analyzer.simulate(initial_state, n_steps=60)
    analyzer.plot_simulation(simulated)
    
    return analyzer


def example_6_interactive():
    """
    Ejemplo 6: Exploración interactiva de opciones
    """
    print("\n" + "="*70)
    print("EJEMPLO 6: Exploración de opciones de filtrado")
    print("="*70)
    
    analyzer = MarkovChainAnalyzer(database_type="NA")
    analyzer.load_data()
    
    # Mostrar todas las opciones disponibles para cada filtro
    print("\n📊 Opciones disponibles para filtrado:")
    
    columns_to_check = ['Location', 'Team Name', 'Seniority', 'Studio', 'Position']
    
    for column in columns_to_check:
        options = analyzer.get_available_options(column)
        print(f"\n{column}:")
        for opt in options:
            count = len(analyzer.data[analyzer.data[column] == opt])
            print(f"   - {opt}: {count} registros")
    
    return analyzer


def example_7_compare_groups():
    """
    Ejemplo 7: Comparar diferentes grupos
    """
    print("\n" + "="*70)
    print("EJEMPLO 7: Comparación entre grupos")
    print("="*70)
    
    results = {}
    
    # Grupo 1: Líderes
    print("\n--- ANALIZANDO LÍDERES ---")
    analyzer_leaders = MarkovChainAnalyzer(database_type="L")
    analyzer_leaders.load_data()
    results['leaders'] = analyzer_leaders.run_complete_analysis(visualize=False)
    
    # Grupo 2: No líderes
    print("\n--- ANALIZANDO NO LÍDERES ---")
    analyzer_employees = MarkovChainAnalyzer(database_type="E")
    analyzer_employees.load_data()
    results['employees'] = analyzer_employees.run_complete_analysis(visualize=False)
    
    # Comparación
    print("\n" + "="*70)
    print("COMPARACIÓN DE RESULTADOS")
    print("="*70)
    
    print("\nLÍDERES:")
    print(f"  Estados: {results['leaders']['states']}")
    print(f"  Ergódica: {results['leaders']['ergodic']}")
    if results['leaders']['ergodic']:
        print(f"  Distribución estacionaria: {results['leaders']['stationary_distribution']}")
    
    print("\nNO LÍDERES:")
    print(f"  Estados: {results['employees']['states']}")
    print(f"  Ergódica: {results['employees']['ergodic']}")
    if results['employees']['ergodic']:
        print(f"  Distribución estacionaria: {results['employees']['stationary_distribution']}")
    
    return results


def example_8_custom_simulation():
    """
    Ejemplo 8: Simulaciones personalizadas
    """
    print("\n" + "="*70)
    print("EJEMPLO 8: Simulaciones personalizadas")
    print("="*70)
    
    analyzer = MarkovChainAnalyzer(database_type="NA")
    analyzer.load_data()
    
    filters = {
        'Location': ['MX/CDMX/CDMX']
    }
    
    analyzer.apply_filters(filters)
    analyzer.preprocess_engagement()
    analyzer.build_transition_matrix()
    
    # Simular desde diferentes estados iniciales
    print("\n🎲 Simulando desde diferentes estados iniciales:")
    
    for initial_state in analyzer.unique_states:
        print(f"\n   Estado inicial: {initial_state}")
        simulated = analyzer.simulate(initial_state, n_steps=30)
        
        # Estadísticas de la simulación
        unique_visited = set(simulated)
        print(f"      Estados visitados: {sorted(unique_visited)}")
        print(f"      Estado final: {simulated[-1]}")
    
    return analyzer


def menu():
    """
    Menú interactivo para elegir ejemplos
    """
    print("\n" + "="*70)
    print("MENÚ DE EJEMPLOS - Markov Chain Analyzer")
    print("="*70)
    print("\n1. Análisis sin filtros (todos los datos)")
    print("2. Filtrado por ubicación (solo México)")
    print("3. Múltiples filtros (Engineering en México, Senior)")
    print("4. Solo líderes")
    print("5. Análisis paso a paso con control manual")
    print("6. Exploración de opciones de filtrado")
    print("7. Comparación entre grupos (líderes vs no líderes)")
    print("8. Simulaciones personalizadas")
    print("0. Salir")
    
    choice = input("\nSeleccione un ejemplo (0-8): ").strip()
    
    examples = {
        '1': example_1_no_filters,
        '2': example_2_location_only,
        '3': example_3_multiple_filters,
        '4': example_4_leaders_only,
        '5': example_5_step_by_step,
        '6': example_6_interactive,
        '7': example_7_compare_groups,
        '8': example_8_custom_simulation
    }
    
    if choice == '0':
        print("\n¡Hasta luego!")
        return None
    
    if choice in examples:
        result = examples[choice]()
        return result
    else:
        print("\n❌ Opción no válida")
        return menu()


if __name__ == "__main__":
    # Ejecutar ejemplo específico o mostrar menú
    
    # Opción 1: Ejecutar un ejemplo directamente
    # result = example_3_multiple_filters()
    
    # Opción 2: Mostrar menú interactivo
    menu()