"""
Aplicación Streamlit - Employee Engagement - Markov Chain Analysis
Menú interactivo de ejemplos predefinidos y análisis personalizado
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from markov_chain_clean import MarkovChainAnalyzer
import io

# Configuración de la página
st.set_page_config(
    page_title="Markov Chain - Employee Engagement",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)


def initialize_session_state():
    """Inicializa el estado de la sesión"""
    if 'selected_example' not in st.session_state:
        st.session_state.selected_example = None
    if 'analyzer' not in st.session_state:
        st.session_state.analyzer = None
    if 'filters' not in st.session_state:
        st.session_state.filters = {}
    if 'results' not in st.session_state:
        st.session_state.results = None


def show_menu():
    """Muestra el menú principal de ejemplos"""
    st.title("📊 Análisis de Engagement - Cadena de Markov")
    st.markdown("### Selecciona un tipo de análisis:")
    st.markdown("---")
    
    # Crear botones para cada ejemplo
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("1️⃣ Todos los Empleados (sin filtros)", use_container_width=True, type="primary"):
            st.session_state.selected_example = 1
            st.rerun()
        
        if st.button("2️⃣ Solo México", use_container_width=True, type="primary"):
            st.session_state.selected_example = 2
            st.rerun()
        
        if st.button("3️⃣ Engineering en México (Senior)", use_container_width=True, type="primary"):
            st.session_state.selected_example = 3
            st.rerun()
        
        if st.button("4️⃣ Solo Líderes", use_container_width=True, type="primary"):
            st.session_state.selected_example = 4
            st.rerun()
    
    with col2:
        if st.button("5️⃣ Comparación Líderes vs Empleados", use_container_width=True, type="primary"):
            st.session_state.selected_example = 5
            st.rerun()
        
        if st.button("6️⃣ Ver Opciones Disponibles", use_container_width=True, type="primary"):
            st.session_state.selected_example = 6
            st.rerun()
        
        if st.button("🎨 Filtros Personalizados", use_container_width=True, type="secondary"):
            st.session_state.selected_example = 'custom'
            st.rerun()
        
        if st.button("🏠 Volver al Menú Principal", use_container_width=True):
            st.session_state.selected_example = None
            st.session_state.results = None
            st.rerun()


def run_example_1():
    """Ejemplo 1: Todos los empleados sin filtros"""
    st.title("📊 Ejemplo 1: Todos los Empleados")
    st.info("Analizando todos los datos sin filtros")
    
    with st.spinner("Cargando datos y ejecutando análisis..."):
        analyzer = MarkovChainAnalyzer(database_type="NA")
        analyzer.load_data()
        results = analyzer.run_complete_analysis(filters=None, visualize=False)
        
        st.session_state.analyzer = analyzer
        st.session_state.results = results
    
    st.success("✅ Análisis completado")


def run_example_2():
    """Ejemplo 2: Solo México"""
    st.title("📊 Ejemplo 2: Solo México")
    st.info("Filtrando por ubicación: MX/CDMX/CDMX y MX/JALISCO/GDL")
    
    with st.spinner("Cargando datos y ejecutando análisis..."):
        analyzer = MarkovChainAnalyzer(database_type="NA")
        analyzer.load_data()
        
        filters = {
            'Location': ['MX/CDMX/CDMX', 'MX/JALISCO/GDL']
        }
        
        results = analyzer.run_complete_analysis(filters=filters, visualize=False)
        
        st.session_state.analyzer = analyzer
        st.session_state.results = results
        st.session_state.filters = filters
    
    st.success("✅ Análisis completado")


def run_example_3():
    """Ejemplo 3: Engineering en México, Senior"""
    st.title("📊 Ejemplo 3: Engineering en México (Senior)")
    st.info("Filtros: México + Studio Engineering + Seniority Senior")
    
    with st.spinner("Cargando datos y ejecutando análisis..."):
        analyzer = MarkovChainAnalyzer(database_type="NA")
        analyzer.load_data()
        
        filters = {
            'Location': ['MX/CDMX/CDMX', 'MX/JALISCO/GDL'],
            'Studio': ['Engineering'],
            'Seniority': ['Sr Level 1', 'Sr Level 2', 'Sr Level 3']
        }
        
        results = analyzer.run_complete_analysis(filters=filters, visualize=False)
        
        st.session_state.analyzer = analyzer
        st.session_state.results = results
        st.session_state.filters = filters
    
    st.success("✅ Análisis completado")


def run_example_4():
    """Ejemplo 4: Solo líderes"""
    st.title("📊 Ejemplo 4: Solo Líderes")
    st.info("Analizando únicamente empleados que son líderes")
    
    with st.spinner("Cargando datos y ejecutando análisis..."):
        analyzer = MarkovChainAnalyzer(database_type="L")
        analyzer.load_data()
        results = analyzer.run_complete_analysis(filters=None, visualize=False)
        
        st.session_state.analyzer = analyzer
        st.session_state.results = results
    
    st.success("✅ Análisis completado")


def run_example_5():
    """Ejemplo 5: Comparación líderes vs empleados"""
    st.title("📊 Ejemplo 5: Comparación Líderes vs No Líderes")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("👔 Líderes")
        with st.spinner("Analizando líderes..."):
            analyzer_leaders = MarkovChainAnalyzer(database_type="L")
            analyzer_leaders.load_data()
            results_leaders = analyzer_leaders.run_complete_analysis(filters=None, visualize=False)
        
        st.metric("Total Registros", len(analyzer_leaders.original_data))
        st.metric("Estados", len(results_leaders['states']))
        st.metric("¿Ergódica?", "✅" if results_leaders['ergodic'] else "❌")
        
        if results_leaders['ergodic']:
            st.write("**Distribución Estacionaria:**")
            df_leaders = pd.DataFrame({
                'Estado': results_leaders['states'],
                'Probabilidad': results_leaders['stationary_distribution']
            })
            st.dataframe(df_leaders.style.format({'Probabilidad': '{:.4f}'}), use_container_width=True)
    
    with col2:
        st.subheader("👥 No Líderes")
        with st.spinner("Analizando empleados..."):
            analyzer_employees = MarkovChainAnalyzer(database_type="E")
            analyzer_employees.load_data()
            results_employees = analyzer_employees.run_complete_analysis(filters=None, visualize=False)
        
        st.metric("Total Registros", len(analyzer_employees.original_data))
        st.metric("Estados", len(results_employees['states']))
        st.metric("¿Ergódica?", "✅" if results_employees['ergodic'] else "❌")
        
        if results_employees['ergodic']:
            st.write("**Distribución Estacionaria:**")
            df_employees = pd.DataFrame({
                'Estado': results_employees['states'],
                'Probabilidad': results_employees['stationary_distribution']
            })
            st.dataframe(df_employees.style.format({'Probabilidad': '{:.4f}'}), use_container_width=True)
    
    # Guardar resultados de líderes para visualización
    st.session_state.analyzer = analyzer_leaders
    st.session_state.results = results_leaders
    
    st.success("✅ Comparación completada")


def run_example_6():
    """Ejemplo 6: Ver opciones disponibles"""
    st.title("📊 Ejemplo 6: Opciones Disponibles para Filtrado")
    
    with st.spinner("Cargando datos..."):
        analyzer = MarkovChainAnalyzer(database_type="NA")
        analyzer.load_data()
    
    st.success(f"✅ {len(analyzer.data)} registros cargados")
    
    columns_to_check = ['Location', 'Team Name', 'Seniority', 'Studio', 'Position']
    
    tabs = st.tabs(columns_to_check)
    
    for i, column in enumerate(columns_to_check):
        with tabs[i]:
            options = analyzer.get_available_options(column)
            st.write(f"**Total de opciones:** {len(options)}")
            
            data_list = []
            for opt in options:
                count = len(analyzer.data[analyzer.data[column] == opt])
                data_list.append({'Opción': opt, 'Registros': count})
            
            df = pd.DataFrame(data_list)
            st.dataframe(df, use_container_width=True, height=400)
    
    st.session_state.analyzer = analyzer


def filter_section():
    """Sección de filtros personalizados (Opción Custom)"""
    st.title("🎨 Filtros Personalizados")
    st.info("Crea tu propio análisis seleccionando los filtros que desees")
    
    # Selector de tipo de datos
    col1, col2 = st.columns([1, 3])
    with col1:
        db_type = st.selectbox(
            "Tipo de empleado",
            options=["NA", "L", "E"],
            format_func=lambda x: {
                "NA": "Todos",
                "L": "Líderes",
                "E": "Empleados"
            }[x],
            key="custom_db_type"
        )
    
    # Cargar datos si no están cargados o si cambió el tipo
    if (st.session_state.analyzer is None or 
        st.session_state.analyzer.database_type != db_type or
        st.session_state.analyzer.original_data is None):
        with st.spinner("Cargando datos..."):
            analyzer = MarkovChainAnalyzer(database_type=db_type)
            analyzer.load_data()
            st.session_state.analyzer = analyzer
            st.session_state.filters = {}
    
    analyzer = st.session_state.analyzer
    
    st.markdown("---")
    st.subheader("🔍 Selecciona Filtros (Opcional)")
    
    # Usar expanders para los filtros
    with st.expander("📍 Ubicación", expanded=False):
        locations = analyzer.get_available_options('Location')
        if locations:
            selected_locations = st.multiselect(
                "Selecciona ubicaciones (vacío = todas)",
                options=locations,
                default=st.session_state.filters.get('Location', []),
                key="location_filter"
            )
            if selected_locations:
                st.session_state.filters['Location'] = selected_locations
            elif 'Location' in st.session_state.filters:
                del st.session_state.filters['Location']
    
    with st.expander("👥 Equipo"):
        temp_analyzer = MarkovChainAnalyzer(database_type=analyzer.database_type)
        temp_analyzer.data = analyzer.original_data.copy()
        if 'Location' in st.session_state.filters:
            temp_analyzer.filter_by_column('Location', st.session_state.filters['Location'])
        
        teams = temp_analyzer.get_available_options('Team Name')
        if teams:
            selected_teams = st.multiselect(
                "Selecciona equipos (vacío = todos)",
                options=teams,
                default=st.session_state.filters.get('Team Name', []),
                key="team_filter"
            )
            if selected_teams:
                st.session_state.filters['Team Name'] = selected_teams
            elif 'Team Name' in st.session_state.filters:
                del st.session_state.filters['Team Name']
    
    with st.expander("📊 Seniority"):
        temp_analyzer = MarkovChainAnalyzer(database_type=analyzer.database_type)
        temp_analyzer.data = analyzer.original_data.copy()
        for key, values in st.session_state.filters.items():
            if key in ['Location', 'Team Name']:
                temp_analyzer.filter_by_column(key, values)
        
        seniorities = temp_analyzer.get_available_options('Seniority')
        if seniorities:
            selected_seniorities = st.multiselect(
                "Selecciona seniority (vacío = todos)",
                options=seniorities,
                default=st.session_state.filters.get('Seniority', []),
                key="seniority_filter"
            )
            if selected_seniorities:
                st.session_state.filters['Seniority'] = selected_seniorities
            elif 'Seniority' in st.session_state.filters:
                del st.session_state.filters['Seniority']
    
    with st.expander("🏢 Studio"):
        temp_analyzer = MarkovChainAnalyzer(database_type=analyzer.database_type)
        temp_analyzer.data = analyzer.original_data.copy()
        for key, values in st.session_state.filters.items():
            if key in ['Location', 'Team Name', 'Seniority']:
                temp_analyzer.filter_by_column(key, values)
        
        studios = temp_analyzer.get_available_options('Studio')
        if studios:
            selected_studios = st.multiselect(
                "Selecciona studios (vacío = todos)",
                options=studios,
                default=st.session_state.filters.get('Studio', []),
                key="studio_filter"
            )
            if selected_studios:
                st.session_state.filters['Studio'] = selected_studios
            elif 'Studio' in st.session_state.filters:
                del st.session_state.filters['Studio']
    
    with st.expander("💼 Posición"):
        temp_analyzer = MarkovChainAnalyzer(database_type=analyzer.database_type)
        temp_analyzer.data = analyzer.original_data.copy()
        for key, values in st.session_state.filters.items():
            temp_analyzer.filter_by_column(key, values)
        
        positions = temp_analyzer.get_available_options('Position')
        if positions:
            selected_positions = st.multiselect(
                "Selecciona posiciones (vacío = todas)",
                options=positions,
                default=st.session_state.filters.get('Position', []),
                key="position_filter"
            )
            if selected_positions:
                st.session_state.filters['Position'] = selected_positions
            elif 'Position' in st.session_state.filters:
                del st.session_state.filters['Position']
    
    # Resumen de filtros activos
    st.markdown("---")
    if st.session_state.filters:
        st.success("### ✅ Filtros Activos:")
        for key, values in st.session_state.filters.items():
            st.write(f"**{key}:** {', '.join(map(str, values))}")
    else:
        st.info("ℹ️ Sin filtros - se usarán todos los datos")
    
    # Botones de acción
    st.markdown("---")
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🚀 Ejecutar Análisis", type="primary", use_container_width=True):
            return True
    
    with col2:
        if st.button("🔄 Limpiar Filtros", use_container_width=True):
            st.session_state.filters = {}
            st.rerun()
    
    return False


def run_analysis():
    """Ejecuta el análisis de Markov para filtros personalizados"""
    analyzer = st.session_state.analyzer
    
    # Crear nueva instancia para el análisis
    analysis_analyzer = MarkovChainAnalyzer(database_type=analyzer.database_type)
    
    # Usar SIEMPRE los datos originales como base
    base_data = analyzer.original_data.copy()
    analysis_analyzer.data = base_data
    analysis_analyzer.original_data = base_data   # <-- ESTA LÍNEA ES LA CLAVE
    
    with st.spinner("Ejecutando análisis de cadena de Markov..."):
        try:
            # Aplicar filtros
            if st.session_state.filters:
                analysis_analyzer.apply_filters(st.session_state.filters)
            
            # Ejecutar análisis completo
            results = analysis_analyzer.run_complete_analysis(
                filters=None,  # Ya aplicamos los filtros
                visualize=False
            )
            
            st.session_state.results = results
            st.session_state.analyzer = analysis_analyzer
            st.success("✅ Análisis completado exitosamente")
            
        except Exception as e:
            st.error(f"❌ Error durante el análisis: {str(e)}")
            st.session_state.results = None


def display_results():
    """Muestra los resultados del análisis"""
    if st.session_state.results is None:
        return
    
    # Verificar que tenemos analyzer
    if st.session_state.analyzer is None:
        st.warning("⚠️ No hay analyzer disponible para mostrar resultados.")
        return
    
    results = st.session_state.results
    analyzer = st.session_state.analyzer
    
    st.header("📈 Resultados del Análisis")
    
    # Métricas principales
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Estados", len(results['states']))
    with col2:
        st.metric("Irreducible", "✅" if results['irreducible'] else "❌")
    with col3:
        st.metric("Aperiódica", "✅" if results['aperiodic'] else "❌")
    with col4:
        st.metric("Ergódica", "✅" if results['ergodic'] else "❌")
    
    st.markdown("---")
    
    # Matriz de transición
    st.subheader("🔢 Matriz de Transición")
    
    # Crear figura de matplotlib
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(results['transition_matrix'], cmap='Blues', interpolation='nearest')
    
    n_states = len(results['states'])
    ax.set_xticks(np.arange(n_states))
    ax.set_yticks(np.arange(n_states))
    ax.set_xticklabels(results['states'])
    ax.set_yticklabels(results['states'])
    ax.set_xlabel('Next State')
    ax.set_ylabel('Current State')
    ax.set_title('Markov Chain Transition Matrix')
    
    # Añadir valores numéricos
    for i in range(n_states):
        for j in range(n_states):
            text = ax.text(j, i, f"{results['transition_matrix'][i, j]:.2f}",
                          ha='center', va='center', color='black')
    
    plt.colorbar(im, ax=ax, label='Transition Probability')
    plt.tight_layout()
    
    st.pyplot(fig)
    
    # Si es ergódica, mostrar distribución estacionaria
    if results['ergodic']:
        st.markdown("---")
        st.subheader("📊 Distribución Estacionaria")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Probabilidades de Largo Plazo")
            df_stationary = pd.DataFrame({
                'Estado': results['states'],
                'Probabilidad': results['stationary_distribution'],
                'Porcentaje': results['stationary_distribution'] * 100
            })
            st.dataframe(
                df_stationary.style.format({
                    'Probabilidad': '{:.4f}',
                    'Porcentaje': '{:.2f}%'
                }),
                use_container_width=True
            )
        
        with col2:
            st.markdown("#### Tiempos Medios de Recurrencia")
            df_times = pd.DataFrame({
                'Estado': results['states'],
                'Tiempo Medio (pasos)': results['mean_recurrence_times']
            })
            st.dataframe(
                df_times.style.format({
                    'Tiempo Medio (pasos)': '{:.2f}'
                }),
                use_container_width=True
            )
        
        # Gráfica de distribución estacionaria
        fig2, ax2 = plt.subplots(figsize=(10, 5))
        ax2.bar(results['states'], results['stationary_distribution'], color='steelblue')
        ax2.set_xlabel('Estado de Engagement')
        ax2.set_ylabel('Probabilidad')
        ax2.set_title('Distribución Estacionaria de Estados')
        ax2.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        
        st.pyplot(fig2)
    
    else:
        st.warning("⚠️ La cadena no es ergódica. No se puede calcular distribución estacionaria.")
        if not results['irreducible']:
            st.info("ℹ️ La cadena NO es irreducible: existen estados que no son alcanzables desde otros estados.")
        if not results['aperiodic']:
            st.info("ℹ️ La cadena NO es aperiódica: existen ciclos periódicos en las transiciones.")
    
    # Simulación
    st.markdown("---")
    st.subheader("🎲 Simulación de Trayectoria")
    
    # Verificar que el analyzer tiene state_to_idx
    if not hasattr(analyzer, 'state_to_idx') or analyzer.state_to_idx is None:
        st.warning("⚠️ No se puede simular: el análisis no se ha completado correctamente.")
        return
    
    # Obtener solo los estados que realmente están en state_to_idx
    available_states = sorted([float(k) for k in analyzer.state_to_idx.keys()])
    
    if len(available_states) == 0:
        st.warning("⚠️ No hay estados disponibles para simular.")
        return
    
    col1, col2 = st.columns([1, 3])
    
    with col1:
        initial_state = st.selectbox(
            "Estado inicial",
            options=available_states,
            index=len(available_states)//2,
            format_func=lambda x: f"{x:.1f}"
        )
        
        n_steps = st.slider(
            "Número de pasos",
            min_value=10,
            max_value=200,
            value=60,
            step=10
        )
        
        if st.button("🎯 Simular"):
            try:
                # Simular usando el estado seleccionado
                simulated = analyzer.simulate(float(initial_state), n_steps=n_steps)
                
                fig3, ax3 = plt.subplots(figsize=(12, 6))
                ax3.plot(simulated, marker='o', linewidth=2, markersize=4)
                ax3.set_xlabel('Paso')
                ax3.set_ylabel('Nivel de Engagement')
                ax3.set_title(f'Simulación de Engagement (Estado inicial: {initial_state})')
                ax3.grid(True, alpha=0.3)
                plt.tight_layout()
                
                with col2:
                    st.pyplot(fig3)
                    
                    # Estadísticas de la simulación
                    st.markdown("**Estadísticas de la simulación:**")
                    unique_visited = sorted(set(simulated))
                    st.write(f"- Estados visitados: {unique_visited}")
                    st.write(f"- Estado final: {simulated[-1]}")
                    st.write(f"- Tiempo en cada estado:")
                    for state in unique_visited:
                        count = simulated.count(state)
                        percentage = (count / len(simulated)) * 100
                        st.write(f"  - Estado {state}: {count} pasos ({percentage:.1f}%)")
                        
            except Exception as e:
                with col2:
                    st.error(f"❌ Error durante la simulación: {str(e)}")
                    st.info(f"Estados disponibles: {available_states}")


def export_results():
    """Sección para exportar resultados"""
    st.sidebar.markdown("---")
    st.sidebar.header("💾 Exportar")
    
    results = st.session_state.results
    
    # Preparar datos para exportar
    export_data = {
        'transition_matrix': pd.DataFrame(
            results['transition_matrix'],
            index=results['states'],
            columns=results['states']
        )
    }
    
    if results['ergodic']:
        export_data['stationary_distribution'] = pd.DataFrame({
            'Estado': results['states'],
            'Probabilidad': results['stationary_distribution'],
            'Tiempo_Medio_Recurrencia': results['mean_recurrence_times']
        })
    
    # Crear Excel
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        export_data['transition_matrix'].to_excel(writer, sheet_name='Matriz')
        if 'stationary_distribution' in export_data:
            export_data['stationary_distribution'].to_excel(writer, sheet_name='Distribucion', index=False)
        
        # Agregar información de filtros
        filters_df = pd.DataFrame([
            {'Filtro': k, 'Valores': ', '.join(map(str, v))}
            for k, v in st.session_state.filters.items()
        ])
        if not filters_df.empty:
            filters_df.to_excel(writer, sheet_name='Filtros', index=False)
    
    st.sidebar.download_button(
        label="📥 Descargar Excel",
        data=output.getvalue(),
        file_name="markov_results.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True
    )


def main():
    """Función principal de la app"""
    # Inicializar sesión
    initialize_session_state()
    
    # Sidebar con información
    st.sidebar.title("📊 Markov Chain Analyzer")
    st.sidebar.markdown("---")
    
    # Mostrar información según el estado
    if st.session_state.selected_example is not None:
        st.sidebar.info(f"**Análisis actual:** Ejemplo {st.session_state.selected_example}")
        
        if st.session_state.results is not None:
            st.sidebar.markdown("### 📈 Resultados")
            st.sidebar.metric("Estados", len(st.session_state.results['states']))
            st.sidebar.metric("Ergódica", "✅" if st.session_state.results['ergodic'] else "❌")
    
    # Si no hay ejemplo seleccionado, mostrar menú
    if st.session_state.selected_example is None:
        show_menu()
        
        # Información en sidebar
        st.sidebar.markdown("### 📖 Acerca de")
        st.sidebar.info(
            "Selecciona un ejemplo predefinido para análisis rápido, "
            "o usa **Filtros Personalizados** para crear tu propio análisis."
        )
        
    # Ejecutar el ejemplo seleccionado
    elif st.session_state.selected_example == 1:
        run_example_1()
        display_results()
        if st.session_state.results:
            export_results()
        
    elif st.session_state.selected_example == 2:
        run_example_2()
        display_results()
        if st.session_state.results:
            export_results()
        
    elif st.session_state.selected_example == 3:
        run_example_3()
        display_results()
        if st.session_state.results:
            export_results()
        
    elif st.session_state.selected_example == 4:
        run_example_4()
        display_results()
        if st.session_state.results:
            export_results()
        
    elif st.session_state.selected_example == 5:
        run_example_5()
        # Este ejemplo muestra su propia visualización
        
    elif st.session_state.selected_example == 6:
        run_example_6()
        
    elif st.session_state.selected_example == 'custom':
        should_analyze = filter_section()
        if should_analyze:
            run_analysis()
        display_results()
        if st.session_state.results:
            export_results()
    
    # Botón para volver al menú (siempre visible en sidebar)
    if st.session_state.selected_example is not None:
        st.sidebar.markdown("---")
        if st.sidebar.button("🏠 Volver al Menú", use_container_width=True, type="primary"):
            st.session_state.selected_example = None
            st.session_state.results = None
            st.session_state.filters = {}
            st.session_state.analyzer = None
            st.rerun()
    
    # Footer
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Versión:** 1.0  \n**Motor:** Cadenas de Markov")


if __name__ == "__main__":
    main()