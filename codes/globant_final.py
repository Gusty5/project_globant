"""
Aplicación Streamlit - Employee Engagement - Markov Chain Analysis
Herramienta para Globant: análisis de engagement con cadenas de Markov
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from markov_chain_clean import MarkovChainAnalyzer
import io

# ------------------- PALETA Y ESTILO GLOBAL GLOBANT -------------------
PRIMARY_GREEN = "#C0D733"   # Verde Globant
DARK_GREY     = "#333333"   # Texto principal
LIGHT_GREY    = "#F2F2F2"   # Fondo sidebar / bloques
BACKGROUND    = "#FFFFFF"   # Fondo principal

# Configuración de la página
st.set_page_config(
    page_title="Globant | Employee Engagement Analyzer",
    page_icon="G",
    layout="wide",
    initial_sidebar_state="expanded"
)

# === CSS GLOBAL: ESTILO TIPO RECOMENDADOR DE GRASAS, PERO MÁS LEGIBLE ===
st.markdown(
    f"""
    <style>
    :root {{
        --primary-color: {PRIMARY_GREEN};
        --text-color: {DARK_GREY};
        --bg-color: {BACKGROUND};
        --sidebar-bg: {LIGHT_GREY};
    }}

    /* ======== BASICS ======== */
    .stApp {{
        background-color: var(--bg-color);
        color: var(--text-color);
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
        font-size: 16px !important;
    }}

    /* HEADER */
    header[data-testid="stHeader"] {{
        background-color: var(--bg-color) !important;
        color: var(--text-color) !important;
        border-bottom: 1px solid #E0E0E0 !important;
        box-shadow: none !important;
    }}
    header[data-testid="stHeader"] * {{
        color: var(--text-color) !important;
    }}

    /* SIDEBAR */
    section[data-testid="stSidebar"] {{
        background-color: var(--sidebar-bg) !important;
        border-right: 1px solid #E0E0E0 !important;
    }}
    section[data-testid="stSidebar"] * {{
        color: {DARK_GREY} !important;
        font-size: 15px !important;
    }}

    /* ======== BUTTONS ======== */
    .stButton button, button[kind="primary"], button[kind="secondary"] {{
        font-size: 15px !important;
        font-weight: 600 !important;
        border-radius: 999px !important;
        padding-top: 0.45rem !important;
        padding-bottom: 0.45rem !important;
    }}

    button[kind="primary"] {{
        background-color: var(--primary-color) !important;
        color: #FFFFFF !important;
        border: none !important;
    }}
    button[kind="primary"]:hover {{
        filter: brightness(0.93) !important;
    }}

    button[kind="secondary"] {{
        background-color: #FFFFFF !important;
        border: 2px solid var(--primary-color) !important;
        color: var(--primary-color) !important;
    }}

    /* ======== INPUTS y SELECTORES ======== */
    .stTextInput > div > div > input,
    .stNumberInput input,
    .stSelectbox > div > div,
    .stMultiSelect > div > div {{
        background-color: #FFFFFF !important;
        color: {DARK_GREY} !important;
        border-radius: 8px !important;
        font-size: 15px !important;
        border: 1px solid #CCCCCC !important;
    }}

    /* ======== HERO TITLES ======== */
    .hero-title {{
        font-size: 2.7rem !important;
        font-weight: 900 !important;
        line-height: 1.05 !important;
        color: {DARK_GREY} !important;
        margin-bottom: 0.5rem !important;
    }}

    .hero-subtitle {{
        font-size: 1.25rem !important;
        color: #555555 !important;
        margin-bottom: 1.5rem !important;
        max-width: 720px !important;
    }}

    .hero-dot {{
        color: var(--primary-color) !important;
    }}

    /* ======== EXPANDERS ======== */
    div[data-testid="stExpander"] > details > summary {{
        background-color: #F7F7F7 !important;
        color: {DARK_GREY} !important;
        border-radius: 8px !important;
        font-size: 16px !important;
    }}

    /* ======== METRICS ======== */
    div[data-testid="stMetricValue"] {{
        color: {DARK_GREY} !important;
        font-weight: 700 !important;
        font-size: 20px !important;
    }}
    div[data-testid="stMetricLabel"] {{
        color: #666666 !important;
        font-size: 14px !important;
    }}

    .stAlert > div {{
        border-radius: 10px !important;
        font-size: 15px !important;
    }}

    /* ======== DATAFRAME (modo claro real) ======== */
    .stDataFrame table {{
        background-color: #ffffff !important;
    }}
    .stDataFrame tbody tr td,
    .stDataFrame thead tr th {{
        background-color: #ffffff !important;
        color: #000000 !important;
        border-color: #E0E0E0 !important;
    }}

    /* ======== DROPDOWNS (selectbox y multiselect MODO CLARO REAL) ======== */
    .stSelectbox div[data-baseweb="select"],
    .stMultiSelect div[data-baseweb="select"] {{
        background-color: #ffffff !important;
        color: #000000 !important;
        border: 1px solid #CCCCCC !important;
        border-radius: 8px !important;
    }}

    /* texto dentro del input */
    .stSelectbox div[data-baseweb="select"] div,
    .stMultiSelect div[data-baseweb="select"] div {{
        color: #000000 !important;
    }}

    /* lista desplegable */
    ul[role="listbox"] {{
        background-color: #ffffff !important;
        color: #000000 !important;
        border-radius: 8px !important;
        border: 1px solid #CCCCCC !important;
    }}

    ul[role="listbox"] li {{
        background-color: #ffffff !important;
        color: #000000 !important;
        font-size: 15px !important;
        padding: 10px !important;
    }}

    ul[role="listbox"] li:hover {{
        background-color: #F2F2F2 !important;
        color: #000000 !important;
    }}

    /* ======== SELECTED TAGS (chips) ======== */
    .stMultiSelect div[data-baseweb="tag"] {{
        background-color: #C0D733 !important;
        color: #000000 !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        padding: 4px 8px !important;
    }}

    .stMultiSelect div[data-baseweb="tag"] svg {{
        fill: #000000 !important;
    }}

    /* ======== TABLAS ESTÁTICAS (st.table) MODO CLARO Y MÁS COMPACTAS ======== */
    .stTable {{
        max-width: 260px !important;
    }}
    .stTable table {{
        background-color: #ffffff !important;
        border-collapse: collapse !important;
        width: auto !important;
    }}
    .stTable thead tr th,
    .stTable tbody tr td {{
        background-color: #ffffff !important;
        color: #000000 !important;
        border: 1px solid #E0E0E0 !important;
        font-size: 15px !important;
        padding: 6px 10px !important;
        text-align: center !important;
    }}
    .stTable thead tr th {{
        font-weight: 600 !important;
    }}

    </style>
    """,
    unsafe_allow_html=True
)


# ================== LÓGICA Y ESTADO ==================

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
    st.markdown("## Escenarios rápidos de análisis")
    st.markdown(
        "Elige un punto de partida para entender cómo evoluciona el engagement en Globant "
        "por país, studio, seniority y liderazgo."
    )

    col1, col2 = st.columns(2)

    with col1:
        if st.button("1. Todos los empleados (sin filtros)", use_container_width=True, type="primary"):
            st.session_state.selected_example = 1
            st.rerun()

        if st.button("2. Solo México", use_container_width=True, type="primary"):
            st.session_state.selected_example = 2
            st.rerun()

        if st.button("3. Engineering en México (Senior)", use_container_width=True, type="primary"):
            st.session_state.selected_example = 3
            st.rerun()

        if st.button("4. Solo líderes", use_container_width=True, type="primary"):
            st.session_state.selected_example = 4
            st.rerun()

    with col2:
        if st.button("5. Comparación líderes vs empleados", use_container_width=True, type="primary"):
            st.session_state.selected_example = 5
            st.rerun()

        if st.button("6. Ver opciones disponibles para filtros", use_container_width=True, type="primary"):
            st.session_state.selected_example = 6
            st.rerun()

        if st.button("Filtros personalizados", use_container_width=True, type="secondary"):
            st.session_state.selected_example = 'custom'
            st.rerun()


def run_example_1():
    """Ejemplo 1: Todos los empleados sin filtros"""
    st.title("Ejemplo 1: Todos los empleados")
    st.info("Se analizan todos los datos disponibles sin aplicar filtros.")

    with st.spinner("Cargando datos y ejecutando análisis..."):
        analyzer = MarkovChainAnalyzer(database_type="NA")
        analyzer.load_data()
        results = analyzer.run_complete_analysis(filters=None, visualize=False)

        st.session_state.analyzer = analyzer
        st.session_state.results = results

    st.success("Análisis completado.")


def run_example_2():
    """Ejemplo 2: Solo México"""
    st.title("Ejemplo 2: Solo México")
    st.info("Se filtra por ubicación: MX/CDMX/CDMX y MX/JALISCO/GDL.")

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

    st.success("Análisis completado.")


def run_example_3():
    """Ejemplo 3: Engineering en México, Senior"""
    st.title("Ejemplo 3: Engineering en México (Senior)")
    st.info("Filtros aplicados: México + Studio Engineering + Seniority Senior.")

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

    st.success("Análisis completado.")


def run_example_4():
    """Ejemplo 4: Solo líderes"""
    st.title("Ejemplo 4: Solo líderes")
    st.info("Se analizan únicamente empleados que son líderes.")

    with st.spinner("Cargando datos y ejecutando análisis..."):
        analyzer = MarkovChainAnalyzer(database_type="L")
        analyzer.load_data()
        results = analyzer.run_complete_analysis(filters=None, visualize=False)

        st.session_state.analyzer = analyzer
        st.session_state.results = results

    st.success("Análisis completado.")


def run_example_5():
    """Ejemplo 5: Comparación líderes vs empleados"""
    st.title("Ejemplo 5: Comparación líderes vs empleados")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Líderes")
        with st.spinner("Analizando líderes..."):
            analyzer_leaders = MarkovChainAnalyzer(database_type="L")
            analyzer_leaders.load_data()
            results_leaders = analyzer_leaders.run_complete_analysis(filters=None, visualize=False)

        st.metric("Total de registros", len(analyzer_leaders.original_data))
        st.metric("Número de estados", len(results_leaders['states']))
        st.metric("¿Cadena ergódica?", "Sí" if results_leaders['ergodic'] else "No")

        if results_leaders['ergodic']:
            st.write("Distribución estacionaria (probabilidad de largo plazo en cada nivel de engagement).")
            df_leaders = pd.DataFrame({
                'Estado': results_leaders['states'],
                'Probabilidad': results_leaders['stationary_distribution']
            })
            st.table(df_leaders.style.format({'Probabilidad': '{:.4f}'}))

    with col2:
        st.subheader("Empleados (no líderes)")
        with st.spinner("Analizando empleados..."):
            analyzer_employees = MarkovChainAnalyzer(database_type="E")
            analyzer_employees.load_data()
            results_employees = analyzer_employees.run_complete_analysis(filters=None, visualize=False)

        st.metric("Total de registros", len(analyzer_employees.original_data))
        st.metric("Número de estados", len(results_employees['states']))
        st.metric("¿Cadena ergódica?", "Sí" if results_employees['ergodic'] else "No")

        if results_employees['ergodic']:
            st.write("Distribución estacionaria (probabilidad de largo plazo en cada nivel de engagement).")
            df_employees = pd.DataFrame({
                'Estado': results_employees['states'],
                'Probabilidad': results_employees['stationary_distribution']
            })
            st.table(df_employees.style.format({'Probabilidad': '{:.4f}'}))

    # Guardar resultados de líderes para visualización general
    st.session_state.analyzer = analyzer_leaders
    st.session_state.results = results_leaders

    st.success("Comparación completada.")


def run_example_6():
    """Ejemplo 6: Ver opciones disponibles"""
    st.title("Ejemplo 6: Opciones disponibles para filtrado")

    with st.spinner("Cargando datos..."):
        analyzer = MarkovChainAnalyzer(database_type="NA")
        analyzer.load_data()

    st.session_state.analyzer = analyzer

    st.markdown("### Columnas disponibles para filtros personalizados")
    st.write(analyzer.get_available_columns())

    st.info(
        "Estas son las columnas que puedes usar para segmentar el análisis "
        "en la opción de filtros personalizados."
    )


def filter_section():
    """Sección de filtros personalizados (Opción Custom)"""
    st.title("Filtros personalizados")
    st.info("Crea tu propio análisis seleccionando los filtros que desees.")

    # Tipo de empleado
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
        )

    with col2:
        st.write(
            "Selecciona si deseas analizar a todos los empleados, "
            "solo líderes o solo colaboradores sin rol de liderazgo."
        )

    # Cargar datos base
    analyzer = MarkovChainAnalyzer(database_type=db_type)
    analyzer.load_data()
    st.session_state.analyzer = analyzer

    st.markdown("### Definir filtros")

    # Filtros por ubicación
    with st.expander("Ubicación"):
        temp_analyzer = MarkovChainAnalyzer(database_type=analyzer.database_type)
        temp_analyzer.data = analyzer.original_data.copy()

        locations = temp_analyzer.get_available_options('Location')
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

    # Filtro por equipo
    with st.expander("Equipo"):
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

    # Filtro por seniority
    with st.expander("Seniority"):
        temp_analyzer = MarkovChainAnalyzer(database_type=analyzer.database_type)
        temp_analyzer.data = analyzer.original_data.copy()

        for key, values in st.session_state.filters.items():
            if key in ['Location', 'Team Name']:
                temp_analyzer.filter_by_column(key, values)

        seniorities = temp_analyzer.get_available_options('Seniority')
        if seniorities:
            selected_seniorities = st.multiselect(
                "Selecciona seniorities (vacío = todos)",
                options=seniorities,
                default=st.session_state.filters.get('Seniority', []),
                key="seniority_filter"
            )
            if selected_seniorities:
                st.session_state.filters['Seniority'] = selected_seniorities
            elif 'Seniority' in st.session_state.filters:
                del st.session_state.filters['Seniority']

    # Filtro por studio
    with st.expander("Studio"):
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

    # Filtro por posición
    with st.expander("Posición"):
        temp_analyzer = MarkovChainAnalyzer(database_type=analyzer.database_type)
        temp_analyzer.data = analyzer.original_data.copy()
        for key, values in st.session_state.filters.items():
            if key in ['Location', 'Team Name', 'Seniority', 'Studio']:
                temp_analyzer.filter_by_column(key, values)

        positions = temp_analyzer.get_available_options('Position Name')
        if positions:
            selected_positions = st.multiselect(
                "Selecciona posiciones (vacío = todas)",
                options=positions,
                default=st.session_state.filters.get('Position Name', []),
                key="position_filter"
            )
            if selected_positions:
                st.session_state.filters['Position Name'] = selected_positions
            elif 'Position Name' in st.session_state.filters:
                del st.session_state.filters['Position Name']

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Ejecutar análisis", type="primary", use_container_width=True):
            return True
    with col2:
        if st.button("Limpiar filtros", use_container_width=True):
            st.session_state.filters = {}
            st.rerun()

    return False


def run_analysis():
    """Ejecuta el análisis de Markov para filtros personalizados"""
    analyzer = st.session_state.analyzer

    # Instancia nueva para análisis
    analysis_analyzer = MarkovChainAnalyzer(database_type=analyzer.database_type)

    base_data = analyzer.original_data.copy()
    analysis_analyzer.data = base_data
    analysis_analyzer.original_data = base_data

    with st.spinner("Ejecutando análisis de cadena de Markov..."):
        try:
            if st.session_state.filters:
                for column, values in st.session_state.filters.items():
                    analysis_analyzer.filter_by_column(column, values)

            results = analysis_analyzer.run_complete_analysis(
                filters=None,
                visualize=False
            )

            st.session_state.results = results
            st.session_state.analyzer = analysis_analyzer
            st.success("Análisis completado correctamente.")

        except Exception as e:
            st.error(f"Ocurrió un error durante el análisis: {str(e)}")
            st.session_state.results = None


def display_results():
    """Muestra los resultados del análisis, con explicaciones al lado de cada visualización"""
    if st.session_state.results is None:
        return

    if st.session_state.analyzer is None:
        st.warning("No hay un análisis disponible para mostrar resultados.")
        return

    results = st.session_state.results
    analyzer = st.session_state.analyzer

    st.header("Resultados del análisis")

    # Métricas principales
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Número de estados", len(results['states']))
    with col2:
        st.metric("Cadena ergódica", "Sí" if results['ergodic'] else "No")
    with col3:
        st.metric("Registros analizados", len(analyzer.data))

    st.subheader("Matriz de transición")

    # Visualización + explicación lado a lado
    col_vis, col_exp = st.columns([2, 1])

    with col_vis:
        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(results['transition_matrix'], cmap='Greens', interpolation='nearest')

        n_states = len(results['states'])
        ax.set_xticks(np.arange(n_states))
        ax.set_yticks(np.arange(n_states))
        ax.set_xticklabels(results['states'])
        ax.set_yticklabels(results['states'])
        ax.set_xlabel('Estado en el siguiente periodo')
        ax.set_ylabel('Estado en el periodo actual')
        ax.set_title('Matriz de transición de engagement')

        for i in range(n_states):
            for j in range(n_states):
                ax.text(
                    j, i,
                    f"{results['transition_matrix'][i, j]:.2f}",
                    ha='center',
                    va='center',
                    color='black'
                )

        plt.colorbar(im, ax=ax, label='Probabilidad de transición')
        plt.tight_layout()
        st.pyplot(fig)

    with col_exp:
        st.markdown(
            """
            **Cómo leer esta matriz**

            - Cada **fila** representa un nivel de engagement **actual**.
            - Cada **columna** representa el nivel de engagement en el **siguiente periodo** (siguiente medición).
            - El color y el número indican la **probabilidad** de moverse de un nivel a otro.

            Ejemplo de lectura:
            - Si en la fila "3" y columna "4" aparece 0.20, significa que:
              > Cuando una persona está en el nivel 3 hoy, tiene 20% de probabilidad de estar en el nivel 4 en la siguiente medición.

            Uso para negocio:
            - Nos permite ver si los equipos **tienden a mejorar, empeorar o mantenerse** en su nivel de engagement.
            """
        )

    # Distribución estacionaria y tiempos de recurrencia
    if results['ergodic']:
        st.subheader("Distribución estacionaria (largo plazo)")

        col_table, col_explain = st.columns([2, 1])

        with col_table:
            df_stationary = pd.DataFrame({
                'Estado': results['states'],
                'Probabilidad': results['stationary_distribution']
            })
            st.table(df_stationary.style.format({'Probabilidad': '{:.4f}'}))

            # Gráfica de barras
            fig2, ax2 = plt.subplots(figsize=(10, 5))
            ax2.bar(results['states'], results['stationary_distribution'], color=PRIMARY_GREEN)
            ax2.set_xlabel('Nivel de engagement')
            ax2.set_ylabel('Probabilidad en el largo plazo')
            ax2.set_title('Distribución estacionaria de niveles de engagement')
            ax2.grid(axis='y', alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig2)

        with col_explain:
            st.markdown(
                """
                **Qué significa la distribución estacionaria**

                - Muestra la **proporción de personas** que, en el largo plazo, esperamos que estén en cada nivel de engagement.
                - No es un snapshot de hoy, sino una visión de **equilibrio** si las probabilidades de transición se mantienen.

                Ejemplo:
                - Si el estado 4 tiene probabilidad 0.40, significa que, a largo plazo, se espera que **40%** de las personas
                  estén en nivel de engagement 4.

                Uso para negocio:
                - Ayuda a responder: *“Si no cambiamos nada, ¿a qué distribución de engagement convergerá la organización?”*.
                - Permite comparar segmentos (por país, studio, seniority) y ver **quiénes tienden a niveles más altos o más bajos**.
                """
            )

        st.subheader("Tiempos medios de recurrencia")

        col_times, col_times_exp = st.columns([2, 1])

        with col_times:
            df_times = pd.DataFrame({
                'Estado': results['states'],
                'Tiempo medio (pasos)': results['mean_recurrence_times']
            })
            st.table(df_times.style.format({'Tiempo medio (pasos)': '{:.2f}'}))

        with col_times_exp:
            st.markdown(
                """
                **Qué es el tiempo medio de recurrencia**

                - Para cada nivel de engagement, indica **cada cuántos periodos, en promedio, se vuelve a visitar ese nivel**.
                - Un número **bajo** significa que el nivel se visita **con frecuencia**.
                - Un número **alto** significa que el nivel es más **raro** o que se tarda más en regresar.

                Uso para negocio:
                - Permite ver qué niveles de engagement son más **estables** o frecuentes.
                - Por ejemplo, si el nivel 2 tiene tiempo medio 2.5, en promedio se regresa a ese nivel cada 2.5 mediciones.
                """
            )
    else:
        st.warning(
            "La cadena no es ergódica. No se puede calcular una distribución estacionaria bien definida "
            "ni tiempos de recurrencia consistentes para todos los estados."
        )

    # Simulación de trayectoria
    st.subheader("Simulación de trayectoria de engagement")

    if not hasattr(analyzer, 'state_to_idx') or analyzer.state_to_idx is None:
        st.warning("No se puede simular porque el análisis no tiene el mapeo de estados configurado.")
        return

    available_states = sorted([float(k) for k in analyzer.state_to_idx.keys()])

    if len(available_states) == 0:
        st.warning("No hay estados disponibles para simular.")
        return

    col_controls, col_plot = st.columns([1, 2])

    with col_controls:
        initial_state = st.selectbox(
            "Estado inicial de engagement",
            options=available_states,
            index=len(available_states) // 2,
            format_func=lambda x: f"{x:.1f}"
        )

        n_steps = st.slider(
            "Número de periodos a simular",
            min_value=10,
            max_value=200,
            value=60,
            step=10
        )

        run_simulation = st.button("Simular trayectoria", type="primary", use_container_width=True)

    if run_simulation:
        try:
            simulated = analyzer.simulate(float(initial_state), n_steps=n_steps)

            col_plot_vis, col_plot_exp = st.columns([2, 1])

            with col_plot_vis:
                fig3, ax3 = plt.subplots(figsize=(12, 6))
                ax3.plot(simulated, marker='o', linewidth=2, markersize=4)
                ax3.set_xlabel('Periodo')
                ax3.set_ylabel('Nivel de engagement')
                ax3.set_title(f'Trayectoria simulada de engagement (estado inicial: {initial_state})')
                ax3.grid(True, alpha=0.3)
                plt.tight_layout()
                st.pyplot(fig3)

            with col_plot_exp:
                st.markdown(
                    """
                    **Cómo interpretar la trayectoria simulada**

                    - Cada punto representa el **nivel de engagement de una persona o grupo** en cada periodo.
                    - La simulación usa la matriz de transición estimada para ir moviéndose de un nivel a otro.
                    - No es un pronóstico exacto, sino un **escenario probable** si las probabilidades se mantienen.

                    Uso para negocio:
                    - Permite ilustrar cómo puede evolucionar el engagement de un equipo a lo largo del tiempo.
                    - Útil para explicarle a líderes de negocio el concepto de “camino” de engagement.
                    """
                )

                st.markdown("**Resumen de la simulación:**")
                unique_visited = sorted(set(simulated))
                st.write(f"- Estados visitados: {unique_visited}")
                st.write(f"- Estado final: {simulated[-1]}")
                st.write(f"- Tiempo en cada estado:")
                for state in unique_visited:
                    count = simulated.count(state)
                    percentage = (count / len(simulated)) * 100
                    st.write(f"  - Estado {state}: {count} periodos ({percentage:.1f}%)")

        except Exception as e:
            st.error(f"Ocurrió un error durante la simulación: {str(e)}")
            st.info(f"Estados disponibles: {available_states}")


def export_results():
    """Sección para exportar resultados"""
    st.sidebar.header("Exportar resultados")

    results = st.session_state.results

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
            'Probabilidad': results['stationary_distribution']
        })

    export_data['mean_recurrence_times'] = pd.DataFrame({
        'Estado': results['states'],
        'Tiempo medio (pasos)': results['mean_recurrence_times']
    })

    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        for sheet_name, df in export_data.items():
            df.to_excel(writer, sheet_name=sheet_name, index=True)

        if st.session_state.filters:
            filters_df = pd.DataFrame([
                {'Columna': k, 'Valores': ", ".join(map(str, v))}
                for k, v in st.session_state.filters.items()
            ])
            if not filters_df.empty:
                filters_df.to_excel(writer, sheet_name='Filtros', index=False)

    st.sidebar.download_button(
        label="Descargar Excel con resultados",
        data=output.getvalue(),
        file_name="markov_results_globant.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True
    )


def main():
    """Función principal de la app de engagement para Globant"""
    initialize_session_state()

    # ---------- SIDEBAR ----------
    with st.sidebar:
        st.markdown("### Globant | Engagement Analyzer")
        st.markdown(
            "Convierte los datos de encuestas de clima en información clara y accionable "
            "sobre cómo evoluciona el engagement en tus equipos."
        )
        
        if st.session_state.selected_example is not None:
            etiqueta = (
                "Filtros personalizados"
                if st.session_state.selected_example == "custom"
                else f"Ejemplo {st.session_state.selected_example}"
            )
            st.info(f"Análisis actual: {etiqueta}")

            if st.session_state.results is not None:
                st.markdown("#### Resumen rápido")
                st.metric("Estados", len(st.session_state.results['states']))
                st.metric(
                    "Cadena ergódica",
                    "Sí" if st.session_state.results['ergodic'] else "No"
                )
        else:
            st.markdown("#### Acerca de esta herramienta")
            st.info(
                "Puedes comenzar con un escenario predefinido o construir un análisis a la medida "
                "según país, studio, seniority y tipo de rol."
            )

    # ---------- HERO ----------
    col_hero_text, col_hero_side = st.columns([2, 1])
    with col_hero_text:
        st.markdown(
            """
            <div class="hero-title">
            Globant | Analizador de engagement<span class="hero-dot">.</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div class="hero-subtitle">'
            'Analiza cómo evoluciona el nivel de engagement de las personas en el tiempo, '
            'segmentando por país, studio, seniority y liderazgo.'
            '</div>',
            unsafe_allow_html=True,
        )

    with col_hero_side:
        st.markdown("#### Qué permite esta herramienta")
        st.markdown(
            "- Analizar patrones de cambio de engagement por segmento.\n"
            "- Comparar líderes vs no líderes.\n"
            "- Diseñar análisis personalizados con filtros avanzados.\n"
            "- Exportar resultados para equipos de HR y negocio."
        )

    # ---------- LÓGICA PRINCIPAL ----------
    if st.session_state.selected_example is None:
        show_menu()
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
    elif st.session_state.selected_example == 6:
        run_example_6()
    elif st.session_state.selected_example == "custom":
        should_analyze = filter_section()
        if should_analyze:
            run_analysis()
        display_results()
        if st.session_state.results:
            export_results()

    # Botón de volver al menú en sidebar
    if st.session_state.selected_example is not None:
        if st.sidebar.button("Volver al menú principal", use_container_width=True, type="primary"):
            st.session_state.selected_example = None
            st.session_state.results = None
            st.session_state.filters = {}
            st.session_state.analyzer = None
            st.rerun()

    st.sidebar.markdown("Versión: 1.0  \nMotor: Cadenas de Markov")


if __name__ == "__main__":
    main()
