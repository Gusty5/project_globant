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
        try:
            analyzer = MarkovChainAnalyzer(database_type="NA")
            analyzer.load_data()
            
            if len(analyzer.data) < 2:
                st.error("❌ No hay suficientes datos para realizar el análisis.")
                return
            
            results = analyzer.run_complete_analysis(filters=None, visualize=False)

            st.session_state.analyzer = analyzer
            st.session_state.results = results
            st.success("✅ Análisis completado.")
            
        except Exception as e:
            st.error(f"❌ Error en el análisis: {str(e)}")
            st.session_state.results = None
            st.session_state.analyzer = None


def run_example_2():
    """Ejemplo 2: Solo México"""
    st.title("Ejemplo 2: Solo México")
    st.info("Se filtra por ubicación: MX/CDMX/CDMX y MX/JALISCO/GDL.")

    with st.spinner("Cargando datos y ejecutando análisis..."):
        try:
            analyzer = MarkovChainAnalyzer(database_type="NA")
            analyzer.load_data()

            filters = {
                'Location': ['MX/CDMX/CDMX', 'MX/JALISCO/GDL']
            }

            results = analyzer.run_complete_analysis(filters=filters, visualize=False)

            st.session_state.analyzer = analyzer
            st.session_state.results = results
            st.session_state.filters = filters
            st.success("✅ Análisis completado.")
            
        except ValueError as ve:
            st.error(f"❌ Error de validación: {str(ve)}")
            st.warning("Los filtros de este ejemplo pueden haber dejado muy pocos datos.")
            st.session_state.results = None
            st.session_state.analyzer = None
        except Exception as e:
            st.error(f"❌ Error en el análisis: {str(e)}")
            st.session_state.results = None
            st.session_state.analyzer = None


def run_example_3():
    """Ejemplo 3: Engineering en México, Senior"""
    st.title("Ejemplo 3: Engineering en México (Senior)")
    st.info("Filtros aplicados: México + Studio Engineering + Seniority Senior.")

    with st.spinner("Cargando datos y ejecutando análisis..."):
        try:
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
            st.success("✅ Análisis completado.")
            
        except ValueError as ve:
            st.error(f"❌ Error de validación: {str(ve)}")
            st.warning("Los filtros de este ejemplo pueden haber dejado muy pocos datos.")
            st.session_state.results = None
            st.session_state.analyzer = None
        except Exception as e:
            st.error(f"❌ Error en el análisis: {str(e)}")
            st.session_state.results = None
            st.session_state.analyzer = None


def run_example_4():
    """Ejemplo 4: Solo líderes"""
    st.title("Ejemplo 4: Solo líderes")
    st.info("Se analizan únicamente empleados que son líderes.")

    with st.spinner("Cargando datos y ejecutando análisis..."):
        try:
            analyzer = MarkovChainAnalyzer(database_type="L")
            analyzer.load_data()
            
            if len(analyzer.data) < 2:
                st.error("❌ No hay suficientes datos de líderes para realizar el análisis.")
                return
                
            results = analyzer.run_complete_analysis(filters=None, visualize=False)

            st.session_state.analyzer = analyzer
            st.session_state.results = results
            st.success("✅ Análisis completado.")
            
        except Exception as e:
            st.error(f"❌ Error en el análisis: {str(e)}")
            st.session_state.results = None
            st.session_state.analyzer = None


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
    st.session_state.results = None  # No hay resultados porque no se ejecutó análisis

    st.markdown("### Columnas disponibles para filtros personalizados")
    
    # Mostrar las columnas que se pueden filtrar
    columnas_filtro = ['Location', 'Studio', 'Seniority', 'Position', 'Team Name']
    
    for columna in columnas_filtro:
        with st.expander(f"📊 {columna}"):
            opciones = analyzer.get_available_options(columna)
            if opciones:
                st.write(f"**Opciones disponibles ({len(opciones)}):**")
                # Mostrar en columnas para mejor visualización
                n_cols = 3
                cols = st.columns(n_cols)
                for idx, opcion in enumerate(opciones):
                    cols[idx % n_cols].write(f"- {opcion}")
            else:
                st.write("No hay opciones disponibles para esta columna")

    st.info(
        "💡 **Cómo usar esta información:**\n\n"
        "Estas son todas las opciones que puedes usar para segmentar el análisis "
        "en la opción de **Filtros personalizados**. Por ejemplo, puedes analizar "
        "solo un Studio específico, una ubicación, o un nivel de Seniority."
    )
    
    st.success(
        "✅ Para aplicar filtros y ejecutar un análisis, "
        "ve al menú principal y selecciona **'Filtros personalizados'**"
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
    # Solo crear nuevo analyzer si no hay resultados previos
    if 'analyzer' not in st.session_state or st.session_state.results is None:
        analyzer = MarkovChainAnalyzer(database_type=db_type)
        analyzer.load_data()
        st.session_state.analyzer = analyzer
    else:
        # Usar el analyzer existente si ya hay resultados
        analyzer = st.session_state.analyzer

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

            # Validar que haya datos después de filtrar
            if len(analysis_analyzer.data) < 2:
                st.error(
                    f"❌ Los filtros aplicados dejaron solo {len(analysis_analyzer.data)} registro(s). "
                    f"Se necesitan al menos 2 registros para construir una cadena de Markov."
                )
                st.warning("Intenta con filtros menos restrictivos o selecciona más valores.")
                st.session_state.results = None
                st.session_state.analyzer = None
                return

            results = analysis_analyzer.run_complete_analysis(
                filters=None,
                visualize=False
            )

            st.session_state.results = results
            st.session_state.analyzer = analysis_analyzer
            st.success("✅ Análisis completado correctamente.")

        except ValueError as ve:
            st.error(f"❌ Error de validación: {str(ve)}")
            st.warning("Revisa los filtros aplicados e intenta nuevamente.")
            st.session_state.results = None
            st.session_state.analyzer = None
        except Exception as e:
            st.error(f"❌ Ocurrió un error durante el análisis: {str(e)}")
            st.warning("Si el problema persiste, intenta con otros filtros o vuelve al menú principal.")
            st.session_state.results = None
            st.session_state.analyzer = None


def display_results():
    """Muestra los resultados del análisis, con explicaciones al lado de cada visualización"""
    if st.session_state.results is None:
        return

    if st.session_state.analyzer is None:
        st.warning("No hay un análisis disponible para mostrar resultados.")
        return

    results = st.session_state.results
    analyzer = st.session_state.analyzer
    
    # Verificar si hubo error en el análisis
    if 'error' in results:
        st.error(f"❌ Error en el análisis: {results['error']}")
        st.info(
            "**Posibles causas:**\n"
            "- Los filtros seleccionados son demasiado restrictivos\n"
            "- No hay suficientes datos para el análisis\n"
            "- Los datos no tienen transiciones válidas\n\n"
            "**Soluciones:**\n"
            "- Intenta con filtros menos restrictivos\n"
            "- Selecciona un grupo más grande de datos\n"
            "- Prueba con otro de los ejemplos predefinidos"
        )
        return

    st.header("📈 Resultados del análisis")

    # Métricas principales más grandes y visibles
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(
            "Niveles de engagement detectados",
            len(results['states']),
            help="Diferentes niveles de engagement encontrados en tus datos"
        )
    with col2:
        ergodic_label = "✅ Estable" if results['ergodic'] else "⚠️ Inestable"
        ergodic_help = (
            "Sistema estable: converge a un equilibrio predecible" if results['ergodic']
            else "Sistema inestable: el comportamiento depende del punto inicial"
        )
        st.metric(
            "Comportamiento del sistema",
            ergodic_label,
            help=ergodic_help
        )
    with col3:
        st.metric(
            "Empleados analizados",
            len(analyzer.data),
            help="Número de registros incluidos en este análisis"
        )

    st.markdown("---")
    st.subheader("🔄 Matriz de transición: Probabilidades de cambio día a día")
    st.caption("Esta matriz muestra cómo cambia el engagement de un día para otro. Lee las filas como 'hoy' y las columnas como 'mañana'.")

    # Visualización + explicación lado a lado
    col_vis, col_exp = st.columns([2, 1])

    with col_vis:
        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(results['transition_matrix'], cmap='Greens', interpolation='nearest')

        n_states = len(results['states'])
        ax.set_xticks(np.arange(n_states))
        ax.set_yticks(np.arange(n_states))
        ax.set_xticklabels(results['states'], fontsize=11, color='#333333')
        ax.set_yticklabels(results['states'], fontsize=11, color='#333333')
        ax.set_xlabel('Nivel de engagement el día siguiente', fontsize=12, fontweight='600', color='#333333')
        ax.set_ylabel('Nivel de engagement hoy', fontsize=12, fontweight='600', color='#333333')
        ax.set_title('¿Cómo cambia el engagement día a día?', fontsize=14, fontweight='bold', color='#333333', pad=15)

        for i in range(n_states):
            for j in range(n_states):
                ax.text(
                    j, i,
                    f"{results['transition_matrix'][i, j]:.2f}",
                    ha='center',
                    va='center',
                    color='#000000',
                    fontsize=10,
                    fontweight='600'
                )

        cbar = plt.colorbar(im, ax=ax, label='Probabilidad')
        cbar.ax.tick_params(labelsize=10, colors='#333333')
        cbar.set_label('Probabilidad', fontsize=11, color='#333333', fontweight='600')
        plt.tight_layout()
        st.pyplot(fig)

    with col_exp:
        st.markdown(
            """
            **📖 Cómo interpretar**

            - Cada **fila**: Nivel de engagement hoy
            - Cada **columna**: Nivel de engagement mañana
            - **Números**: % de probabilidad de cambio

            **Ejemplo práctico:**

            Si ves 0.30 en la fila "3" y columna "4":

            → Las personas en nivel 3 hoy tienen 30% de probabilidad de subir a nivel 4 mañana.

            **💼 Para tu negocio:**

            - Diagonal fuerte = Estabilidad (se mantienen)
            - Arriba de diagonal = Mejora (suben de nivel)
            - Abajo de diagonal = Deterioro (bajan de nivel)
            """
        )

    # Distribución estacionaria y tiempos de recurrencia
    if results['ergodic']:
        st.markdown("---")
        st.subheader("📊 Proyección de engagement a largo plazo")
        st.caption("Si las condiciones actuales se mantienen, esta es la distribución esperada de tu equipo en el futuro.")

        col_table, col_explain = st.columns([2, 1])

        with col_table:
            df_stationary = pd.DataFrame({
                'Estado': results['states'],
                'Probabilidad': results['stationary_distribution']
            })
            st.table(df_stationary.style.format({'Probabilidad': '{:.4f}'}))

            # Gráfica de barras
            fig2, ax2 = plt.subplots(figsize=(10, 5))
            ax2.bar(results['states'], results['stationary_distribution'], color=PRIMARY_GREEN, edgecolor='#333333', linewidth=1.5)
            ax2.set_xlabel('Nivel de engagement', fontsize=12, fontweight='600', color='#333333')
            ax2.set_ylabel('% de empleados esperado', fontsize=12, fontweight='600', color='#333333')
            ax2.set_title('¿Dónde estará mi equipo en el futuro?', fontsize=14, fontweight='bold', color='#333333', pad=15)
            ax2.tick_params(axis='both', labelsize=11, colors='#333333')
            ax2.grid(axis='y', alpha=0.3, linestyle='--')

            # Agregar valores encima de las barras
            for i, (state, prob) in enumerate(zip(results['states'], results['stationary_distribution'])):
                ax2.text(i, prob + 0.01, f'{prob:.1%}', ha='center', va='bottom', fontsize=10, fontweight='bold', color='#333333')

            plt.tight_layout()
            st.pyplot(fig2)

        with col_explain:
            st.markdown(
                """
                **💡 ¿Qué estás viendo?**

                Esta gráfica muestra **dónde estará tu equipo** si las condiciones actuales se mantienen.

                **Ejemplo:**

                Si el nivel 4 muestra 35%:

                → En el futuro, esperamos que 35 de cada 100 personas estén en nivel 4.

                **🎯 Decisiones de negocio:**

                ✓ Identificar si el futuro es favorable o preocupante

                ✓ Comparar diferentes equipos/países/studios

                ✓ Justificar inversiones en mejora de clima

                ✓ Establecer metas realistas basadas en datos
                """
            )

        st.markdown("---")
        st.subheader("⏱️ Frecuencia de visita a cada nivel")
        st.caption("¿Cada cuántos días, en promedio, los empleados vuelven a experimentar cada nivel de engagement?")

        col_times, col_times_exp = st.columns([2, 1])

        with col_times:
            df_times = pd.DataFrame({
                'Nivel': results['states'],
                'Cada cuántos días se visita': results['mean_recurrence_times']
            })
            st.table(df_times.style.format({'Cada cuántos días se visita': '{:.1f} días'}))

        with col_times_exp:
            st.markdown(
                """
                **💡 Interpretación simple**

                **Número bajo** (ej: 3 días)

                → Nivel muy común, se visita frecuentemente

                **Número alto** (ej: 25 días)

                → Nivel raro, se visita ocasionalmente

                **🎯 Para tu negocio:**

                ✓ Niveles con frecuencia baja = Tu "zona habitual"

                ✓ Niveles con frecuencia alta = Estados poco comunes

                ✓ Útil para entender la "normalidad" de tu equipo
                """
            )
    else:
        st.markdown("---")
        st.warning(
            "⚠️ **Sistema Inestable:** No se puede calcular una proyección única de largo plazo porque el sistema no converge a un equilibrio."
        )

        # Explicación detallada del problema
        with st.expander("🔍 ¿Por qué no es ergódica? (Click para detalles)"):
            st.markdown(
                f"""
                **🔍 Diagnóstico técnico:**

                - **Conectividad completa:** {'✅ Sí' if results['irreducible'] else '❌ No - Hay niveles aislados'}
                - **Sin ciclos forzados:** {'✅ Sí' if results['aperiodic'] else '❌ No - Hay patrones cíclicos'}

                Para ser un sistema estable (ergódico), necesita ambas condiciones.

                ---

                **🔎 ¿Qué está pasando?**

                {'**Niveles aislados:** Algunos niveles de engagement no se conectan con otros. Puede haber "trampas" de las que es difícil salir.' if not results['irreducible'] else ''}

                {'**Patrones cíclicos:** El engagement sigue ciclos regulares en lugar de poder cambiar libremente en cualquier momento.' if not results['aperiodic'] else ''}

                ---

                **💼 Impacto en el negocio:**

                ⚠️ No hay un "futuro único" predecible - el resultado final depende mucho de dónde empiezas

                ⚠️ Puede haber niveles de los que es muy difícil salir una vez que llegas ahí

                ⚠️ Las intervenciones de HR tendrán efectos diferentes según el nivel actual del equipo

                **💡 Recomendación:**

                Analiza los filtros aplicados. Grupos muy pequeños o muy homogéneos pueden generar este comportamiento.
                """
            )

        # Mostrar tiempos de recurrencia estimados por simulación
        if 'mean_recurrence_times' in results and results.get('recurrence_method') == 'simulation':
            st.markdown("---")
            st.subheader("⏱️ Frecuencia de visita a cada nivel (estimación)")
            st.caption("Valores estimados mediante simulación. Indican qué tan común es cada nivel de engagement.")

            col_times, col_times_exp = st.columns([2, 1])

            with col_times:
                # Preparar datos para la tabla
                recurrence_data = []
                for state, time in zip(results['states'], results['mean_recurrence_times']):
                    if np.isinf(time):
                        recurrence_data.append({'Nivel': state, 'Cada cuántos días se visita': '∞ (no alcanzable)'})
                    else:
                        recurrence_data.append({'Nivel': state, 'Cada cuántos días se visita': f"{time:.1f} días"})

                df_times = pd.DataFrame(recurrence_data)
                st.table(df_times)

                st.info(
                    "ℹ️ **Nota:** Valores estimados mediante simulación de 50,000 días. "
                    "Proporcionan una buena aproximación del comportamiento real."
                )

            with col_times_exp:
                st.markdown(
                    """
                    **💡 Interpretación**

                    **Número bajo** (ej: 5 días)

                    → Nivel frecuente en tu equipo

                    **Número alto** (ej: 30 días)

                    → Nivel poco común

                    **∞ (infinito)**

                    → Nivel prácticamente inalcanzable o "trampa" de la que no se sale

                    **⚠️ Importante:**

                    Al no ser ergódica, el comportamiento depende mucho del punto de partida. Estos son promedios estimados.

                    **🎯 Para tu negocio:**

                    ✓ Detecta niveles "trampa" (difíciles de abandonar)

                    ✓ Identifica niveles inaccesibles

                    ✓ Entiende la dinámica real de tu equipo
                    """
                )

    # ================== SECCIÓN DE SIMULACIÓN PROTEGIDA (V3.0 - ULTRA ROBUSTA) ==================
    try:
        # Validación inicial del analyzer
        if (not hasattr(analyzer, 'state_to_idx') or 
            analyzer.state_to_idx is None or 
            len(analyzer.state_to_idx) == 0):
            st.info(
                "ℹ️ **Primero ejecuta un análisis** - Selecciona uno de los ejemplos del menú principal o crea un análisis personalizado con filtros."
            )
            return
        
        st.markdown("---")
        st.subheader("🎲 Simulador: ¿Cómo evolucionará el engagement?")
        st.caption("Genera una trayectoria probable del engagement basada en los patrones históricos de tu equipo.")
        
        # Nota especial si no es ergódica
        if not results['ergodic']:
            st.info(
                "ℹ️ **Nota:** Puedes simular de todos modos, pero el resultado dependerá mucho del nivel inicial que elijas."
            )

        # Obtener estados disponibles (SIN return si falla)
        available_states = []
        try:
            available_states = sorted([float(k) for k in analyzer.state_to_idx.keys()])
        except Exception:
            pass
        
        if len(available_states) == 0:
            st.warning("⚠️ No hay estados disponibles para simular.")
            st.info("Intenta ejecutar el análisis nuevamente o con filtros diferentes.")
            return

        # Inicializar session_state para persistir resultados
        if 'simulation_result' not in st.session_state:
            st.session_state.simulation_result = None
        if 'simulation_params' not in st.session_state:
            st.session_state.simulation_params = None
        
        # ============ CONTROLES - SIEMPRE VISIBLES ============
        col_ctrl1, col_ctrl2, col_ctrl3 = st.columns([2, 2, 1])
        
        with col_ctrl1:
            initial_state = st.selectbox(
                "Estado inicial de engagement",
                options=available_states,
                index=len(available_states) // 2,
                format_func=lambda x: f"{x:.1f}",
                key="sim_initial_state"
            )

        with col_ctrl2:
            n_steps = st.slider(
                "Número de días a simular",
                min_value=10,
                max_value=200,
                value=60,
                step=10,
                key="sim_n_steps"
            )
        
        with col_ctrl3:
            st.write("")  # Espaciado
            st.write("")  # Espaciado
            run_simulation = st.button("🚀 Simular", type="primary", use_container_width=True, key="sim_button")
        
        # ============ EJECUCIÓN - SOLO CUANDO SE PRESIONA BOTÓN ============
        if run_simulation:
            # Limpiar resultado anterior
            st.session_state.simulation_result = None
            st.session_state.simulation_params = None
            
            with st.spinner("Ejecutando simulación..."):
                simulation_error = None
                simulated = None
                
                try:
                    # Obtener índice y probabilidades
                    current_idx = analyzer.state_to_idx[float(initial_state)]
                    row_probs = analyzer.transition_matrix[current_idx]
                    
                    # Validar que hay transiciones
                    if row_probs.sum() == 0:
                        simulation_error = {
                            'type': 'no_transitions',
                            'state': initial_state
                        }
                    else:
                        # Mostrar advertencia si es casi absorbente (pero simular de todos modos)
                        if row_probs[current_idx] > 0.95:
                            st.warning(
                                f"⚠️ El estado {initial_state} es casi absorbente "
                                f"(probabilidad {row_probs[current_idx]:.2%} de quedarse). "
                                "La simulación probablemente se quedará mayormente en este nivel."
                            )
                        
                        # Ejecutar simulación
                        simulated = analyzer.simulate(float(initial_state), n_steps=n_steps)
                        
                        # Guardar en session_state
                        st.session_state.simulation_result = simulated
                        st.session_state.simulation_params = {
                            'initial_state': initial_state,
                            'n_steps': n_steps
                        }
                
                except Exception as e:
                    simulation_error = {
                        'type': 'execution_error',
                        'message': str(e),
                        'state': initial_state
                    }
                
                # Mostrar errores si los hay
                if simulation_error:
                    if simulation_error['type'] == 'no_transitions':
                        st.error(
                            f"⚠️ El estado {simulation_error['state']} no tiene transiciones salientes "
                            f"en los datos filtrados. La simulación se quedaría congelada."
                        )
                        st.info("💡 **Solución:** Selecciona otro estado inicial o usa filtros menos restrictivos.")
                    else:
                        st.error(f"❌ Error durante la simulación: {simulation_error['message']}")
                        with st.expander("Ver detalles del error"):
                            st.write(f"**Estados disponibles:** {available_states}")
                            st.write(f"**Estado inicial:** {simulation_error['state']}")
                            st.write(f"**Pasos:** {n_steps}")
                            if 'message' in simulation_error:
                                st.code(simulation_error['message'])
                        st.info("💡 Intenta con un estado inicial diferente.")
        
        # ============ MOSTRAR RESULTADOS SI EXISTEN ============
        if st.session_state.simulation_result is not None:
            simulated = st.session_state.simulation_result
            params = st.session_state.simulation_params
            
            st.success(f"✅ Simulación completada - {len(simulated)} estados generados")
            st.markdown("---")
            
            # Gráfica
            fig3, ax3 = plt.subplots(figsize=(14, 6))
            ax3.plot(simulated, marker='o', linewidth=2.5, markersize=5, color=PRIMARY_GREEN, markeredgecolor='#333333', markeredgewidth=1)
            ax3.set_xlabel('Día', fontsize=13, fontweight='600', color='#333333')
            ax3.set_ylabel('Nivel de engagement', fontsize=13, fontweight='600', color='#333333')
            ax3.set_title(
                f'Evolución simulada del engagement - Iniciando en nivel {params["initial_state"]}',
                fontsize=15,
                fontweight='bold',
                color='#333333',
                pad=20
            )
            ax3.tick_params(axis='both', labelsize=11, colors='#333333')
            ax3.grid(True, alpha=0.3, linestyle='--')
            ax3.set_facecolor('#FAFAFA')
            plt.tight_layout()
            st.pyplot(fig3)
            plt.close(fig3)
            
            # Información y estadísticas
            col_info1, col_info2 = st.columns([1, 1])
            
            with col_info1:
                st.markdown(
                    """
                    **💡 ¿Qué estás viendo?**

                    Esta es una **trayectoria posible** del engagement basada en tus datos históricos.

                    **No es una predicción exacta**, sino un escenario probable que muestra cómo puede fluctuar el engagement día a día.

                    **🎯 Útil para:**

                    ✓ Visualizar la volatilidad del engagement

                    ✓ Entender rangos típicos de variación

                    ✓ Comunicar patrones a stakeholders

                    ✓ Planificar intervenciones
                    """
                )

            with col_info2:
                st.markdown("**📊 Estadísticas de la simulación**")
                unique_visited = sorted(set(simulated))
                st.write(f"**Niveles visitados:** {', '.join(map(str, unique_visited))}")
                st.write(f"**Nivel final:** {simulated[-1]}")
                st.write(f"")
                st.write(f"**Tiempo en cada nivel:**")
                for state in unique_visited:
                    count = simulated.count(state)
                    percentage = (count / len(simulated)) * 100
                    st.write(f"• Nivel {state}: **{percentage:.0f}%** ({count} días)")
    except Exception as e:
        st.error(f"❌ Error al preparar la sección de simulación: {e}")
        with st.expander("Ver detalles técnicos"):
            import traceback
            st.code(traceback.format_exc())
        st.info(
            "💡 Es posible que el subconjunto de datos seleccionado genere una cadena muy "
            "degenerada. Prueba relajando los filtros o eligiendo otro ejemplo."
        )



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

    # Solo incluir mean_recurrence_times si están disponibles (cuando es ergódica)
    if 'mean_recurrence_times' in results:
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