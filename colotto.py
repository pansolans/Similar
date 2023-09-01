import pandas as pd
import numpy as np
import streamlit as st
from scipy.spatial import distance

# Función para cargar el dataset
@st.cache_resource
def load_data():
    return pd.read_excel("DFseptiembreColotto3.xlsx")

# Función principal de Streamlit
def main():
    df = load_data()

    # Widgets para los filtros
    st.sidebar.header("Pampa Fútbol Analytics")

    # Widgets para los filtros
    st.sidebar.header("Filtros")
    
    # Filtro para seleccionar las variables
    cols_to_select = list(range(2,107)) + list(range(136,163))
    selected_vars = st.sidebar.multiselect("Seleccionar variables:", options=df.columns[cols_to_select].tolist())
    
    # Filtro para competitionId
    competition_ids = st.sidebar.multiselect("Seleccionar competitionId:", options=df["competitionId"].unique().tolist())
    
    # Filtro para Pos_principal
    pos_principales = st.sidebar.multiselect("Seleccionar Pos_principal:", options=df["Pos_principal"].unique().tolist())
    
    # Filtro para minutesOnField
    min_minutes, max_minutes = st.sidebar.slider("Seleccionar rango de minutesOnField:", int(df["minutesOnField"].min()), int(df["minutesOnField"].max()), (int(df["minutesOnField"].min()), int(df["minutesOnField"].max())))
    
    if not hasattr(st.session_state, 'merged_df'):
        st.session_state.merged_df = None

    # Si no se ha seleccionado nada en los filtros, mostrar una advertencia y no continuar
    if not selected_vars or not competition_ids or not pos_principales:
        st.warning("Por favor, selecciona opciones en todos los filtros para continuar.")
        return

    # Filtrar DataFrame según las selecciones
    filtered_df = df[(df["competitionId"].isin(competition_ids)) & (df["Pos_principal"].isin(pos_principales)) & (df["minutesOnField"] >= min_minutes) & (df["minutesOnField"] <= max_minutes)]
    
    # Filtro para seleccionar un Jugador
    name = st.sidebar.selectbox("Seleccionar Jugador:", options=[None] + filtered_df["Jugador"].unique().tolist())
    if not name:
        st.warning("Por favor, selecciona un jugador para continuar.")
        return

    if st.sidebar.button("Búsqueda"):
        # Procesamiento
        Datos_Ejercicio_4 = filtered_df[selected_vars]
        Datos_Ejercicio_4["Name"] = filtered_df["Jugador"]
        Datos_Ejercicio_4.fillna(0, inplace=True)
        
        # Regularización de la matriz de covarianza
        regularization_factor = 1e-4
        cov_matrix = Datos_Ejercicio_4.drop(columns="Name").cov()
        regularized_cov_matrix = cov_matrix + np.eye(cov_matrix.shape[0]) * regularization_factor
        inv_cov = np.linalg.inv(regularized_cov_matrix)
        
        # Calcular la distancia de Mahalanobis
        player_data_df = Datos_Ejercicio_4[Datos_Ejercicio_4["Name"] == name].drop(columns="Name")
        player_data = player_data_df.values[0]
        sim = Datos_Ejercicio_4.apply(lambda row: distance.mahalanobis(row.drop("Name").values, player_data, inv_cov), axis=1)
        
        sim = pd.DataFrame(sim, columns=["similitud"])
        sims95 = sim["similitud"].quantile(0.95)
        sim["similitud_2"] = 100 - (sim["similitud"] / sims95) * 100
        sim["Name"] = Datos_Ejercicio_4["Name"].values
        sim = sim[["Name", "similitud", "similitud_2"]]
        sim.columns = ["Name", "Dist_Euc_OP_Norm_mah", "Sim_Euc_OP_Norm_mah"]
        
        sim_sorted = sim.sort_values(by="Sim_Euc_OP_Norm_mah", ascending=False)
        sim_filtered = sim_sorted[["Name", "Sim_Euc_OP_Norm_mah"]]
        merged_df = pd.merge(sim_filtered, filtered_df[["Jugador", "Edad", "urlImagen.x", "equipoActual_nombre", "urlteam", "competitionId"]], left_on="Name", right_on="Jugador", how="left")
        st.session_state.merged_df = merged_df
    st.header("Jugadores Similares")


    # Filtro post-búsqueda para Edad
    min_age, max_age = st.slider("Rango de Edad:", int(df["Edad"].min()), int(df["Edad"].max()), (int(df["Edad"].min()), int(df["Edad"].max())))
    
    # Filtro post-búsqueda para competitionId
            # Título "Jugadores Similares"
  
    post_search_competition_ids = st.multiselect("Seleccionar competitionId para filtrar la tabla:", options=st.session_state.merged_df["competitionId"].unique().tolist())
    
    if st.session_state.merged_df is not None:
        filtered_by_age = st.session_state.merged_df[(st.session_state.merged_df["Edad"] >= min_age) & (st.session_state.merged_df["Edad"] <= max_age)]
        
        if post_search_competition_ids:
            filtered_by_age = filtered_by_age[filtered_by_age["competitionId"].isin(post_search_competition_ids)]
        

        # Visualización con imágenes
        for index, row in filtered_by_age.head(20).iterrows():
            col1, col2, col3 = st.columns([1,3,1])
            if isinstance(row["urlteam"], str):
                col1.image(row["urlteam"], width=50)
            if isinstance(row["urlImagen.x"], str):
                col2.image(row["urlImagen.x"], caption=row["Name"], width=100)
            col3.markdown(f"<span style='font-weight: bold;'>Edad:</span> {int(row['Edad'])}", unsafe_allow_html=True)
            col3.markdown(f"<span style='font-weight: bold;'>Equipo:</span> {row['equipoActual_nombre']}", unsafe_allow_html=True)
            col3.markdown(f"<span style='font-weight: bold; color: red;'>Similitud:</span> {row['Sim_Euc_OP_Norm_mah']:.2f}", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
