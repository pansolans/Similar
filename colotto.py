import pandas as pd
import numpy as np
import streamlit as st
from scipy.spatial import distance
import requests
from io import BytesIO

# Función para cargar el dataset
@st.cache_resource
def load_data():
    url = "https://raw.githubusercontent.com/pansolans/Similar/main/df24.xlsx"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.content
        df = pd.read_excel(BytesIO(data))
        return df
    else:
        st.error("No se pudo descargar el archivo.")
        return pd.DataFrame()  # Retorna un DataFrame vacío en caso de error

def calculate_mahalanobis_distance(Datos_Ejercicio_4, name):
    regularization_factor = 1e-4
    cov_matrix = Datos_Ejercicio_4.drop(columns="Name").cov()
    regularized_cov_matrix = cov_matrix + np.eye(cov_matrix.shape[0]) * regularization_factor
    inv_cov = np.linalg.inv(regularized_cov_matrix)

    player_data_df = Datos_Ejercicio_4[Datos_Ejercicio_4["Name"] == name].drop(columns="Name")
    player_data = player_data_df.values[0]
    sim = Datos_Ejercicio_4.apply(lambda row: distance.mahalanobis(row.drop("Name").values, player_data, inv_cov), axis=1)
    return sim

def export_to_excel(df, file_name):
    df.to_excel(file_name, index=False)

def main():
    df = load_data()

    if df.empty:
        st.warning("No se pudo cargar el archivo de datos.")
        return

    st.sidebar.header("Pampa Fútbol Analytics")
    st.sidebar.header("Filtros")

    cols_to_select = list(range(23,166))
    selected_vars = st.sidebar.multiselect("Seleccionar variables:", options=df.columns[cols_to_select].tolist())
    competition_ids = st.sidebar.multiselect("Seleccionar competitionId:", options=df["competitionId"].unique().tolist())
    season_ids = st.sidebar.multiselect("Seleccionar season:", options=df["season"].unique().tolist())
    pos_principales = st.sidebar.multiselect("Seleccionar Pos_principal:", options=df["Pos_principal"].unique().tolist())
    min_minutes, max_minutes = st.sidebar.slider("Seleccionar rango de minutesOnField:", int(df["minutesOnField"].min()), int(df["minutesOnField"].max()), (int(df["minutesOnField"].min()), int(df["minutesOnField"].max())))

    if not selected_vars or not competition_ids or not season_ids or not pos_principales:
        st.warning("Por favor, selecciona opciones en todos los filtros para continuar.")
        return

    filtered_df = df[
        (df["competitionId"].isin(competition_ids)) & 
        (df["season"].isin(season_ids)) & 
        (df["Pos_principal"].isin(pos_principales)) & 
        (df["minutesOnField"] >= min_minutes) & 
        (df["minutesOnField"] <= max_minutes)
    ]

    name = st.sidebar.selectbox("Seleccionar Jugador:", options=[None] + filtered_df["Jugador"].unique().tolist())
    if not name:
        st.warning("Por favor, selecciona un jugador para continuar.")
        return

    if st.sidebar.button("Búsqueda"):
        Datos_Ejercicio_4 = filtered_df[selected_vars]
        Datos_Ejercicio_4["Name"] = filtered_df["Jugador"]
        Datos_Ejercicio_4.fillna(0, inplace=True)

        sim = calculate_mahalanobis_distance(Datos_Ejercicio_4, name)

        sim = pd.DataFrame(sim, columns=["similitud"])
        sims95 = sim["similitud"].quantile(0.95)
        sim["similitud_2"] = 100 - (sim["similitud"] / sims95) * 100
        sim["Name"] = Datos_Ejercicio_4["Name"].values
        sim = sim[["Name", "similitud", "similitud_2"]]
        sim.columns = ["Name", "Dist_Euc_OP_Norm_mah", "Sim_Euc_OP_Norm_mah"]

        sim_sorted = sim.sort_values(by="Sim_Euc_OP_Norm_mah", ascending=False)
        sim_filtered = sim_sorted[["Name", "Sim_Euc_OP_Norm_mah"]]
        merged_df = pd.merge(sim_filtered, filtered_df[["Jugador", "Edad", "urlImagen.y", "equipoActual_nombre", "urlImagen.x", "competitionId", "minutesOnField"]], left_on="Name", right_on="Jugador", how="left")
        st.session_state.merged_df = merged_df
        st.session_state.displayed_df = None

    st.header("Jugadores Similares")

    min_age, max_age = st.slider("Minutos jugados:", int(df["minutesOnField"].min()), int(df["minutesOnField"].max()), (int(df["minutesOnField"].min()), int(df["minutesOnField"].max())))
    post_search_competition_ids = st.multiselect("Seleccionar competitionId para filtrar la tabla:", options=st.session_state.merged_df["competitionId"].unique().tolist())

    if st.session_state.merged_df is not None:
        filtered_by_age = st.session_state.merged_df[(st.session_state.merged_df["minutesOnField"] >= min_age) & (st.session_state.merged_df["minutesOnField"] <= max_age)]

        if post_search_competition_ids:
            filtered_by_age = filtered_by_age[filtered_by_age["competitionId"].isin(post_search_competition_ids)]

        st.session_state.displayed_df = filtered_by_age.head(20)  # Almacena solo los primeros 20 registros

        for index, row in st.session_state.displayed_df.iterrows():
            col1, col2, col3 = st.columns([1,3,1])
            if isinstance(row["urlImagen.y"], str):
                col1.image(row["urlImagen.y"], width=50)
            if isinstance(row["urlImagen.x"], str):
                col2.image(row["urlImagen.x"], caption=row["Name"], width=100)
            col3.markdown(f"<span style='font-weight: bold;'>Edad:</span> {int(row['Edad'])}", unsafe_allow_html=True)
            col3.markdown(f"<span style='font-weight: bold;'>Equipo:</span> {row['equipoActual_nombre']}", unsafe_allow_html=True)
            col3.markdown(f"<span style='font-weight: bold;'>Similitud:</span> {row['Sim_Euc_OP_Norm_mah']:.2f}", unsafe_allow_html=True)
            col3.markdown(f"<span style='font-weight: bold; color: red;'>Minutos:</span> {row['minutesOnField']}", unsafe_allow_html=True)

    # Botón para exportar a Excel
    if st.session_state.displayed_df is not None:
        if st.button('Exportar a Excel'):
            file_name = 'jugadores_similares.xlsx'  # Define el nombre del archivo aquí
            
            # Debug: Print DataFrame
            st.write("Exportando DataFrame:")
            st.dataframe(st.session_state.displayed_df)
            
            # Debug: Check if file is being created
            export_to_excel(st.session_state.displayed_df, file_name)
            st.write("Archivo exportado como:", file_name)
            
            # Provide download link
            with open(file_name, "rb") as file:
                btn = st.download_button(label="Descargar Excel",
                                         data=file,
                                         file_name=file_name,
                                         mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
                st.success(f'Archivo listo para descargar: {file_name}')

if __name__ == "__main__":
    main()

