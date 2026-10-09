from pathlib import Path
import io
import warnings

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore", category=FutureWarning)

st.set_page_config(page_title="BankMarketing | EDA", page_icon="📊", layout="wide")
sns.set_theme(style="whitegrid")

EXPECTED_COLUMNS = [
    "age", "job", "marital", "education", "default", "housing", "loan",
    "contact", "month", "day_of_week", "duration", "campaign", "pdays",
    "previous", "poutcome", "emp.var.rate", "cons.price.idx", "cons.conf.idx",
    "euribor3m", "nr.employed", "y"
]

class DataAnalyzer:
    """Encapsula tareas frecuentes de análisis exploratorio."""
    def __init__(self, data: pd.DataFrame):
        self.data = data.copy()

    def classify_variables(self):
        numeric = self.data.select_dtypes(include=np.number).columns.tolist()
        categorical = self.data.select_dtypes(exclude=np.number).columns.tolist()
        return numeric, categorical

    def descriptive_stats(self):
        return self.data.describe(include="all").T

    def missing_summary(self):
        result = self.data.isna().sum().to_frame("Valores nulos")
        result["% nulos"] = (result["Valores nulos"] / max(len(self.data), 1) * 100).round(2)
        return result.sort_values("Valores nulos", ascending=False)

    def target_rate(self):
        if "y" not in self.data.columns or self.data.empty:
            return np.nan
        target = self.data["y"].astype(str).str.strip().str.lower()
        return float(target.eq("yes").mean() * 100)

    def numeric_summary(self, column):
        return self.data[column].describe() if column in self.data.select_dtypes(include=np.number) else None


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Limpia nombres de columnas y espacios comunes sin alterar los valores."""
    df = df.copy()
    df.columns = [str(c).strip().strip('"').strip("'") for c in df.columns]
    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = df[col].astype("string").str.strip().str.strip('"')
        df[col] = df[col].replace({"": pd.NA, "?": pd.NA})
    return df


def load_csv(uploaded_file) -> pd.DataFrame:
    """Prueba separadores habituales y devuelve el archivo que tenga más columnas."""
    raw = uploaded_file.getvalue()
    attempts = []
    for sep in [";", ",", "\t", None]:
        try:
            candidate = pd.read_csv(io.BytesIO(raw), sep=sep, engine="python" if sep is None else "c")
            attempts.append(candidate)
        except Exception:
            continue
    if not attempts:
        raise ValueError("No fue posible interpretar el CSV. Verifica el archivo y su codificación.")
    df = max(attempts, key=lambda item: item.shape[1])
    if df.shape[1] <= 1:
        raise ValueError("El archivo parece tener una sola columna. Verifica el separador del CSV.")
    return normalize_columns(df)


def show_fig(fig):
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


def safe_rate_table(df, row, target="y"):
    tmp = df[[row, target]].copy()
    tmp[target] = tmp[target].astype(str).str.lower().str.strip()
    tab = pd.crosstab(tmp[row].fillna("Sin dato"), tmp[target], normalize="index") * 100
    return tab


def main():
    st.sidebar.title("📊 BankMarketing")
    st.sidebar.caption("Análisis exploratorio de datos · 2026")
    module = st.sidebar.radio("Navegación", ["Home", "Carga del dataset", "Análisis exploratorio", "Conclusiones"])
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Autora**  \nElena Loayza Arteaga")
    st.sidebar.markdown("Python for Analytics")

    if module == "Home":
        st.title("BankMarketing")
        st.subheader("Análisis exploratorio de campañas de marketing bancario")
        left, right = st.columns([1.7, 1])
        with left:
            st.markdown("""
            Este proyecto explora los datos de una campaña de marketing de una institución financiera para identificar patrones asociados con la aceptación de la oferta (`y`). El propósito es convertir los datos en información útil para revisar segmentos, canales y características de contacto.

            **Pregunta de negocio:** ¿qué comportamientos y variables muestran diferencias entre clientes que aceptaron la campaña y quienes no la aceptaron?

            El caso de estudio señala que la efectividad pasó del 12% al 8% durante los últimos seis meses. El análisis ayuda a explorar posibles relaciones en los datos, sin construir un modelo predictivo ni atribuir causalidad.
            """)
            st.markdown("**Autora:** Elena Loayza Arteaga  \n**Especialización:** Python for Analytics  \n**Año:** 2026")
        with right:
            st.metric("Enfoque", "EDA")
            st.metric("Variables esperadas", "21")
            st.info("Carga el archivo CSV desde el módulo **Carga del dataset**. La aplicación no ejecuta análisis hasta que se haya cargado y validado un archivo.")
        st.markdown("### Acerca del dataset")
        st.write("El conjunto incluye variables demográficas y de empleo, situación crediticia, canal y momento del contacto, historial de campañas e indicadores económicos. La variable objetivo `y` indica si el cliente aceptó la campaña (`yes`) o no (`no`).")
        st.markdown("### Tecnologías utilizadas")
        st.write("Python · Streamlit · Pandas · NumPy · Matplotlib · Seaborn")
        st.markdown("### Flujo de trabajo")
        c1, c2, c3 = st.columns(3)
        c1.info("**1. Cargar**\n\nValidar y revisar el CSV")
        c2.info("**2. Explorar**\n\nDescribir, comparar y visualizar")
        c3.info("**3. Concluir**\n\nResumir hallazgos y decisiones posibles")
        return

    st.title("BankMarketing | Exploración de datos")
    uploaded = st.sidebar.file_uploader("Carga BankMarketing.csv", type=["csv"], help="Se aceptan archivos CSV separados por punto y coma o coma.")
    if uploaded is None:
        st.warning("Primero carga el archivo BankMarketing.csv en el panel lateral. Los módulos de análisis permanecerán bloqueados hasta cargar los datos.")
        st.markdown("**Formato esperado:** CSV con encabezados y variables como `age`, `job`, `duration` y `y`.")
        return
    try:
        df = load_csv(uploaded)
    except Exception as exc:
        st.error(f"No se pudo cargar el archivo: {exc}")
        return
    if df.empty or len(df.columns) < 2:
        st.error("El dataset está vacío o no contiene suficientes columnas para analizar.")
        return
    analyzer = DataAnalyzer(df)

    if module == "Carga del dataset":
        st.header("Módulo 2 · Carga y validación")
        st.success(f"Archivo cargado correctamente: {uploaded.name}")
        a, b, c = st.columns(3)
        a.metric("Filas", f"{df.shape[0]:,}")
        b.metric("Columnas", f"{df.shape[1]:,}")
        c.metric("Valores nulos", f"{int(df.isna().sum().sum()):,}")
        st.subheader("Vista previa del dataset")
        st.dataframe(df.head(10), use_container_width=True)
        st.caption(f"Se muestran las primeras 10 filas de {df.shape[0]:,} registros.")
        missing_cols = [c for c in EXPECTED_COLUMNS if c not in df.columns]
        if missing_cols:
            st.warning("No se encontraron algunas columnas esperadas del caso: " + ", ".join(missing_cols) + ". La aplicación seguirá funcionando con las columnas disponibles.")
        else:
            st.success("Se identificaron las 21 columnas descritas en el caso de estudio.")
        st.download_button("Descargar copia limpia de encabezados y espacios", df.to_csv(index=False).encode("utf-8-sig"), "BankMarketing_limpio.csv", "text/csv")
        return

    if module == "Conclusiones":
        st.header("Conclusiones y recomendaciones")
        if "y" not in df.columns:
            st.warning("No existe la columna objetivo `y`; revisa que hayas cargado el dataset correcto.")
            return
        target = df["y"].astype(str).str.strip().str.lower()
        rate = target.eq("yes").mean() * 100
        st.metric("Tasa de aceptación observada", f"{rate:.2f}%")
        st.caption("Cálculo descriptivo sobre el archivo cargado; no implica causalidad ni predicción.")
        conclusions = [
            f"La tasa de aceptación en el archivo cargado es {rate:.2f}%; sirve como línea base para comparar segmentos y campañas.",
            "La duración del contacto debe interpretarse junto con el resto de variables: una asociación descriptiva no demuestra que alargar llamadas cause más aceptación.",
            "Las diferencias entre canales, meses y días pueden orientar una revisión operativa, siempre considerando el tamaño de cada grupo.",
            "Los valores desconocidos o faltantes deben reportarse explícitamente antes de comparar perfiles; no conviene asumir que equivalen a una categoría real.",
            "Se recomienda usar los hallazgos para formular hipótesis y diseñar seguimiento comercial medible, no para afirmar resultados futuros ni causalidad."
        ]
        for i, item in enumerate(conclusions, 1):
            st.markdown(f"**{i}.** {item}")
        st.markdown("### Nota metodológica")
        st.write("Las conclusiones se generan de forma descriptiva y deben revisarse a la luz de las tablas y gráficos del EDA. Si se modifica el archivo, los resultados y la tasa de aceptación se actualizarán.")
        return

    st.header("Análisis Exploratorio de Datos (EDA)")
    st.write("Explora calidad, distribución y relaciones entre variables. Usa los controles para adaptar las visualizaciones al dataset cargado.")
    tabs = st.tabs(["1 · General", "2 · Variables", "3 · Descriptiva", "4 · Faltantes", "5 · Numéricas", "6 · Categóricas", "7 · Numérica vs objetivo", "8 · Categórica vs objetivo", "9 · Análisis dinámico", "10 · Hallazgos"])

    with tabs[0]:
        st.subheader("Ítem 1. Información general del dataset")
        c1, c2, c3 = st.columns(3)
        c1.metric("Registros", f"{len(df):,}")
        c2.metric("Variables", len(df.columns))
        c3.metric("Duplicados", f"{int(df.duplicated().sum()):,}")
        st.markdown("**Tipos de datos y valores no nulos**")
        info = pd.DataFrame({"Tipo de dato": df.dtypes.astype(str), "No nulos": df.notna().sum(), "Nulos": df.isna().sum()})
        st.dataframe(info, use_container_width=True)
        buffer = io.StringIO()
        df.info(buf=buffer)
        with st.expander("Ver salida equivalente a DataFrame.info()"):
            st.code(buffer.getvalue())

    with tabs[1]:
        st.subheader("Ítem 2. Clasificación de variables")
        numeric, categorical = analyzer.classify_variables()
        st.markdown("La función personalizada `classify_variables()` clasifica las columnas según su tipo almacenado en Pandas.")
        c1, c2 = st.columns(2)
        c1.metric("Variables numéricas", len(numeric))
        c2.metric("Variables categóricas / no numéricas", len(categorical))
        st.dataframe(pd.DataFrame({"Tipo": ["Numérica"] * len(numeric) + ["Categórica / no numérica"] * len(categorical), "Variable": numeric + categorical}), use_container_width=True)
        st.caption("La clasificación depende del tipo de dato detectado. Una columna codificada como texto puede requerir revisión manual.")

    with tabs[2]:
        st.subheader("Ítem 3. Estadísticas descriptivas")
        st.write("La media resume el promedio; la mediana es el valor central y la desviación estándar describe la dispersión. La moda muestra el valor más frecuente cuando aplica.")
        st.dataframe(analyzer.descriptive_stats(), use_container_width=True)
        numeric, _ = analyzer.classify_variables()
        if numeric:
            col = st.selectbox("Variable numérica para interpretar", numeric, key="desc_num")
            s = df[col].dropna()
            if not s.empty:
                a, b, c, d = st.columns(4)
                a.metric("Media", f"{s.mean():,.2f}")
                b.metric("Mediana", f"{s.median():,.2f}")
                c.metric("Moda", f"{s.mode().iloc[0]:,.2f}" if not s.mode().empty else "N/D")
                d.metric("Desv. estándar", f"{s.std():,.2f}" if len(s) > 1 else "N/D")

    with tabs[3]:
        st.subheader("Ítem 4. Análisis de valores faltantes")
        miss = analyzer.missing_summary()
        st.dataframe(miss, use_container_width=True)
        miss_nonzero = miss[miss["Valores nulos"] > 0]
        if miss_nonzero.empty:
            st.success("No se detectaron valores nulos estándar en el dataset. Aun así, revisa categorías como 'unknown' si existen.")
        else:
            fig, ax = plt.subplots(figsize=(10, max(3, len(miss_nonzero) * 0.28)))
            sns.barplot(data=miss_nonzero.reset_index(names="Variable"), x="Valores nulos", y="Variable", ax=ax)
            ax.set_title("Cantidad de valores nulos por variable")
            show_fig(fig)
            st.write("Los valores faltantes pueden sesgar comparaciones. Antes de imputar o eliminar, conviene evaluar su proporción y significado en cada variable.")
        obj_cols = df.select_dtypes(include=["object", "string"]).columns
        unknowns = {c: int(df[c].astype(str).str.lower().isin(["unknown", "desconocido", "n/a"]).sum()) for c in obj_cols}
        unknowns = {k: v for k, v in unknowns.items() if v > 0}
        if unknowns:
            st.info("También hay etiquetas explícitas de desconocido que no son nulos técnicos: " + ", ".join(f"{k}: {v:,}" for k, v in unknowns.items()))

    with tabs[4]:
        st.subheader("Ítem 5. Distribución de variables numéricas")
        numeric, _ = analyzer.classify_variables()
        if not numeric:
            st.warning("No se detectaron variables numéricas.")
        else:
            chosen = st.multiselect("Selecciona variables numéricas", numeric, default=numeric[:min(3, len(numeric))], key="hist_cols")
            bins = st.slider("Número de intervalos del histograma", min_value=5, max_value=60, value=25, step=5)
            if chosen:
                for start in range(0, len(chosen), 2):
                    cols = st.columns(2)
                    for j, col in enumerate(chosen[start:start+2]):
                        with cols[j]:
                            fig, ax = plt.subplots(figsize=(6, 3.5))
                            sns.histplot(df[col].dropna(), bins=bins, kde=True, ax=ax)
                            ax.set_title(f"Distribución de {col}")
                            show_fig(fig)
                            st.caption(f"Media: {df[col].mean():,.2f} · Mediana: {df[col].median():,.2f}")

    with tabs[5]:
        st.subheader("Ítem 6. Análisis de variables categóricas")
        _, categorical = analyzer.classify_variables()
        if not categorical:
            st.warning("No se detectaron variables categóricas.")
        else:
            col = st.selectbox("Selecciona una variable categórica", categorical, key="cat_col")
            top_n = st.slider("Categorías a mostrar", 5, 30, 12, key="cat_top")
            counts = df[col].fillna("(nulo)").astype(str).value_counts().head(top_n)
            proportions = (counts / max(df[col].notna().sum(), 1) * 100).round(2)
            table = pd.DataFrame({"Conteo": counts, "% aproximado": proportions})
            left, right = st.columns([1, 1.3])
            left.dataframe(table, use_container_width=True)
            with right:
                fig, ax = plt.subplots(figsize=(7, 4))
                sns.barplot(x=counts.values, y=counts.index, ax=ax)
                ax.set_title(f"Frecuencia de {col}")
                ax.set_xlabel("Registros")
                ax.set_ylabel(col)
                show_fig(fig)

    with tabs[6]:
        st.subheader("Ítem 7. Análisis bivariado: numérica vs objetivo")
        if "y" not in df.columns:
            st.warning("No se encontró la columna `y`.")
        else:
            numeric, _ = analyzer.classify_variables()
            numeric = [c for c in numeric if c != "y"]
            if numeric:
                col = st.selectbox("Variable numérica", numeric, index=numeric.index("duration") if "duration" in numeric else 0, key="num_target")
                plot_df = df[[col, "y"]].dropna().copy()
                plot_df["y"] = plot_df["y"].astype(str)
                cap = st.checkbox("Limitar visualización al percentil 99 para facilitar la lectura", value=True)
                if cap and not plot_df.empty:
                    limit = plot_df[col].quantile(.99)
                    plot_df = plot_df[plot_df[col] <= limit]
                fig, ax = plt.subplots(figsize=(8, 4))
                sns.boxplot(data=plot_df, x="y", y=col, ax=ax)
                ax.set_title(f"{col} por resultado de campaña (y)")
                show_fig(fig)
                st.dataframe(plot_df.groupby("y")[col].agg(["count", "mean", "median", "std"]).round(2), use_container_width=True)
                st.caption("La comparación describe diferencias entre grupos; no demuestra que la variable provoque la aceptación.")

    with tabs[7]:
        st.subheader("Ítem 8. Análisis bivariado: categórica vs objetivo")
        if "y" not in df.columns:
            st.warning("No se encontró la columna `y`.")
        else:
            _, categorical = analyzer.classify_variables()
            categorical = [c for c in categorical if c != "y"]
            if categorical:
                col = st.selectbox("Variable categórica", categorical, index=categorical.index("education") if "education" in categorical else 0, key="cat_target")
                rate_table = safe_rate_table(df, col)
                if "yes" in rate_table.columns:
                    rate_table = rate_table.sort_values("yes", ascending=False)
                st.markdown("**Porcentaje de aceptación dentro de cada categoría**")
                st.dataframe(rate_table.round(2), use_container_width=True)
                if not rate_table.empty:
                    fig, ax = plt.subplots(figsize=(9, 4))
                    plot_series = rate_table["yes"] if "yes" in rate_table.columns else rate_table.iloc[:, 0]
                    sns.barplot(x=plot_series.values, y=plot_series.index.astype(str), ax=ax)
                    ax.set_xlabel("Porcentaje (%)")
                    ax.set_ylabel(col)
                    ax.set_title(f"Aceptación por {col}")
                    show_fig(fig)

    with tabs[8]:
        st.subheader("Ítem 9. Análisis dinámico con parámetros seleccionados")
        numeric, categorical = analyzer.classify_variables()
        left, right = st.columns(2)
        with left:
            xcol = st.selectbox("Variable para agrupar", categorical or df.columns.tolist(), key="dynamic_group")
        with right:
            value_options = numeric if numeric else df.columns.tolist()
            ycol = st.selectbox("Variable para resumir", value_options, key="dynamic_value")
        aggregation = st.selectbox("Estadístico", ["Media", "Mediana", "Suma", "Conteo"], key="dynamic_agg")
        groups = st.slider("Número máximo de grupos", 5, 30, 15, key="dynamic_n")
        if st.checkbox("Incluir solo grupos con al menos 10 registros", value=False, key="dynamic_filter"):
            counts = df[xcol].value_counts()
            dynamic_df = df[df[xcol].isin(counts[counts >= 10].index)]
        else:
            dynamic_df = df
        if aggregation == "Media":
            result = dynamic_df.groupby(xcol, dropna=False)[ycol].mean().sort_values(ascending=False)
        elif aggregation == "Mediana":
            result = dynamic_df.groupby(xcol, dropna=False)[ycol].median().sort_values(ascending=False)
        elif aggregation == "Suma":
            result = dynamic_df.groupby(xcol, dropna=False)[ycol].sum().sort_values(ascending=False)
        else:
            result = dynamic_df.groupby(xcol, dropna=False)[ycol].count().sort_values(ascending=False)
        result = result.head(groups)
        st.dataframe(result.rename(aggregation).to_frame(), use_container_width=True)
        if not result.empty:
            fig, ax = plt.subplots(figsize=(9, 4))
            sns.barplot(x=result.values, y=result.index.astype(str), ax=ax)
            ax.set_title(f"{aggregation} de {ycol} por {xcol}")
            show_fig(fig)

    with tabs[9]:
        st.subheader("Ítem 10. Hallazgos clave")
        if "y" in df.columns:
            target = df["y"].astype(str).str.strip().str.lower()
            rate = target.eq("yes").mean() * 100
            a, b, c = st.columns(3)
            a.metric("Aceptación", f"{rate:.2f}%")
            b.metric("Registros", f"{len(df):,}")
            b.metric("Variables", len(df.columns))
            if "duration" in df.columns:
                med_yes = df.loc[target.eq("yes"), "duration"].median()
                med_no = df.loc[target.eq("no"), "duration"].median()
                c.metric("Mediana duración (yes / no)", f"{med_yes:.0f} s / {med_no:.0f} s" if pd.notna(med_yes) and pd.notna(med_no) else "N/D")
            if "contact" in df.columns:
                rates = safe_rate_table(df, "contact")
                if "yes" in rates.columns and not rates.empty:
                    best = rates["yes"].idxmax()
                    st.write(f"**Asociación descriptiva:** el canal con mayor tasa de aceptación en este archivo es **{best}** ({rates.loc[best, 'yes']:.2f}%). Comprueba el tamaño de cada grupo antes de tomar decisiones.")
            if "duration" in df.columns and pd.notna(med_yes) and pd.notna(med_no):
                st.write(f"La mediana de duración es {med_yes:.0f} segundos para `yes` y {med_no:.0f} segundos para `no`. Es una diferencia observada, no una prueba de causalidad.")
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.countplot(x=target, order=["no", "yes"], ax=ax)
            ax.set_title("Distribución del resultado de campaña")
            ax.set_xlabel("Resultado (y)")
            ax.set_ylabel("Cantidad de registros")
            show_fig(fig)
            st.markdown("**Lectura recomendada:** utilizar estos resultados para priorizar preguntas de negocio, segmentar el seguimiento y diseñar pruebas operativas. No interpretar asociaciones como predicciones ni causalidad.")
        else:
            st.warning("No se encontró `y`; revisa el dataset para poder calcular hallazgos de aceptación.")

if __name__ == "__main__":
    main()
