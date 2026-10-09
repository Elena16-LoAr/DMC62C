# BankMarketing — Análisis Exploratorio de Datos (EDA)

Aplicación interactiva en Streamlit para explorar el dataset de campañas de marketing bancario. El proyecto se centra en describir la calidad, distribución y relaciones entre variables para apoyar la toma de decisiones; **no desarrolla modelos predictivos**.

## Autora
- **Elena Loayza Arteaga**
- Especialización: **Python for Analytics**
- Año: **2026**

## Funcionalidades
- Presentación del proyecto y contexto del caso.
- Carga de CSV con validación del separador, vista previa y dimensiones.
- Diez áreas de EDA: información general, clasificación de variables, estadística descriptiva, faltantes, histogramas, variables categóricas, análisis numérico vs. objetivo, análisis categórico vs. objetivo, análisis dinámico y hallazgos.
- Widgets Streamlit: sidebar, tabs, columns, file uploader, selectbox, multiselect, slider y checkbox.
- Clase `DataAnalyzer` para encapsular clasificación de variables, estadísticas descriptivas y resumen de valores faltantes.
- Cinco conclusiones redactadas con enfoque descriptivo y de negocio.

## Dataset
El archivo `BankMarketing.csv` debe colocarse en la raíz del repositorio. La aplicación también permite cargar el archivo desde el menú lateral. Se contemplan separadores `;`, `,` y tabulación. El dataset contiene datos de clientes, características de contacto, campañas anteriores, indicadores económicos y el resultado `y` (`yes`/`no`).

> Usa los resultados como análisis descriptivo. Las asociaciones observadas no prueban causalidad ni predicen resultados futuros.

## Ejecutar localmente
Requiere Python 3.10 o posterior recomendado.

```bash
# 1. Crear y activar un entorno virtual (opcional pero recomendado)
python -m venv .venv

# Windows PowerShell
.\.venv\Scripts\Activate.ps1

# macOS / Linux
source .venv/bin/activate

# 2. Instalar dependencias
pip install -r requirements.txt

# 3. Iniciar la aplicación
streamlit run app.py
```

La aplicación abrirá una dirección local en el navegador. En el panel lateral, carga `BankMarketing.csv` para habilitar los análisis.

## Publicar en Streamlit Community Cloud
1. Sube `app.py`, `requirements.txt`, `README.md` y `BankMarketing.csv` a un repositorio de GitHub.
2. Entra a https://share.streamlit.io/ e inicia sesión con GitHub.
3. Selecciona **Create app**, el repositorio, la rama y `app.py` como archivo principal.
4. Despliega y prueba la URL pública desde una ventana privada.
5. Añade aquí los enlaces reales cuando estén publicados:
   - GitHub: `PENDIENTE_DE_PUBLICAR`
   - Aplicación: `PENDIENTE_DE_DESPLEGAR`

## Capturas de pantalla
Añade capturas reales de la aplicación ya ejecutada en una carpeta `screenshots/` y enlázalas aquí. No se incluyen capturas ficticias.

```markdown
![Home](screenshots/home.png)
![Carga del dataset](screenshots/carga.png)
![Análisis exploratorio](screenshots/eda.png)
```

## Tecnologías
Python · Streamlit · Pandas · NumPy · Matplotlib · Seaborn · GitHub

## Estructura del repositorio
```text
BankMarketing/
├── app.py
├── requirements.txt
├── README.md
└── BankMarketing.csv
```
