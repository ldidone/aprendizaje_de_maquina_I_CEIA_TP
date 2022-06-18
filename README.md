# Aprendizaje de maquina I CEIA - TP Integrador: Acoustic Extinguisher Fire Dataset.

Repositorio de código para el TP Integrador del curso de aprendizaje de máquina 1, de la CEIA - FIUBA sobre clasificación de extinción de fuego.

## Indice de archivos:

- [0. Exploratory Data Analyisis.ipynb](https://github.com/ldidone/aprendizaje_de_maquina_I_CEIA_TP/blob/main/0.%20Exploratory%20Data%20Analyisis.ipynb): Aquí encontrará un breve análisis exploratorio de los datos con el objetivo de comprender el dataset y la variable que se intenta predecir.
- [1. Experimentation - Baseline.ipynb](https://github.com/ldidone/aprendizaje_de_maquina_I_CEIA_TP/blob/main/1.%20Experimentation%20-%20Baseline.ipynb): Aquí encontrará el flujo de pre-procesamiento de los datos, así como también el planteo de un modelo base (baseline), junto a su entrenamiento y evaluación.
- [2. Experimentation - Full.ipynb](https://github.com/ldidone/aprendizaje_de_maquina_I_CEIA_TP/blob/main/2.%20Experimentation%20%20-%20Full.ipynb): Aquí encontrará el flujo de pre-procesamiento de los datos, y la experimentación con diferentes modelos, adoptando dos enfoques: selección manual de modelos y AutoML.
- [3. Random Forest Experimentation.ipynb](https://github.com/ldidone/aprendizaje_de_maquina_I_CEIA_TP/blob/main/3.%20Random%20Forest%20Experimentation.ipynb): Aquí encontrará el flujo de pre-procesamiento de los datos, y la búsqueda hiper-paramétrica utilizando GridSearchCV sobre el modelo de Random Forest con el objetivo modelo que mejor clasifique los datos.

*Los modelos entrenados se encuentran disponibles haciendo click [aquí](https://github.com/ldidone/aprendizaje_de_maquina_I_CEIA_TP/tree/main/models).*

----------------------------------------------------------------------------------------------
# API pública:

- Podrá interactuar con el modelo obtenido utilizando la siguiente API: https://acousticextinguisherfireapi.herokuapp.com/docs

**Curl de ejemplo:**

    curl -X 'POST' \
      'https://acousticextinguisherfireapi.herokuapp.com/predict' \
      -H 'accept: application/json' \
      -H 'Content-Type: application/json' \
      -d '{
      "size": 1,
      "fuel": "gasoline",
      "distance": 10,
      "desibel": 96,
      "airflow": 2.6,
      "frequency": 70
    }'
	
# Aplicación WEB:

- Podrá interactuar con el modelo obtenido haciendo uso de la aplicación web haciendo click [aquí](https://ldidone.github.io/).

----------------------------------------------------------------------------------------------
> Alumnos:
> - Emmanuel Cardozo
> - Carlos Pallares
> - Lucas Didone

-------------------------------------------------------------------------------------------
> ### Citation
> Yavuz Selim TASPINAR, Murat KOKLU and Mustafa ALTIN CV:https://www.muratkoklu.com/en/publications/ DATASET: https://www.muratkoklu.com/datasets/ Citation Request : 1: KOKLU M., TASPINAR Y.S., (2021). Determining the Extinguishing Status of Fuel Flames With Sound Wave by Machine Learning Methods. IEEE Access, 9, pp.86207-86216, Doi: 10.1109/ACCESS.2021.3088612 Link: https://ieeexplore.ieee.org/document/9452168 (Open Access) https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9452168

> 2: TASPINAR Y.S., KOKLU M., ALTIN M., (2021). Classification of Flame Extinction Based on Acoustic Oscillations using Artificial Intelligence Methods. Case Studies in Thermal Engineering, 28, 101561, Doi: 10.1016/j.csite.2021.101561 Link: https://www.sciencedirect.com/science/article/pii/S2214157X21007243 (Open Access) https://www.sciencedirect.com/sdfe/reader/pii/S2214157X21007243/pdf

> 3: TASPINAR Y.S., KOKLU M., ALTIN M., (2022). Acoustic-Driven Airflow Flame Extinguishing System Design and Analysis of Capabilities of Low Frequency in Different Fuels. Fire Technology, Doi: 10.1007/s10694-021-01208-9 Link: https://link.springer.com/content/pdf/10.1007/s10694-021-01208-9.pdf
