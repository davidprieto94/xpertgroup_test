import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

# Rutas de entrada y salida
input_file = "../outputs/cleaned_data.parquet"
output_predictions = "../outputs/predictions.parquet"

try:
    # 1. Cargar Datos
    print("Cargando datos en formato Parquet...")
    data = pd.read_parquet(input_file)

    # Verificar columnas disponibles
    print("Columnas disponibles en el dataset:")
    print(data.columns)

    # 2. Validar existencia de la columna 'Price'
    if "Price" not in data.columns:
        raise KeyError("La columna 'Price' no se encontró en el dataset. Verifica el archivo.")

    # 3. Selección de Variables
    X = data.drop(columns=["Price"])  # Variables predictoras
    y = data["Price"]  # Variable objetivo

    # 4. Codificación de Variables Categóricas
    print("Codificando variables categóricas...")
    X = pd.get_dummies(X, drop_first=True)

    # 5. División del Conjunto de Datos
    print("Dividiendo datos en entrenamiento y prueba...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 6. Entrenamiento del Modelo
    print("Entrenando el modelo Random Forest...")
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # 7. Evaluación del Modelo
    print("Evaluando el modelo...")
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    print(f"MAE: {mae}")
    print(f"RMSE: {rmse}")

    # 8. Guardar Predicciones en formato Parquet
    print("Guardando predicciones en formato Parquet...")
    predictions = pd.DataFrame({"Actual": y_test, "Prediccion": y_pred})
    predictions.to_parquet(output_predictions, index=False)

    print(f"Predicciones guardadas en {output_predictions}")

except FileNotFoundError:
    print(f"El archivo {input_file} no fue encontrado. Verifica la ruta.")
except KeyError as e:
    print(f"Error en el dataset: {e}")
except Exception as e:
    print(f"Ocurrió un error inesperado: {e}")
