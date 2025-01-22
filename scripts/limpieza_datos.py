import pandas as pd

# Rutas de entrada y salida
input_file = "../sources/clothes_price_prediction_dat.csv"
output_file = "../outputs/cleaned_data.parquet"

# Cargar los datos
data = pd.read_csv(input_file)

# Identificar valores nulos y duplicados
print("Valores nulos antes de la limpieza:")
print(data.isnull().sum())
print("\nDuplicados antes de la limpieza:")
print(data.duplicated().sum())

# Limpieza de datos
data_cleaned = data.dropna()
data_cleaned = data_cleaned.drop_duplicates()

# Guardar datos limpios en formato Parquet
data_cleaned.to_parquet(output_file, index=False)

print("\nLimpieza completada. Datos guardados en formato Parquet:", output_file)