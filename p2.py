import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

vol = pd.read_csv("20250127_datosvolumen.csv", sep=";", encoding="latin-1")
vel = pd.read_csv("20250127_datosvelocidad.csv", sep=";", encoding="latin-1")
est = pd.read_csv("estaciones.csv", sep=";", encoding="latin-1")

# Busqueda dinamica de columnas de ligeros y pesados
ligeros = [c for c in vol.columns if "ligeros" in c]
pesados = [c for c in vol.columns if "pesados" in c]

# Añadir columnas de totales
vol["Total_vehiculos"] = vol[ligeros + pesados].sum(axis=1)
vol["Total_ligeros"] = vol[ligeros].sum(axis=1)
vol["Total_pesados"] = vol[pesados].sum(axis=1)


# Eliminar columnas originales
vol = vol.drop(columns=ligeros + pesados)

# Eliminar la columna "Unnamed: 13" de velocidad, que es una columna vacía
vel = vel.drop(columns=["Unnamed: 13"])

# Eliminar el segundo valor del rango Horario
vel["Hora"] = vel["Hora"].str.split(" - ").str[0]


# Modificar la columna "Hora" restando 1 hora 
vel["h"] = vel["Hora"].str.split(":").str[0].astype(int) - 1
vel["m"] = vel["Hora"].str.split(":").str[1]
vel["s"] = vel["Hora"].str.split(":").str[2]

# Modificar la columna "Hora" restando 1 hora para cuadrar el formato de Hora
vol["h"] = vol["Hora"].str.split(":").str[0].astype(int) - 1
vol["m"] = vol["Hora"].str.split(":").str[1]

# Reescribir la columna "Hora" con el nuevo formato
vol["Hora"] = vol["h"].astype(str).str.zfill(2) + ":" + vol["m"]
vel["Hora"] = vel["h"].astype(str).str.zfill(2) + ":" + vel["m"].astype(str).str.zfill(2) + ":" + vel["s"]

# Eliminar columnas auxiliares
vol = vol.drop(columns=["h", "m"])
vel = vel.drop(columns=["h", "m", "s"])

# Convertir la fecha a formato datetime con la Hora suprimiendo los espacios en blanco de la columna Fecha

vol["Fecha"] = vol["Fecha"].astype(str).str.strip()
vol["Hora"] = vol["Hora"].astype(str).str.strip()

vol["Fecha"] = pd.to_datetime(vol["Fecha"] + " " + vol["Hora"], format="%d/%m/%Y %H:%M")
vel["Fecha"] = pd.to_datetime(vel["Fecha"] + " " + vel["Hora"], format="%Y-%m-%d %H:%M:%S")

vol = vol.drop(columns=["Hora"])
vel = vel.drop(columns=["Hora"])


# Reemplazar sistema por código ETD en velocidad y que cuadre con volumen
vel["Estacion"] = vel["ETD"].str.extract(r"\]\s*(\d+)-ETD")

vol["Estacion"] = pd.to_numeric(vol["Estacion"], errors="coerce")
vel["Estacion"] = pd.to_numeric(vel["Estacion"], errors="coerce")


# Unir los dataframes de volumen y velocidad
df = vol.merge(
    vel,
    on=["Fecha", "Estacion"],
    how="inner"
)

df = df.merge(
    est,
    left_on="Estacion",
    right_on="ETD code",
    how="left"
)

# Crear una nueva columna "Hora_num" con la hora en formato numérico a partir de la columna "Fecha"

df["Hora_num"] = pd.to_datetime(df["Fecha"], errors="coerce").dt.hour

features = [
    "Total_vehiculos",
    "Total_ligeros",
    "Total_pesados",
    "Velocidad media (km/h)",
    "Hora_num"
]

# Limpiar y convertir las columnas numéricas a formato numérico, reemplazando los valores vacíos por NaN
for c in ["Total_vehiculos", "Total_ligeros", "Total_pesados", "Velocidad media (km/h)"]:
    df[c] = (
        df[c]
        .astype(str)
        .str.strip()
        .replace("", pd.NA)
        .replace(" ", pd.NA)
        .str.replace(",", ".", regex=False)
    )
    df[c] = pd.to_numeric(df[c], errors="coerce")

X = df[features].dropna()

# Escalar las características para KMeans
X_scaled = StandardScaler().fit_transform(X)


# Método del codo para determinar el número óptimo de clusters
inertia = []

for k in range(2, 10):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)

plt.plot(range(2, 10), inertia, marker="o")
plt.xlabel("Número de clusters")
plt.ylabel("Inercia")
plt.title("Método del codo")
plt.show()

# Aplicar KMeans con el número óptimo de clusters 
kmeans = KMeans(n_clusters=4, random_state=42)
df["cluster"] = kmeans.fit_predict(X_scaled)

df.groupby("cluster")[features].mean()

# PCA para visualizar los clusters


pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.scatter(X_pca[:,0], X_pca[:,1], c=df["cluster"])
plt.title("Clusters visualizados con PCA")
plt.show()
