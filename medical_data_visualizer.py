import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# 1: Importar y cargar los datos
df = pd.read_csv('medical_examination.csv')

# 2: Añadir la columna de sobrepeso (calcular IMC)
df['overweight'] = (df['weight'] / ((df['height'] / 100) ** 2)) > 25
df['overweight'] = df['overweight'].astype(int)  # Convertir de booleano a 0 y 1

# 3: Normalizar los datos de colesterol y glucosa
df['cholesterol'] = df['cholesterol'].apply(lambda x: 0 if x == 1 else 1)
df['gluc'] = df['gluc'].apply(lambda x: 0 if x == 1 else 1)

# 4: Función para graficar el gráfico categórico
def draw_cat_plot():
    # Reformatear los datos con pd.melt
    df_cat = pd.melt(df, id_vars=["cardio"], value_vars=["cholesterol", "gluc", "smoke", "alco", "active", "overweight"])

    # Agrupar y contar los valores
    df_cat = df_cat.groupby(['cardio', 'variable', 'value'], as_index=False).size().reset_index()
    df_cat = df_cat.rename(columns={'size': 'count'})  # Renombrar la columna 'size' a 'count'

    # Modificar la leyenda para que sea más comprensible
    df_cat['value'] = df_cat['value'].replace({
        0: 'Healthy',
        1: 'Not Healthy',
    })

    # Cambiar los nombres de las tablas para hacerlos más comprensibles
    df_cat['cardio'] = df_cat['cardio'].replace({
        0: 'without cardiovascular disease',  # Cardio 0
        1: 'with cardiovascular disease'  # Cardio 1
    })

    # Graficar con sns.catplot
    g = sns.catplot(data=df_cat, x='variable', hue='value', col='cardio', kind='bar', height=5, aspect=1.5, y='count')

    # Mejorar la visualización
    g.set_axis_labels("variable", "total")
    g.set_titles("{col_name}")
    g.set_xticklabels(rotation=45)  # Asegurarse de que las etiquetas de las variables no se sobrepongan

    # Añadir título al gráfico completo
    g.figure.suptitle("Categorical Plot of Cardiovascular Risk Factors", fontsize=16, y=1.05)

    fig = g.figure
    fig.savefig('catplot.png')

    return fig


# Función para graficar el mapa de calor con máscara
def draw_heat_map():
    # Limpiar los datos según las condiciones especificadas
    df_heat = df[(df['ap_lo'] <= df['ap_hi']) & 
                 (df['height'] >= df['height'].quantile(0.025)) & 
                 (df['height'] <= df['height'].quantile(0.975)) & 
                 (df['weight'] >= df['weight'].quantile(0.025)) & 
                 (df['weight'] <= df['weight'].quantile(0.975))]

    # Crear una copia para evitar modificar el DataFrame original
    df_heat = df_heat.copy()

    # Calcular la matriz de correlación
    corr = df_heat.corr()

    # Generar la máscara para ocultar la diagonal superior
    mask = np.triu(np.ones_like(corr, dtype=bool))

    # Crear la figura para el gráfico
    fig, ax = plt.subplots(figsize=(10, 8))

    # Crear el mapa de calor con valores redondeados
    sns.heatmap(corr.round(1), mask=mask, annot=True, fmt='.1f', cmap='coolwarm', 
                cbar_kws={'shrink': 0.5}, vmin=-0.1, vmax=0.25, square=True, ax=ax)

    # Añadir título al gráfico
    fig.suptitle("Correlation Heatmap of Medical Data", fontsize=16, y=0.98)

    # Guardar la figura como 'heatmap.png'
    fig.savefig('heatmap.png')

    return fig




