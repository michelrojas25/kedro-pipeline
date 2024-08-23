import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import RobustScaler
from typing import Dict, Tuple
import logging

def load_datasets(order_items: pd.DataFrame, products: pd.DataFrame, payments: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    return {'order_items': order_items, 'products': products, 'payments': payments}

def validate_data(datasets: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    order_items = datasets['order_items']
    products = datasets['products']
    payments = datasets['payments']
    
    # Ejemplo de validaciones con advertencias en lugar de excepciones
    if order_items.isnull().values.any():
        logging.warning("order_items contiene valores nulos")
    if products.isnull().values.any():
        logging.warning("products contiene valores nulos")
    if payments.isnull().values.any():
        logging.warning("payments contiene valores nulos")
    
    return datasets

def enrich_data(datasets: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    order_items = datasets['order_items']
    products = datasets['products']
    payments = datasets['payments']

    # Ejemplo de enriquecimiento de datos
    products['product_density'] = products['product_weight_g'] / (products['product_length_cm'] * products['product_height_cm'] * products['product_width_cm'])

    datasets['products'] = products
    return datasets

def preprocess_data(datasets: Dict[str, pd.DataFrame], threshold: int) -> Tuple[pd.DataFrame, list]:
    order_items = datasets['order_items']
    products = datasets['products']
    payments = datasets['payments']
    
    products_clean = products.copy()
    products_clean['product_category_name'] = products_clean['product_category_name'].fillna(products_clean['product_category_name'].mode()[0])
    products_clean['product_name_lenght'] = products_clean['product_name_lenght'].fillna(products_clean['product_name_lenght'].mean())
    products_clean['product_description_lenght'] = products_clean['product_description_lenght'].fillna(products_clean['product_description_lenght'].mean())
    products_clean['product_photos_qty'] = products_clean['product_photos_qty'].fillna(products_clean['product_photos_qty'].mean())
    products_clean['product_weight_g'] = products_clean['product_weight_g'].fillna(products_clean['product_weight_g'].mean())
    products_clean['product_length_cm'] = products_clean['product_length_cm'].fillna(products_clean['product_length_cm'].mean())
    products_clean['product_height_cm'] = products_clean['product_height_cm'].fillna(products_clean['product_height_cm'].mean())
    products_clean['product_width_cm'] = products_clean['product_width_cm'].fillna(products_clean['product_width_cm'].mean())

    data = order_items.merge(products_clean, on='product_id').merge(payments, on='order_id')

    data['price_per_weight'] = data['price'] / data['product_weight_g']
    data['product_volume'] = data['product_length_cm'] * data['product_height_cm'] * data['product_width_cm']

    data['payment_value'] = data['payment_value'].astype(float)

    features = ['price', 'freight_value', 'product_weight_g', 'product_name_lenght', 'product_description_lenght',
                'product_photos_qty', 'product_length_cm', 'product_height_cm', 'product_width_cm', 'price_per_weight', 'product_volume', 'product_density']
    
    def adjust_outliers(df, column, threshold):
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR
        df[column] = np.where(df[column] > upper_bound, upper_bound,
                              np.where(df[column] < lower_bound, lower_bound, df[column]))
        return df

    for col in features:
        data = adjust_outliers(data, col, threshold)
    
    data['log_payment_value'] = np.log1p(data['payment_value'])

    scaler = RobustScaler()
    data[features] = scaler.fit_transform(data[features])

    return data, features

def generar_graficos(data: pd.DataFrame, features: list) -> Dict[str, plt.Figure]:
    graficos = {}

    # Distribución de la variable objetivo
    fig1, ax1 = plt.subplots()
    sns.histplot(data['payment_value'], kde=True, ax=ax1)
    ax1.set_title('Distribución de payment_value')
    graficos["dist_payment_value"] = fig1

    # Matriz de correlación
    fig2, ax2 = plt.subplots()
    corr_matrix = data[features + ['payment_value']].corr()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', ax=ax2)
    ax2.set_title('Matriz de correlación')
    graficos["correlation_matrix"] = fig2

    # Detectar valores atípicos usando boxplots
    fig3, axes = plt.subplots(3, 4, figsize=(15, 10))
    for i, col in enumerate(features):
        sns.boxplot(x=data[col], ax=axes[i // 4, i % 4])
        axes[i // 4, i % 4].set_title(col)
    fig3.tight_layout()
    graficos["boxplots"] = fig3

    return graficos

