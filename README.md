# 🧠 Sprint 16 – Predicción de Edad con Visión Artificial (CNN + ResNet50)

## 📌 Descripción del Proyecto

En este proyecto se implementa una **red neuronal convolucional** basada en la arquitectura **ResNet50** para predecir la edad de personas a partir de sus fotografías. La iniciativa forma parte de una investigación sobre visión artificial aplicada a perfiles humanos, utilizando técnicas de **transfer learning** sobre el conjunto de datos de Kaggle/ImageNet.

El objetivo principal es alcanzar un **Error Absoluto Medio (EAM ≤ 8)** en el conjunto de prueba.

## 🎯 Objetivos del Proyecto

- Cargar y preprocesar imágenes con `ImageDataGenerator`.
- Aplicar **transfer learning** con ResNet50 (`weights='imagenet'`).
- Entrenar un modelo de regresión con `Keras` y medir la métrica **MAE**.
- Evaluar el desempeño del modelo en datos reales de validación.

## 📁 Estructura del Dataset

- Carpeta: `/datasets/train/final_files/`
- CSV: `/datasets/train/labels.csv`
- Cada archivo de imagen tiene un nombre y una edad real (`real_age`) asociada.

## 🧰 Funcionalidades del Proyecto

### 🧩 Funciones Clave

- `load_data(path, subset)`: carga imágenes desde el directorio y etiquetas desde `labels.csv`.
- `create_model(input_shape)`: define la red neuronal con ResNet50 como backbone.
- `train_model(model, train_data, test_data)`: entrena el modelo utilizando función de pérdida `mse` y métrica `mae`.

### 🧪 Entrenamiento

- `batch_size`: 16  
- `epochs`: 3  
- `optimizer`: Adam con `lr=0.0005`  
- Se mide el `MAE` en el conjunto de prueba (`subset='testing'`).

## 🧠 Herramientas utilizadas

- Python  
- TensorFlow / Keras  
- Pandas  
- ImageDataGenerator  
- Preentrenamiento con `imagenet` (ResNet50)  

---

📌 Proyecto desarrollado como parte del Sprint 16 del programa de Ciencia de Datos en **TripleTen**.
