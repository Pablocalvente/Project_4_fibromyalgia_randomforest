import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.exceptions import DataConversionWarning
import warnings

# Suprime las advertencias de DataConversionWarning
warnings.filterwarnings("ignore", category=DataConversionWarning)

# Carga los datos desde el archivo CSV (con las columnas correctas)
file_path = "F:/Utilidades/Hello-Python-main/Proyecto_4_fibromialgia/Fibromyalgia_patients.csv"
data = pd.read_csv(file_path)

# Separar las características (X) y las etiquetas (y)
X = data.drop(columns=["Fibromyalgia"])
y = data["Fibromyalgia"]

# Escalar los datos en modo silencioso
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Inicializar y entrenar el modelo de regresión logística (igual que antes)
model = LogisticRegression(max_iter=1000)
model.fit(X_scaled, y)

# Función para predecir la probabilidad de tener fibromialgia
def predecir_probabilidad_fibromialgia():
    # Solicitar inputs al usuario
    print("Por favor, ingrese la información del nuevo paciente:")
    
    # Definir una función para solicitar la entrada del usuario y manejar errores
    def solicitar_entrada(mensaje, valid_values=None):
        while True:
            entrada = input(mensaje)
            if not entrada:
                print("Error: Este campo es obligatorio. Por favor, ingréselo.")
                continue
            if valid_values is not None:
                if entrada not in valid_values:
                    print(f"Error: Por favor, ingrese un valor válido ({', '.join(valid_values)}).")
                    continue
            try:
                if mensaje.lower() == "edad:":
                    entrada = float(entrada)
                else:
                    entrada = int(entrada)
                return entrada
            except ValueError:
                print("Error: Ingrese un valor numérico válido")

    # Solicitar información de cada variable
    age = solicitar_entrada("Edad: ")
    gender = solicitar_entrada("Género (0 para masculino, 1 para femenino): ", valid_values=["0", "1"])
    family_history = solicitar_entrada("Historial familiar (0 para no, 1 para sí): ", valid_values=["0", "1"])
    emotional_stress = solicitar_entrada("Estrés emocional (0 o 1): ", valid_values=["0", "1"])
    physical_trauma = solicitar_entrada("Trauma físico (0 o 1): ", valid_values=["0", "1"])
    previous_infection = solicitar_entrada("Infección previa (0 o 1): ", valid_values=["0", "1"])
    depression = solicitar_entrada("Depresión (0 o 1): ", valid_values=["0", "1"])
    rheumatoid_arthritis = solicitar_entrada("Artritis reumatoide (0 o 1): ", valid_values=["0", "1"])
    chronic_fatigue_syndrome = solicitar_entrada("Síndrome de fatiga crónica (0 o 1): ", valid_values=["0", "1"])
    arthritis = solicitar_entrada("Artritis (0 o 1): ", valid_values=["0", "1"])
    migraine = solicitar_entrada("Migraña (0 o 1): ", valid_values=["0", "1"])
    generalized_pain = solicitar_entrada("Dolor generalizado (0 o 1): ", valid_values=["0", "1"])
    anxiety = solicitar_entrada("Ansiedad (0 o 1): ", valid_values=["0", "1"])
    fatigue = solicitar_entrada("Fatiga (0 o 1): ", valid_values=["0", "1"])

    # Agregar "Extra Feature" con un valor predeterminado de 0
    extra_feature = 0

    # Escala las características del nuevo paciente en modo silencioso
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        nuevo_paciente_scaled = scaler.transform([[age, gender, family_history, emotional_stress, physical_trauma,
                                                  previous_infection, depression, rheumatoid_arthritis,
                                                  chronic_fatigue_syndrome, arthritis, migraine,
                                                  generalized_pain, anxiety, fatigue, extra_feature]])

    # Realiza la predicción de probabilidad utilizando el modelo
    probabilidad = model.predict_proba(nuevo_paciente_scaled)[0][1] * 100

    # Muestra el resultado al usuario
    print(f"El paciente tiene un {probabilidad:.2f}% de probabilidad de tener fibromialgia.")

# Llama a la función para predecir la probabilidad
predecir_probabilidad_fibromialgia()
