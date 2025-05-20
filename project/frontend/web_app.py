from flask import Flask, render_template, request
import requests
import os
import base64
import io
import matplotlib.pyplot as plt

app = Flask(__name__)

# Configuración de la URL base de la API
# Se puede configurar a través de variables de entorno (Docker) o por defecto localhost:8000 (se ejecuta sin Docker)
API_HOST = os.environ.get("API_HOST", "api")
API_PORT = os.environ.get("API_PORT", "8000")
API_URL = f"http://{API_HOST}:{API_PORT}/predict"

# Puerto de la aplicación web
WEB_APP_PORT = int(os.environ.get("WEB_APP_PORT", "8001"))

@app.route("/", methods=["GET"])
def index():
    # Renderiza un formulario HTML
    return render_template("index.html")

import base64
import io
import matplotlib.pyplot as plt
import numpy as np
from markupsafe import Markup

@app.route("/predict_temperature", methods=["POST"])
def predict_temperature():
    numero = request.form.get("number_to_generate")

    payload = {
        "label": int(numero),
    }

    try:
        response = requests.post(API_URL, json=payload)
        response.raise_for_status()
        data = response.json()

        if "image" in data:
            image_array = np.array(data["image"])

            # Convertir matriz a imagen PNG en base64
            fig, ax = plt.subplots()
            ax.imshow(image_array, cmap="gray")
            ax.axis('off')
            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            image_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
            plt.close(fig)

            img_html = f'<img src="data:image/png;base64,{image_base64}" alt="Imagen generada">'
            return f"<h2>Imagen generada para el número {numero}</h2>{Markup(img_html)}"
        else:
            return "La respuesta de la API no contiene una imagen generada."

    except requests.exceptions.RequestException as e:
        return f"Error en la conexión con la API: {e}"



if __name__ == "__main__":
    # Ejecutar la aplicación web en el puerto especificado
    app.run(host="0.0.0.0", port=WEB_APP_PORT, debug=True)
