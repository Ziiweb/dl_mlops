from flask import Flask, render_template, request
import requests
import os

app = Flask(__name__)

# Configuración de la URL base de la API
# Se puede configurar a través de variables de entorno (Docker) o por defecto localhost:8000 (se ejecuta sin Docker)
API_HOST = os.environ.get("API_HOST", "localhost")
API_PORT = os.environ.get("API_PORT", "8000")
API_URL = f"http://{API_HOST}:{API_PORT}/predict"

# Puerto de la aplicación web
WEB_APP_PORT = os.environ.get("WEB_APP_PORT", "8001")


@app.route("/", methods=["GET"])
def index():
    # Renderiza un formulario HTML
    return render_template("index.html")

@app.route("/predict_temperature", methods=["POST"])
def predict_price():
    # Recogemos los datos del formulario
    feature_AA = request.form.get("feature_AA")
    feature_AB = request.form.get("feature_AB")
    feature_BA = request.form.get("feature_BA")
    feature_BB = request.form.get("feature_BB")
    feature_CA = request.form.get("feature_CA")
    feature_CB = request.form.get("feature_CB")

    # Construimos la carga en JSON para la API
    payload = {
        "feature_AA": float(feature_AA),
        "feature_AB": float(feature_AB),
        "feature_BA": float(feature_BA),
        "feature_BB": float(feature_BB),
        "feature_CA": float(feature_CA),
        "feature_CB": float(feature_CB),
    }

    # Hacemos la petición POST a la API
    response = requests.post(API_URL, json=payload)
    
    if response.status_code == 200:
        data = response.json()
        predicted_price = data["predicted_price"]
        return f"El precio estimado de la casa es: {predicted_price:.2f}"
    else:
        return "Error en la API. No se pudo obtener la predicción."

if __name__ == "__main__":
    # Ejecutar la aplicación web en el puerto especificado
    app.run(host="0.0.0.0", port=WEB_APP_PORT, debug=True)