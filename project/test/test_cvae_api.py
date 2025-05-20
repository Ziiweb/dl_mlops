#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
test_cvae_api.py: Tests para la API de generación de imágenes CVAE.

⚠️ Requiere que el servidor FastAPI esté corriendo en http://api:8000,
por ejemplo ejecutando: uvicorn inferencia_lstm_fastapi:app --reload
"""

import requests
import time
import os

API_HOST = os.environ.get("API_HOST", "localhost")
API_PORT = int(os.environ.get("API_PORT", 8000))
API_URL = f"http://{API_HOST}:{API_PORT}"

def check_server(api_url=API_URL, retries=3, delay=5, timeout=2):
    print(f"🔍 Verificando disponibilidad del servidor en {api_url}...")
    for attempt in range(1, retries + 1):
        try:
            response = requests.get(f"{api_url}/docs", timeout=timeout)
            if response.status_code == 200:
                print("✅ El servidor está disponible.")
                return
            else:
                print(f"⚠️ Respuesta inesperada (status {response.status_code}), intento {attempt}/{retries}")
        except requests.RequestException as err:
            print(f"⚠️ Error al conectar: {err}, intento {attempt}/{retries}")

        if attempt < retries:
            time.sleep(delay)

    raise RuntimeError(f"🚨 El servidor no respondió correctamente tras {retries} intentos.")

def test_generate_image_success():
    """Test para petición válida, espera matriz 28x28 en 'image'."""
    for label in range(10):
        payload = {"label": label}
        response = requests.post(f"{API_URL}/predict", json=payload)
        print(f"Test generate_image_success para label={label}:")
        assert response.status_code == 200, f"Status code: {response.status_code}"
        data = response.json()
        assert "image" in data, "Falta campo 'image' en respuesta"
        image = data["image"]
        assert isinstance(image, list), "'image' no es una lista"
        assert len(image) == 28, f"'image' debe tener 28 filas, tiene {len(image)}"
        assert all(len(row) == 28 for row in image), "Cada fila debe tener 28 columnas"
        print(f"✅ Test para label={label} PASADO.")

def test_generate_image_invalid_label():
    """Test para etiquetas inválidas: fuera de rango o tipo incorrecto."""
    invalid_labels = [-1, 10, 100, "a", None]
    for label in invalid_labels:
        payload = {"label": label}
        response = requests.post(f"{API_URL}/predict", json=payload)
        print(f"Test generate_image_invalid_label con label={label}:")
        assert response.status_code >= 400, f"Esperado error, status code: {response.status_code}"
        print(f"✅ Test para label inválido={label} PASADO.")

# def test_generate_image_missing_field():
#     """Test para petición sin campo label."""
#     payload = {}
#     response = requests.post(f"{API_URL}/predict", json=payload)
#     print(f"Test generate_image_missing_field:")
#     assert response.status_code == 422, f"Esperado error 422, status code: {response.status_code}"
#     print(f"✅ Test para campo faltante PASADO.")

if __name__ == "__main__":
    print("🧪 Verificando que el servidor esté levantado...")
    check_server()
    print("✅ Servidor activo. Ejecutando tests...\n")
    test_generate_image_success()
    test_generate_image_invalid_label()
    #test_generate_image_missing_field()
    print("🎉 Todos los tests han sido ejecutados exitosamente.")
