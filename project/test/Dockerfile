FROM python:3.10


# Crear directorio en el contenedor
WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python", "test_cvae_api.py"]