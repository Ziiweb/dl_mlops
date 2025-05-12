# Imagen base con Python
FROM python:3.10

# Crear directorio en el contenedor
WORKDIR /app

# Copiar dependencias
COPY requirements.txt .

# Instalar dependencias
RUN pip install --no-cache-dir -r requirements.txt

# Copiar el código
COPY . .

# Exponer el puerto de FastAPI
EXPOSE 8000

# Comando para arrancar la API.
# "app" se refiere a la variable "app", podría ser una instancia de la clase FastAPI.
CMD ["uvicorn", "nombre_del_script:app", "--host", "0.0.0.0", "--port", "8000"]