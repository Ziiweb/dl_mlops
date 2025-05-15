# dl_mlops

LOCAL

#### Construir la imagen

Para construir la imagen,

0. Instala docker.

1. Comprueba que el servicio de docker está activo con `sudo systemctl status docker`.

2. Si no estuviese activo, arráncalo a través de `sudo systemctl enable docker`.

3. **Construimos** la imagen usando `sudo docker build -t mi-modelo-api .` a partir del archivo Dockerfile. Puedes verla ejecutando `sudo docker images`. Ahora el contenedor está listo para ejecutarse con `run`.

4. Hacemos **accesible** la imagen desde el navegador: `sudo docker run -p 8000:8000 mi-modelo-api`



A continuación, será necesario installar lo siguiente en un **entorno virtual** a ser posible:


uvicorn: `pip install uvicorn`

Ahora podemos arrancar el contenedor ahora usando `sudo docker run -p 8000:8000 mi-modelo-api`.




