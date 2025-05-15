# dl_mlops

LOCAL

0. Instala docker.

1. Comprueba que el servicio de docker está activo con `sudo systemctl status docker`.

2. Si no estuviese activo, arráncalo a través de `sudo systemctl enable docker`.

3. **Construimos** la imagen usando `sudo docker build -t mi-modelo-api .`. Como vemos, en el comando esta presente el que será el nombre de la imagen que creamos: `mi-model-api`. Al construir la imagen en realidad lo que estamos haciendo es ejecutar las lineas presentes en Dockerfile, por ejemplo la linea `RUN pip install --no-cache-dir -r requirements.txt` que instala los paquetes que hay en el archivo `requirements.txt`. 


4. Comprueba que la imagen se ha creado ejecutando `sudo docker images -a`. OJO: aunque aparezca en la lista, no quiere decir que se esté ejecutando.  Ahora el contenedor está listo para ejecutarse llamando a `run`. Si quisieses borrar la imagen creada ejecuta `sudo docker rmi -f <id de la imagen>` y `sudo docker system prune -a`.    

4. Hacemos **accesible** la imagen desde el navegador: `sudo docker run -p 8000:8000 mi-modelo-api`.






