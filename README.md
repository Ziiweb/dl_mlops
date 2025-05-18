# dl_mlops

LOCAL

0. Instala docker.

1. Comprueba que el servicio de docker está activo con `sudo systemctl status docker`.

2. Si no estuviese activo, arráncalo a través de `sudo systemctl enable docker`.

3. Creamos una archivo de entorno `.env`.

4. **Construimos la imágenes** y **creamos los contenedores** usando 
    
    - `sudo docker-compose up --build`

Como vemos, en el comando pasamos como parametro el nombre de la imágenes que creamos: `mi-model-api` y `web_app`. Al construir la imágenes, en realidad lo que hacemos es ejecutar las lineas presentes en los archivos Dockerfile respectivos, por ejemplo la linea `RUN pip install --no-cache-dir -r requirements.txt` que instala los paquetes indicados en los archivos `requirements.txt`. 

5. Comprueba que las imágenes se ha creado ejecutando `sudo docker images -a`. OJO: aunque aparezcan en la lista, no quiere decir que estén <em>corriendo</em>. Ahora los contenedores están listos para ejecutarse llamando a `run`. Si quisieses borrar alguna de las imágenes creadas ejecuta:

    - `sudo docker-compose down --volumes --remove-orphans`
    - `sudo docker system prune -af`

5. Hacemos **accesibles** las imágenes desde el navegador con: 

    - `sudo docker run -p 8000:8000 mi-modelo-api`
    - `sudo docker run -p 8001:8000 web_app`

Como vemos, hacemos uso de diferentes puertos para cada contenedor: 8000 y 8001.

6. Comprobamos que los contenedores estan corriendo a través de `sudo docker ps`.



### SUBIR MODELO A WANDB

Para subir el modelo a Wandb como un artefacto vamos a ejecutar `project/api/subir_modelo.py`.

A continuación podemos comprobar que le modelo se ha subido a través de la interfaz de Wandb.


