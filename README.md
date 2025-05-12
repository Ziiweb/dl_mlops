# dl_mlops

LOCAL

#### Construir la imagen

Para construir la imagen,

0. Instala docker.
1. Comprueba que el servicio de docker está activo a través de `sudo systemctl status docker`.
2. Si no estuviese activo, arráncalo a través de `sudo systemctl enable docker`.
3. Arrancar el la imagen usando `docker build -t mi-modelo-api .` 
NOTA sobre el punto 3: si da un error, prueba a poner `sudo` delante. 


Ahora podemos arrancar el contenedor ahora usando `sudo docker run -p 8000:8000 mi-modelo-api`.




