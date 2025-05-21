**CONFIGURACIÓN EN LOCAL**

0. Instala docker.

1. Comprueba que el servicio de docker está activo con `sudo systemctl status docker`.

2. Si no estuviese activo, arráncalo a través de `sudo systemctl enable docker`.

3. Creamos una archivo de entorno `.env` con este contenido:

`WANDB_API_KEY=tu_token_de_login_a_wandb`

4. Para subir el modelo a Wandb como un artefacto ejecutamos `project/api/subir_modelo.py`.

A continuación podemos comprobar que le modelo se ha subido a través de la interfaz de Wandb.


5. **Construimos la imágenes** y **creamos los contenedores** usando 
    
    - `sudo docker-compose up --build`

**OJO**: si ya has ejecutado previamente ese comando, quizás sea necesario que ejecutes antes `sudo docker-compose down --volumes --remove-orphans` para hacer un reset de contenedores, redes, etc.

Justo despues verás la siguiente salida:

```
Creating network "project_app-network" with driver "bridge"
Creating project_api_1 ... done
Creating project_web_1 ... done
Attaching to project_api_1, project_web_1
web_1  |  * Serving Flask app 'web_app'
web_1  |  * Debug mode: on
web_1  | WARNING: This is a development server. Do not use it in a production deployment. Use a production WSGI server instead.
web_1  |  * Running on all addresses (0.0.0.0)
web_1  |  * Running on http://127.0.0.1:8001
web_1  |  * Running on http://172.18.0.3:8001
web_1  | Press CTRL+C to quit
web_1  |  * Restarting with stat
web_1  |  * Debugger is active!
web_1  |  * Debugger PIN: 892-166-819
wandb:   3 of 3 files downloaded.  loaded...
api_1  | INFO:     Started server process [1]
api_1  | INFO:     Waiting for application startup.
api_1  | INFO:     Application startup complete.
api_1  | INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
```

Puedes hacer ctrl+click sobre `http://172.18.0.3:8001` y se abrirá el navegador con un formulario.


**CONFIGURACIÓN PARA AZURE**

He intentado el despliegue en Azure pero me ha salido un error. Se puede ver el archivo `.github/workflows/azure-deploy.yml` que he usado.

**LINK GITHUB**

https://github.com/Ziiweb/dl_mlops

**LINKS WANDB**

- Proyecto: https://wandb.ai/javiergarpe1979-upm/dl_mlops/overview
- Experimento (run): https://wandb.ai/javiergarpe1979-upm/dl_mlops/runs/wa9rdh0v?nw=nwuserjaviergarpe1979
- Report: https://wandb.ai/javiergarpe1979-upm/dl_mlops/reports/Report-CVAE-generaci-n-n-meros--VmlldzoxMjg3ODE5Nw 