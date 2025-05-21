
import os
assert os.path.isfile("/home/tirengarfio/Downloads/dl_mlops/project/api/artifacts/cvae_model.pth"), "Archivo modelo no encontrado"



import wandb
import os

# Inicia una corrida (no necesitas loguear métricas si solo vas a subir artifacts)
wandb.init(project="dl_mlops", job_type="upload-model")

# Crear un artifact
artifact = wandb.Artifact("cvae_model", type="model")

# Añadir archivos al artifact
artifact.add_file("/home/tirengarfio/Downloads/dl_mlops/project/api/artifacts/cvae_model_state_dict.pth")
artifact.add_file("/home/tirengarfio/Downloads/dl_mlops/project/api/artifacts/cvae_model.py")  # Ruta al archivo que define la clase CVAE
artifact.add_file("/home/tirengarfio/Downloads/dl_mlops/project/api/artifacts/config.json")  # Si guardaste esto

# Subir el artifact
wandb.log_artifact(artifact)

wandb.finish()