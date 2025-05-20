# from pathlib import Path
# import wandb

# run = wandb.init(project="dl_mlops")

# artifact_filepath = Path("./project/api/cvae_model.pth")

# if not artifact_filepath.exists():
#     raise FileNotFoundError(f"No se encontró el archivo: {artifact_filepath.resolve()}")

# artifact = wandb.Artifact(
#     name="cvae_model",
#     type="model",
#     description="Modelo CVAE entrenadoo",
# )

# artifact.add_file(str(artifact_filepath))
# run.log_artifact(artifact)

# run.link_artifact(
#     artifact=artifact,
#     target_path="javiergarpe1979-upm/dl_mlops"
# )

# run.finish()

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