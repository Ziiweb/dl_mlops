from pathlib import Path
import wandb

run = wandb.init(project="dl_mlops")

artifact_filepath = Path("./quantile_lstm_checkpoint.pth")
  
logged_artifact = run.log_artifact(
  artifact_filepath,
  "forecasting_temperature",
  type="model"
)
run.link_artifact(   
  artifact=logged_artifact,  
  #target_path="javiergarpe1979-upm-org/wandb-registry-model/dl_mlops"
  target_path="javiergarpe1979-upm/dl_mlops"
)
run.finish()