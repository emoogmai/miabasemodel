import os
import yaml

mia_model_config = None

#Load the configuration file for mia base llm
with open("configs/mia_config.yml", "rt") as f:
    mia_model_config = yaml.safe_load(f.read())
