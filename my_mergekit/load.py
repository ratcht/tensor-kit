import torch
from transformers import AutoModel, MistralModel

# Merge configurations

def get_device_map() -> str:
  return 'cuda' if torch.cuda.is_available() else 'cpu'
device = get_device_map()  # 'cpu'

class ModelLoader:
  def __init__(self, model_names: list[str], model_weights: list[float], offload_folder="offload"):
    self.model_names = model_names
    self.model_weights = model_weights
    self.models = [MistralModel.from_pretrained(model, device_map="auto", offload_folder=offload_folder) for model in model_names]

