import torch
import torch.nn as nn

class MyEnsembleELM(nn.Module):
    def __init__(self, models=[]):
        super().__init__()
        self.models = models

    def forward(self, inputs):
        with torch.no_grad():
            model_predictions = []

            for model in self.models:
                model_predictions.append(model(inputs))
                
            avg_prediction = torch.mean(torch.stack(model_predictions), dim=0)
        
        return avg_prediction
