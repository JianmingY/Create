import os
import torch
from torch import nn
import pandas
import timm
from sklearn.metrics import jaccard_score
from scipy.stats import mode
import argparse

class ViT_Transformer(nn.Module):
    def __init__(self, num_classes, num_features):
        super(ViT_Transformer, self).__init__()
        self.vit_model = timm.create_model('vit_base_patch16_224', pretrained=True)
        self.vit_model.head = nn.Linear(self.vit_model.head.in_features, num_features)

        self.transformer_model = nn.Transformer(d_model=num_features, nhead=4, num_encoder_layers=2)
        self.fc = nn.Linear(num_features, num_classes)

    def forward(self, x):
        x = self.vit_model(x)  # Pass the input through the ViT model
        x = self.transformer_model(x)  # Pass the output from the ViT model to the Transformer model
        x = self.fc(x)  # Pass the output from the Transformer model to the fully connected layer
        return x

def adjust_learning_rate(lr, optimizer):
    lr = lr*0.7
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    return lr

class FocalLoss(nn.Module):
    def __init__(self, gamma=2, weight=None):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.weight = weight

    def forward(self, input, target):
        logpt = F.log_softmax(input, dim=1)
        pt = torch.exp(logpt)
        logpt = (1-pt)**self.gamma * logpt
        loss = F.nll_loss(logpt, target, self.weight)
        return loss

if __name__ == "__main__":
    parser = argparse.ArgumentParser('ViT_Transformer training script')
    args = parser.parse_args()
