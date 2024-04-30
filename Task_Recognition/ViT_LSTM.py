import numpy
import os
import cv2
import torch
from torch import nn, optim
from torchvision import models,transforms
from torch.nn.utils import weight_norm
import torch.nn.functional as F
import yaml
import copy
import timm
import numpy as np
import PIL


class ViT_FeatureExtractor(nn.Module):
    def __init__(self, num_output_features, num_classes):
        super(ViT_FeatureExtractor, self).__init__()

        # Define the transformations
        self.transforms = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # Load the pretrained Vision Transformer model
        self.vit = timm.create_model('vit_base_patch16_224', pretrained=True)

        # Remove the classification head
        self.vit.head = nn.Identity()

        # Add a new linear layer for feature extraction
        num_previous_layer_features = 768
        self.linear1 = nn.Linear(num_previous_layer_features, num_output_features)

        # Add a sigmoid activation function
        self.sig = nn.Sigmoid()

        # Add a new linear layer for the output
        self.output_head = nn.Linear(num_output_features, num_classes)

    def forward(self, x):
        # Check if x is a PIL Image or numpy ndarray
        if isinstance(x, (np.ndarray, PIL.Image.Image)):
            # Apply the transformations
            x = self.transforms(x)

        # Pass the input through the Vision Transformer model
        x = self.vit(x)

        # Pass the output through the linear layer
        x = self.linear1(x)

        # Apply the sigmoid activation function
        x = self.sig(x)

        # Pass the output through the output head
        x = self.output_head(x)

        return x
class ViT_LSTM:
    def __init__(self, num_output_features, num_classes, num_features, sequence_length, device):
        self.vit_model = ViT_FeatureExtractor(num_output_features, num_classes)
        self.lstm_model = WorkflowLSTM(num_features, num_classes)
        self.sequence_length = sequence_length
        self.sequence = numpy.zeros((self.sequence_length, num_features))
        self.device = device

    def loadModel(self, vit_model_path, lstm_model_path):
        vit_ckpt = torch.load(vit_model_path, map_location="cpu")
        self.vit_model.load_state_dict(vit_ckpt["model"], strict=True)
        lstm_ckpt = torch.load(lstm_model_path, map_location="cpu")
        self.lstm_model.load_state_dict(lstm_ckpt["model"], strict=True)

    def predict(self, image):
        self.vit_model.eval()
        self.lstm_model.eval()
        with torch.no_grad():
            img_tensor = self.vit_model.transforms(image.resize((224,224)))
            image = torch.from_numpy(numpy.array([img_tensor])).cuda(self.device)
            preds = self.vit_model.forward(image)
            pred = preds.cpu().detach().numpy()
            self.sequence = numpy.concatenate((self.sequence[:-1,],pred),axis=0)
            expanded_sequence = numpy.expand_dims(self.sequence,axis=0).astype(float)
            taskPrediction = self.lstm_model(torch.from_numpy(expanded_sequence).float().cuda(self.device))
            taskPrediction = taskPrediction.cpu().numpy()
            class_num = numpy.argmax(taskPrediction[0][-1])
            networkOutput = str(self.task_class_mapping[class_num]) + str(taskPrediction)
            return networkOutput
class WorkflowLSTM(nn.Module):
    def __init__(self,num_features,num_classes):
        super().__init__()
        self.hidden_size = 128
        self.lstm = nn.LSTM(input_size=num_features, hidden_size=self.hidden_size, num_layers=2, batch_first = True,bidirectional=False, dropout=0.2)
        self.linear_1 = nn.Linear(128,64)
        self.relu1 = nn.ReLU()
        self.droput1 = nn.Dropout(0.2)
        self.linear_2 = nn.Linear(64, num_classes)
        self.relu2 = nn.ReLU()
        self.droput2 = nn.Dropout(0.2)
        self.linear_3 = nn.Linear(num_classes, num_classes)
        self.softmax = nn.Softmax(dim=-1)
        for name, param in self.lstm.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight' in name:
                nn.init.xavier_normal_(param)

    def forward(self,sequence):
        sequence = sequence
        #h_0 = torch.autograd.Variable(torch.zeros(1, sequence.size(0), self.hidden_size)).cuda("cuda")  # hidden state
        #c_0 = torch.autograd.Variable(torch.zeros(1, sequence.size(0), self.hidden_size)).cuda("cuda")
        x, (hn,cn) = self.lstm(sequence)
        last_time_step = x[:, -1, :]
        #x = self.relu1(x)
        x = self.linear_1(x)
        x = self.relu1(x)
        x = self.linear_2(x)

        scores = F.log_softmax(x,dim=1)
        '''x = self.relu1(x)
        x = self.droput1(x)
        x = self.linear_2(x)
        x = self.relu2(x)
        x = self.droput2(x)
        x = self.linear_3(x)'''
        #x = self.softmax(x)
        return x
class MultiStageModel(nn.Module):
    def __init__(self, stages = 4, layers = 10, feature_maps = 64, feature_dimension = 2048,out_features = 10, causal_conv = True):
        self.num_stages = stages #hparams.mstcn_stages  # 4 #2
        self.num_layers = layers #hparams.mstcn_layers  # 10  #5
        self.num_f_maps = feature_maps #hparams.mstcn_f_maps  # 64 #64
        self.dim = feature_dimension #hparams.mstcn_f_dim  #2048 # 2048
        self.num_classes = out_features #hparams.out_features  # 7
        self.causal_conv = causal_conv #hparams.mstcn_causal_conv
        print(
            f"num_stages_classification: {self.num_stages}, num_layers: {self.num_layers}, num_f_maps:"
            f" {self.num_f_maps}, dim: {self.dim}")
        super(MultiStageModel, self).__init__()
        self.stage1 = SingleStageModel(self.num_layers,
                                       self.num_f_maps,
                                       self.dim,
                                       self.num_classes,
                                       causal_conv=self.causal_conv)
        self.stages = nn.ModuleList([
            copy.deepcopy(
                SingleStageModel(self.num_layers,
                                 self.num_f_maps,
                                 self.num_classes,
                                 self.num_classes,
                                 causal_conv=self.causal_conv))
            for s in range(self.num_stages - 1)
        ])
        self.smoothing = False
        self.softmax = nn.Softmax(dim=2)

    def forward(self, x):
        batch_output = []
        for i in range(x.size(0)):
            out_classes = self.stage1(x[i].unsqueeze(0).transpose(2,1))
            outputs_classes = out_classes.unsqueeze(0)
            for s in self.stages:
                out_classes = s(F.softmax(out_classes, dim=1))
                outputs_classes = torch.cat(
                    (outputs_classes, out_classes.unsqueeze(0)), dim=0)
            #outputs_classes = self.softmax(outputs_classes)
            outputs_classes=outputs_classes.transpose(2,3)
            batch_output.append(outputs_classes)
        return torch.stack(batch_output).squeeze(2)

    @staticmethod
    def add_model_specific_args(parser):  # pragma: no cover
        mstcn_reg_model_specific_args = parser.add_argument_group(
            title='mstcn reg specific args options')
        mstcn_reg_model_specific_args.add_argument("--mstcn_stages",
                                                   default=4,
                                                   type=int)
        mstcn_reg_model_specific_args.add_argument("--mstcn_layers",
                                                   default=10,
                                                   type=int)
        mstcn_reg_model_specific_args.add_argument("--mstcn_f_maps",
                                                   default=64,
                                                   type=int)
        mstcn_reg_model_specific_args.add_argument("--mstcn_f_dim",
                                                   default=2048,
                                                   type=int)
        mstcn_reg_model_specific_args.add_argument("--mstcn_causal_conv",
                                                   action='store_true')
        return parser
class SingleStageModel(nn.Module):
    def __init__(self,
                 num_layers,
                 num_f_maps,
                 dim,
                 num_classes,
                 causal_conv=False):
        super(SingleStageModel, self).__init__()
        self.conv_1x1 = nn.Conv1d(dim, num_f_maps, 1)

        self.layers = nn.ModuleList([
            copy.deepcopy(
                DilatedResidualLayer(2**i,
                                     num_f_maps,
                                     num_f_maps,
                                     causal_conv=causal_conv))
            for i in range(num_layers)
        ])
        self.conv_out_classes = nn.Conv1d(num_f_maps, num_classes, 1)

    def forward(self, x):
        out = self.conv_1x1(x)
        for layer in self.layers:
            out = layer(out)
        out_classes = self.conv_out_classes(out)
        return out_classes
class DilatedResidualLayer(nn.Module):
    def __init__(self,
                 dilation,
                 in_channels,
                 out_channels,
                 causal_conv=False,
                 kernel_size=3):
        super(DilatedResidualLayer, self).__init__()
        self.causal_conv = causal_conv
        self.dilation = dilation
        self.kernel_size = kernel_size
        if self.causal_conv:
            self.conv_dilated = nn.Conv1d(in_channels,
                                          out_channels,
                                          kernel_size,
                                          padding=(dilation *
                                                   (kernel_size - 1)),
                                          dilation=dilation)
        else:
            self.conv_dilated = nn.Conv1d(in_channels,
                                          out_channels,
                                          kernel_size,
                                          padding=dilation,
                                          dilation=dilation)
        self.conv_1x1 = nn.Conv1d(out_channels, out_channels, 1)
        self.dropout = nn.Dropout()

    def forward(self, x):
        out = F.relu(self.conv_dilated(x))
        if self.causal_conv:
            out = out[:, :, :-(self.dilation * 2)]
        out = self.conv_1x1(out)
        out = self.dropout(out)
        return (x + out)
class DilatedSmoothLayer(nn.Module):
    def __init__(self, causal_conv=True):
        super(DilatedSmoothLayer, self).__init__()
        self.causal_conv = causal_conv
        self.dilation1 = 1
        self.dilation2 = 5
        self.kernel_size = 5
        if self.causal_conv:
            self.conv_dilated1 = nn.Conv1d(7,
                                           7,
                                           self.kernel_size,
                                           padding=self.dilation1 * 2 * 2,
                                           dilation=self.dilation1)
            self.conv_dilated2 = nn.Conv1d(7,
                                           7,
                                           self.kernel_size,
                                           padding=self.dilation2 * 2 * 2,
                                           dilation=self.dilation2)

        else:
            self.conv_dilated1 = nn.Conv1d(7,
                                           7,
                                           self.kernel_size,
                                           padding=self.dilation1 * 2,
                                           dilation=self.dilation1)
            self.conv_dilated2 = nn.Conv1d(7,
                                           7,
                                           self.kernel_size,
                                           padding=self.dilation2 * 2,
                                           dilation=self.dilation2)
        self.conv_1x1 = nn.Conv1d(7, 7, 1)
        self.dropout = nn.Dropout()

    def forward(self, x):
        x1 = self.conv_dilated1(x)
        x1 = self.conv_dilated2(x1[:, :, :-4])
        out = F.relu(x1)
        if self.causal_conv:
            out = out[:, :, :-((self.dilation2 * 2) * 2)]
        out = self.conv_1x1(out)
        out = self.dropout(out)
        return (x + out)
