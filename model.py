import torch
import torchvision.models as models
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence

"""
    You can check available models here. We use ResNet101 model.
    https://github.com/pytorch/vision/tree/master/torchvision/models
"""

class Encoder(nn.Module):
    def __init__(self, embed_dim):
        super(Encoder, self).__init__()
        self.embed_dim = embed_dim
        resnet = models.resnet101(pretrained=True) # load pretrained model
        resnet_modules = list(resnet.children())[:-1]   # except the last fc layer
        self.ResNet = nn.Sequential(*resnet_modules)
        self.fc = nn.Linear(resnet.fc.in_features, embed_dim)
        self.BN = nn.BatchNorm1d(embed_dim)
        # print(resnet)

    def forward(self, x):
        # Do not training ResNet.
        x = self.ResNet(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = self.BN(x)

        return x


class Decoder(nn.Module):
    def __init__(self, embed_dim, hidden_size, vocab_size, num_layers):
        super(Decoder, self).__init__()
        self.Embedding = nn.Embedding(vocab_size, embed_dim)
        self.LSTM = nn.LSTM(embed_dim, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, encoder_output, x, lengths):
        embed = self.Embedding(x)
        embed = torch.cat((encoder_output.unsqueeze(1), embed), 1)
        embed = pack_padded_sequence(embed, lengths, batch_first=True)
        output, _ = self.LSTM(embed)
        output = self.fc(output[0])

        return output

    def sample(self, features, states=None):
        sampled_ids = []
        inputs = features.unsqueeze(1)
        for i in range(20):
            hiddens, states = self.LSTM(inputs, states)
            outputs = self.fc(hiddens.squeeze(1))
            _, predicted = outputs.max(1)
            sampled_ids.append(predicted)
            inputs = self.Embedding(predicted)
            inputs = inputs.unsqueeze(1)
        sampled_ids = torch.stack(sampled_ids, 1)
        return sampled_ids

