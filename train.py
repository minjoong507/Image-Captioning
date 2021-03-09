import torch
from data_loader import get_dataloader
from model import Encoder, Decoder
from torchvision.transforms import transforms
import argparse
import torch.nn as nn
import torch.optim as optim
from make_vocab import Make_vocab, Vocab
import pickle
from torch.nn.utils.rnn import pack_padded_sequence
import os

device = 'cuda:5' if torch.cuda.is_available() else 'cpu'
torch.cuda.set_device(device)
cuda = torch.device(device)
print(cuda)


class Image_Captioning:
    def __init__(self):
        parser = argparse.ArgumentParser(description='Image Captioning')
        parser.add_argument('--root', default='../../../cocodataset/', type=str)
        parser.add_argument('--crop_size', default=224, type=int)
        parser.add_argument('--epochs', default=100, type=int)
        parser.add_argument('--lr', default=1e-4, type=float)
        parser.add_argument('--batch_size', default=128, help='')
        parser.add_argument('--num_workers', default=4, type=int)
        parser.add_argument('--embed_dim', default=256, type=int)
        parser.add_argument('--hidden_size', default=512, type=int)
        parser.add_argument('--num_layers', default=1, type=int)
        parser.add_argument('--model_path', default='./model/', type=str)
        parser.add_argument('--vocab_path', default='./vocab/', type=str)
        parser.add_argument('--save_step', default=1000, type=int)

        self.args = parser.parse_args()
        self.Multi_GPU = False

        # if torch.cuda.device_count() > 1:
        #     print('Multi GPU Activate!')
        #     print('Using GPU :', int(torch.cuda.device_count()))
        #     self.Multi_GPU = True

        os.makedirs(self.args.model_path, exist_ok=True)

        transform = transforms.Compose([
        transforms.RandomCrop(self.args.crop_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
                             (0.229, 0.224, 0.225))
        ])

        with open(self.args.vocab_path + 'vocab.pickle', 'rb') as f:
            data = pickle.load(f)

        self.vocab = data

        self.DataLoader = get_dataloader(root=self.args.root,
                                         transform=transform,
                                         shuffle=True,
                                         batch_size=self.args.batch_size,
                                         num_workers=self.args.num_workers,
                                         vocab=self.vocab)

        self.Encoder = Encoder(embed_dim=self.args.embed_dim)
        self.Decoder = Decoder(embed_dim=self.args.embed_dim,
                               hidden_size=self.args.hidden_size,
                               vocab_size=len(self.vocab),
                               num_layers=self.args.num_layers)
        # print(self.Encoder)
        # print(self.Decoder)


    def train(self):
        if self.Multi_GPU:
            self.Encoder = torch.nn.DataParallel(self.Encoder)
            self.Decoder = torch.nn.DataParallel(self.Decoder)
            parameters = list(self.Encoder.module.fc.parameters()) + list(self.Encoder.module.BN.parameters()) + list(self.Decoder.parameters())
        else:
            parameters = list(self.Encoder.fc.parameters()) + list(self.Encoder.BN.parameters()) + list(self.Decoder.parameters())

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(parameters, lr=self.args.lr)

        self.Encoder.cuda()
        self.Decoder.cuda()

        self.Encoder.train()
        self.Decoder.train()

        print('-' * 100)
        print('Now Training')
        print('-' * 100)

        for epoch in range(self.args.epochs):
            total_loss = 0
            for batch_idx, (image, captions, lengths) in enumerate(self.DataLoader):
                optimizer.zero_grad()
                image, captions = image.cuda(), captions.cuda()

                targets = pack_padded_sequence(captions, lengths, batch_first=True)[0]

                if self.Multi_GPU:
                    img_features = nn.parallel.DataParallel(self.Encoder, image)
                    outputs = nn.parallel.DataParallel(self.Decoder, (img_features, captions, lengths))
                else:
                    img_features = self.Encoder(image)
                    outputs = self.Decoder(img_features, captions, lengths)

                loss = criterion(outputs, targets)
                total_loss += loss.item()

                loss.backward()
                optimizer.step()

                if batch_idx % 30 == 0:
                    print('Epoch : {}, Step : [{}/{}], Step Loss : {:.4f}'.format(epoch, batch_idx, len(self.DataLoader), loss.item()))

            print('Epoch : [{}/{}], Total loss : {:.4f}'.format(epoch, self.args.epochs, total_loss / len(self.DataLoader)))

        print('Now saving the models')
        torch.save(self.Encoder.state_dict(), self.args.model_path + 'Encoder-{}.ckpt'.format(self.args.epochs))
        torch.save(self.Decoder.state_dict(), self.args.model_path + 'Decoder-{}.ckpt'.format(self.args.epochs))


if __name__ == '__main__':
    b = Image_Captioning()
    b.train()