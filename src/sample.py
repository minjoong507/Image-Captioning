import torch
from data_loader import get_dataloader
from model import Encoder, Decoder
from torchvision.transforms import transforms
import argparse
from make_vocab import Make_vocab, Vocab
import pickle
from PIL import Image

device = 'cuda' if torch.cuda.is_available() else 'cpu'
cuda = torch.device(device)
print(cuda)

class sample:
    def __init__(self):
        parser = argparse.ArgumentParser(description='Image Captioning')
        parser.add_argument('--root', default='../../../cocodataset/', type=str)
        parser.add_argument('--sample_image', default='../../../cocodataset/val2017/000000579321.jpg', type=str)
        parser.add_argument('--epochs', default=100, type=int)
        parser.add_argument('--lr', default=1e-4, type=float)
        parser.add_argument('--batch_size', default=128, help='')
        parser.add_argument('--num_workers', default=4, type=int)
        parser.add_argument('--embed_dim', default=256, type=int)
        parser.add_argument('--hidden_size', default=512, type=int)
        parser.add_argument('--num_layers', default=1, type=int)
        parser.add_argument('--encoder_path', default='./model/Encoder-100.ckpt', type=str)
        parser.add_argument('--decoder_path', default='./model/Deocder-100.ckpt', type=str)
        parser.add_argument('--vocab_path', default='./vocab/', type=str)

        self.args = parser.parse_args()

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
            transforms.Resize((224, 224))
        ])

        with open(self.args.vocab_path + 'vocab.pickle', 'rb') as f:
            data = pickle.load(f)

        self.vocab = data

        self.DataLoader = get_dataloader(root=self.args.root,
                                         transform=self.transform,
                                         shuffle=True,
                                         batch_size=self.args.batch_size,
                                         num_workers=self.args.num_workers,
                                         vocab=self.vocab)

        self.Encoder = Encoder(embed_dim=self.args.embed_dim)
        self.Decoder = Decoder(embed_dim=self.args.embed_dim,
                               hidden_size=self.args.hidden_size,
                               vocab_size=len(self.vocab),
                               num_layers=self.args.num_layers)

    def load_image(self, image_path):
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image).unsqueeze(0)

        return image

    def main(self):
        self.Encoder.load_state_dict(torch.load(self.args.encoder_path))
        self.Decoder.load_state_dict(torch.load(self.args.decoder_path))

        self.Encoder = self.Encoder.cuda().eval()
        self.Decoder = self.Decoder.cuda().eval()

        sample_image = self.load_image(self.args.sample_image).cuda()
        output = self.Encoder(sample_image)
        output = self.Decoder.sample(output)[0].cpu().numpy()
        sample_caption = []

        for idx in output:
            word = self.vocab.idx2word[idx]
            sample_caption.append(word)
            if word == '<end>':
                break

        sentence = ' '.join(sample_caption)
        print(sentence)


if __name__ == '__main__':
    a = sample()
    a.main()










