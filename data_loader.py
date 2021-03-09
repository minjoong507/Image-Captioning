import torch
import nltk
import numpy as np
from pycocotools.coco import COCO
from torchvision.transforms import transforms
from torch.utils import data
from PIL import Image
from make_vocab import Make_vocab

"""
    Please do first preprocess function. 
"""


class ImageCaption_DataLoader(data.Dataset):
    def __init__(self, root, transform, vocab):
        self.root = root
        self.vocab = vocab
        self.coco = COCO(self.root + 'annotations/captions_val2017.json')
        self.coco_ids = list(self.coco.anns.keys())
        self.ids = list(np.load('coco idx.npy')) # self.preprocess_idx()
        self.transform = transform

    def __getitem__(self, idx):
        data = self.coco.anns[self.ids[idx]]
        captions = data['caption']
        img_id = str(data['image_id'])

        # load image data
        img_id = '0' * (12 - len(img_id)) + img_id
        img_path = self.root + 'val2017/' + str(img_id) + '.jpg'
        image = Image.open(img_path).convert('RGB')
        image = self.transform(image)

        # load caption data
        caption = []
        tokens = nltk.tokenize.word_tokenize(str(captions).lower())
        caption.append(self.vocab('<start>'))
        caption.extend([self.vocab(token) for token in tokens])
        caption.append(self.vocab('<end>'))
        caption = torch.tensor(caption)

        return image, caption

    def __len__(self):
        return len(self.ids)

    def preprocess_idx(self):
        T = transforms.ToTensor()
        preprocess = []
        for i in range(len(self.coco_ids)):
            data = self.coco.anns[self.coco_ids[i]]
            img_id = str(data['image_id'])

            # load image data
            img_id = '0' * (12 - len(img_id)) + img_id
            img_path = self.root + 'val2017/' + str(img_id) + '.jpg'
            image = Image.open(img_path).convert('RGB')
            image = T(image)

            if image.shape[1] < 224 or image.shape[2] < 224:
                continue
            else:
                preprocess.append(self.coco_ids[i])

            if (i + 1) % 1000 == 0:
                print("[{}/{}] Checking the images.".format(i + 1, len(self.coco_ids)))

        print('Saved coco idx file!')

        np.save('coco idx.npy', np.array(preprocess))

        return preprocess


def collate_fn(data):
    data.sort(key=lambda x: len(x[1]), reverse=True)
    images, captions = zip(*data)

    images = torch.stack(images, 0)
    lengths = [len(caption) for caption in captions]
    target = torch.zeros((len(captions), max(lengths))).long()

    for i, cap in enumerate(captions):
        end = lengths[i]
        target[i, :end] = cap[:end]

    return images, target, lengths


def get_dataloader(root, transform, batch_size, shuffle, num_workers, vocab):
    CocoData = ImageCaption_DataLoader(root, transform, vocab)
    Dataloader = torch.utils.data.DataLoader(dataset=CocoData,
                                             batch_size=batch_size,
                                             shuffle=shuffle,
                                             num_workers=num_workers,
                                             collate_fn=collate_fn,
                                             drop_last=True)

    return Dataloader


# transform = transforms.Compose([
#         transforms.RandomCrop(224),
#         transforms.RandomHorizontalFlip(),
#         transforms.ToTensor(),
#         transforms.Normalize((0.485, 0.456, 0.406),
#                              (0.229, 0.224, 0.225))
#         ])

# cc = Make_vocab()
# a = ImageCaption_DataLoader('../../../cocodataset/', transform, cc)
# a.main()