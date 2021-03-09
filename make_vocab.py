from pycocotools.coco import COCO
import nltk
from collections import Counter
import pickle
import os

class Vocab:
    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0

    def add_word(self, word):
        self.word2idx[word] = self.idx
        self.idx2word[self.idx] = word
        self.idx += 1

    def __call__(self, word):
        if not word in self.word2idx:
            return self.word2idx['<unk>']
        return self.word2idx[word]

    def __len__(self):
        return len(self.word2idx)


class Make_vocab:
    def __init__(self, json='../../../cocodataset/annotations/captions_val2017.json', min_fq=5):
        self.coco = COCO(json)
        self.ids = self.coco.anns.keys()
        self.Counter = Counter()
        self.min_fq = min_fq
        self.length = 0
        self.vocab = Vocab()

    def get_vocab(self):
        for i, id in enumerate(self.ids):
            caption = str(self.coco.anns[id]['caption'])
            tokens = nltk.word_tokenize(caption.lower())
            self.length = max(len(tokens), self.length)
            self.Counter.update(tokens)

            if (i + 1) % 1000 == 0:
                print("[{}/{}] Tokenized the captions.".format(i + 1, len(self.ids)))

        self.vocab.add_word('<pad>')
        self.vocab.add_word('<start>')
        self.vocab.add_word('<end>')
        self.vocab.add_word('<unk>')

        words = [word for word, cnt in self.Counter.items() if cnt >= self.min_fq]

        for i, word in enumerate(words):
            self.vocab.add_word(word)

        return self.vocab

    def save_vocab(self):
        with open('vocab/vocab.pickle', 'wb') as f:
            pickle.dump(self.vocab, f)


if __name__ == '__main__':
    os.makedirs('./vocab/', exist_ok=True)

    a = Make_vocab()
    _ = a.get_vocab()
    a.save_vocab()

    with open('vocab/vocab.pickle', 'rb') as f:
        data = pickle.load(f)


