from collections import Counter

class SimpleTokenizer:
    def __init__(self, texts):
        counter = Counter(" ".join(texts).split())
        self.vocab = {word: i for i, (word, _) in enumerate(counter.items())}
        self.ivocab = {i: w for w, i in self.vocab.items()}

    def encode(self, text):
        return [self.vocab[word] for word in text.split() if word in self.vocab]

    def decode(self, ids):
        return " ".join([self.ivocab[i] for i in ids if i in self.ivocab])

    def __len__(self):
        return len(self.vocab)
