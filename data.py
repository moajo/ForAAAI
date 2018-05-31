from torchtext import data
import unicodedata
from pathlib import Path
import pickle

class Data():
    def __init__(self, root, size, lang, batch_size, min_freq=2, device=-1, tokenizer=None):
        self.root = root
        self.size = size
        self.lang = lang
        self.batch_size = batch_size
        self.min_freq = min_freq
        self.device = device if device >= 0 else None
        self.tokenizer = tokenizer if tokenizer is not None else self.default_tokenizer
        
    def make_dataset(self, save_path=None, logger=print):
        # prepare Field
        TEXT_src = data.Field(sequential=True, tokenize=self.tokenizer, init_token='<sos>', eos_token='<eos>')
        TEXT_trg = data.Field(sequential=True, tokenize=self.tokenizer, init_token='<sos>', eos_token='<eos>')

        # file path
        train_file = self.root / self.lang / self.size / 'train.{}'.format(self.lang)
        val_file   = self.root / self.lang / 'dev.{}'.format(self.lang)
        test_file  = self.root / self.lang / 'test.{}'.format(self.lang)

        # load dataset and tokenize
        train, val, test = data.TabularDataset.splits(
            path='./', train=train_file, validation=val_file, test=test_file, format='tsv',
            fields=[('src', TEXT_src), ('trg', TEXT_trg)])

        # make vocab
        TEXT_src.build_vocab(train, min_freq=self.min_freq)
        TEXT_trg.build_vocab(train, min_freq=self.min_freq)
        if save_path is not None:
            with open(save_path / 'vocab.pkl', 'wb') as f:
                pickle.dump(TEXT_src, f)
                pickle.dump(TEXT_trg, f)
            logger('TEXT object is saved on {}'.format(save_path / 'vocab.pkl'))

        # make batch
        train_iter, val_iter = data.Iterator.splits(
            (train, val), batch_size=self.batch_size, device=self.device, repeat=False,
            sort_key=lambda x: data.interleave_keys(len(x.src), len(x.trg)))

        test_iter = data.Iterator(
            test, batch_size=self.batch_size, device=self.device, train=False, sort=False)

        # set attribute
        self.TEXT_src = TEXT_src
        self.TEXT_trg = TEXT_trg
        self.train_iter = train_iter
        self.val_iter   = val_iter
        self.test_iter  = test_iter

    def default_tokenizer(self, text):
        return unicodedata.normalize('NFKC', text).lower().split(' ')

if __name__ == '__main__':
    root = Path('Data')
    size = '20k'
    lang = 'ja-en'
    batch_size = 100
    gpu_id = -1
    d = Data(root, size, lang, batch_size, device=gpu_id)
    d.make_dataset()
    train_iter = d.train_iter
    print(len(d.TEXT_src.vocab))
    exit()
    epoch = 20
    for e in range(epoch):
        # e+1: current epoch
        for i, batch in enumerate(train_iter):
            # i+1: current iteration in epoch
            print('epoch:{}, iteration:{}, src size:{}, trg size:{}'.
                  format(e+1, i+1, batch.src.size(), batch.trg.size()))
        exit()
