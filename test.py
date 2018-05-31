#!/usr/bin/env python
import argparse
from pathlib import Path
import torch
import torch.nn as nn
from data import Data
from model import Seq2Seq
from tools import log, strip_token, label2word

class Classifier(nn.Module):
    def __init__(self, model_s2t, model_t2s, mapping=None):
        super(Classifier, self).__init__()
        self.model_s2t = model_s2t
        self.model_t2s = model_t2s
        self.mapping = mapping
        self.loss_fun = nn.CrossEntropyLoss(ignore_index=1)
        
    def __call__(self, src, trg, train=False, direction='s2t'):
        if direction == 's2t':
            _src = src
            _model = self.model_s2t
        else:# direction == 't2s':
            _src = trg
            _model = self.model_t2s
        
        labels, _, _ = _model(_src, None)
        return labels

def main():
    parser = argparse.ArgumentParser(description='Custom_loop PyTorch')
    parser.add_argument('--batchsize', '-b', type=int, default=100,
                        help='Number of images in each mini-batch')
    parser.add_argument('--size', '-s', type=str, default='100k')
    parser.add_argument('--freq', type=int, default=2)
    parser.add_argument('--lang', type=str, default='ja-en')
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--out', '-o', type=str, default='result',
                        help='Directory to output the result')
    parser.add_argument('--model', '-m', type=str, default=None,
                        help='model')
    parser.add_argument('--interval', '-i', type=int, default=5,
                        help='Log interval')
    parser.add_argument('--direction', '-d', type=str, default='s2t')
    args = parser.parse_args()

    log('GPU: {}'.format(args.gpu))
    log('# Minibatch-size: {}'.format(args.batchsize))

    # Load the dataset
    root = Path('Data')

    data = Data(root, args.size, args.lang, args.batchsize, min_freq=args.freq, device=args.gpu)
    data.make_dataset(Path(args.out), logger=log)
    test_iter = data.test_iter
    source_vocab_size = len(data.TEXT_src.vocab)
    target_vocab_size = len(data.TEXT_trg.vocab)
    log('vocab size source:{}, target:{}'.format(source_vocab_size, target_vocab_size))

    if args.gpu >= 0:
        # Make a speciied GPU current
        device = torch.device('cuda:{}'.format(args.gpu))
    else:
        device = torch.device('cpu')
    
    # Set up a neural network to train
    source2target = Seq2Seq(source_vocab_size, target_vocab_size)
    target2source = Seq2Seq(target_vocab_size, source_vocab_size)
    model = Classifier(source2target, target2source, mapping=None)
    if args.model is not None:
        model.load_state_dict(torch.load(args.model))
    model.to(device=device)

    # Test
    model.eval()
    outputs = []
    for i, batch in enumerate(test_iter):
        with torch.no_grad():
            output = model(batch.src, batch.trg, train=False, direction=args.direction)
        output = output.transpose(0, 1).cpu().numpy().tolist()
        if args.direction == 's2t':
            vocab = data.TEXT_trg.vocab.itos
        else:
            vocab = data.TEXT_src.vocab.itos
        if i % args.interval == 0:
            print('{}\n'.format(
                label2word(strip_token(output[0]), vocab=vocab)))
        outputs.extend([label2word(strip_token(line), vocab=vocab) for line in output])
    with open(Path(args.out) / 'output.txt', 'w') as f:
        print('\n'.join(outputs), file=f)

if __name__ == '__main__':
    main()
