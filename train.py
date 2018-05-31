#!/usr/bin/env python
import argparse
from pathlib import Path
from datetime import datetime
import torch
import torch.nn as nn
import torch.optim as optim
from data import Data
from model import Seq2Seq#, Map
from tools import log, strip_token, label2word, calc_bleu_score

class Classifier(nn.Module):
    def __init__(self, model_s2t, model_t2s, mapping=None):
        super(Classifier, self).__init__()
        self.model_s2t = model_s2t
        self.model_t2s = model_t2s
        self.mapping = mapping
        self.loss_fun = nn.CrossEntropyLoss(ignore_index=1)

    def forward(self, src, trg, train=True):
        # target2source 
        src_sos = src[ :-1] if train else None
        src_eos = src[1:  ]
        src_length = None if train else len(src_eos)
        labels_t2s, output_t2s, attn_t2s = self.model_t2s(trg, src_sos, length=src_length)

        # sorce2target
        trg_sos = trg[ :-1] if train else None
        trg_eos = trg[1:  ]
        trg_length = None if train else len(trg_eos)
        labels_s2t, output_s2t, attn_s2t = self.model_s2t(src, trg_sos, length=trg_length)

        # sorce2target loss
        L, B, V = output_s2t.size()
        output_s2t = output_s2t.view(L * B, V)
        trg_eos = trg_eos.view(L * B)
        loss_s2t = self.loss_fun(output_s2t, trg_eos)
        
        # target2source loss
        L, B, V = output_t2s.size()
        output_t2s = output_t2s.view(L * B, V)
        src_eos = src_eos.view(L * B)
        loss_t2s = self.loss_fun(output_t2s, src_eos)

        return labels_s2t, labels_t2s, loss_s2t, loss_t2s

def main():
    parser = argparse.ArgumentParser(description='Custom_loop PyTorch')
    parser.add_argument('--batchsize', '-b', type=int, default=100,
                        help='Number of images in each mini-batch')
    parser.add_argument('--epoch', '-e', type=int, default=5,
                        help='Number of sweeps over the dataset to train')
    parser.add_argument('--size', '-s', type=str, default='100k')
    parser.add_argument('--freq', type=int, default=2)
    parser.add_argument('--lang', type=str, default='ja-en')
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--out', '-o', type=str, default='result',
                        help='Directory to output the result')
    parser.add_argument('--model', '-m', type=str, default=None,
                        help='model')
    parser.add_argument('--interval', '-i', type=int, default=100,
                        help='Log interval')
    args = parser.parse_args()

    log('GPU: {}'.format(args.gpu))
    log('# Minibatch-size: {}'.format(args.batchsize))
    log('# epoch: {}'.format(args.epoch))

    # Load the dataset
    root = Path('Data')
    
    data = Data(root, args.size, args.lang, args.batchsize, min_freq=args.freq, device=args.gpu)
    data.make_dataset(Path(args.out), logger=log)
    train_iter = data.train_iter
    val_iter   = data.val_iter
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
    #mapping = Map()
    mapping = None
    model = Classifier(source2target, target2source, mapping)
    if args.model is not None:
        model.load_state_dict(torch.load(args.model))
    model.to(device=device)

    # Setup an optimizer
    optimizer_s2t = optim.Adam(source2target.parameters())
    optimizer_t2s = optim.Adam(target2source.parameters())
    #optimizer_map = optim.Adam(mapping.parameters())

    # training loop
    start_time = datetime.now()
    start_epoch = 0 if args.model is None else int(args.model.split('.')[0][-1])
    for e in range(start_epoch, args.epoch):
        epoch = e + 1
        # train
        model.train()
        sum_loss_train_s2t, sum_loss_train_t2s = 0, 0
        tmp_loss_train_s2t, tmp_loss_train_t2s = 0, 0
        for i, batch in enumerate(train_iter):
            iteration = i + 1
            optimizer_s2t.zero_grad()
            optimizer_t2s.zero_grad()
            #mapping.zero_grad()
            _, _, loss_s2t, loss_t2s = model(batch.src, batch.trg)
            sum_loss_train_s2t += loss_s2t.item()
            sum_loss_train_t2s += loss_t2s.item()
            tmp_loss_train_s2t += loss_s2t.item()
            tmp_loss_train_t2s += loss_t2s.item()
            loss_t2s.backward()
            loss_s2t.backward()
            optimizer_s2t.step()
            optimizer_t2s.step()
            #mapping.step()
            if iteration % args.interval == 0:
                log('epoch:{},\titeration:{},\tloss(s2t):{},\tloss(t2s):{}'.
                    format(epoch, iteration,
                           tmp_loss_train_s2t/args.interval,
                           tmp_loss_train_t2s/args.interval),
                    overwrite=True
                )
                tmp_loss_train_s2t, tmp_loss_train_t2s = 0, 0

        # validation
        sum_loss_val_s2t, sum_loss_val_t2s = 0, 0
        model.eval()
        for batch in val_iter:
            with torch.no_grad():
                _, _, loss_s2t, loss_t2s = model(batch.src, batch.trg)
            sum_loss_val_s2t += loss_s2t.item()
            sum_loss_val_t2s += loss_t2s.item()
        # bleu
        hyp_s2t, ref_s2t, hyp_t2s, ref_t2s = [], [], [], []
        for batch in val_iter:
            with torch.no_grad():
                labels_s2t, labels_t2s, _, _ = model(batch.src, batch.trg, train=False)
            labels = labels_s2t.transpose(0, 1).cpu().numpy().tolist()
            golds = batch.trg.transpose(0, 1).cpu().numpy().tolist()
            # target2source
            hyp_s2t.extend([label2word(strip_token(line),
                                       vocab=data.TEXT_trg.vocab.itos,
                                       return_str=False)
                            for line in labels])
            ref_s2t.extend([[label2word(strip_token(line),
                                        vocab=data.TEXT_trg.vocab.itos,
                                        return_str=False)]
                            for line in golds])
            # target2source
            labels = labels_t2s.transpose(0, 1).cpu().numpy().tolist()
            golds = batch.src.transpose(0, 1).cpu().numpy().tolist()
            hyp_t2s.extend([label2word(strip_token(line),
                                       vocab=data.TEXT_src.vocab.itos,
                                       return_str=False) for line in labels])
            ref_t2s.extend([[label2word(strip_token(line),
                                        vocab=data.TEXT_src.vocab.itos,
                                        return_str=False)] for line in golds])
        bleu_s2t = calc_bleu_score(hyp_s2t, ref_s2t)
        bleu_t2s = calc_bleu_score(hyp_t2s, ref_t2s)
        
        # log stdout
        log('epoch:{}'.format(epoch), start_time)
        log('[train]\tepoch:{}\tloss(s2t):{},\tloss(t2s):{}'
            .format(epoch,
                    sum_loss_train_s2t/len(train_iter),
                    sum_loss_train_t2s/len(train_iter)))
        log('[val]\tepoch:{}\tloss(s2t):{},\tloss(t2s):{}'
            .format(epoch,
                    sum_loss_val_s2t/len(val_iter),
                    sum_loss_val_t2s/len(val_iter)))
        log('[val]\tepoch:{}\tbleu(s2t):{},\tbleu(t2s):{}'
            .format(epoch, bleu_s2t, bleu_t2s))
        # log file
        log('epoch\t{}\tloss(s2t)\t{}\tloss(t2s)\t{}'
            .format(epoch,
                    sum_loss_train_s2t/len(train_iter),
                    sum_loss_train_t2s/len(train_iter)),
            filename=Path(args.out) / 'train.log')
        log('epoch\t{}\tloss(s2t)\t{}\tloss(t2s)\t{}'
            .format(epoch,
                    sum_loss_val_s2t/len(val_iter),
                    sum_loss_val_t2s/len(val_iter)),
            filename=Path(args.out) / 'val.log')
        log('epoch\t{}\tbleu(s2t)\t{}\tbleu(t2s)\t{}'.
            format(epoch, bleu_s2t, bleu_t2s),
            filename=Path(args.out) / 'bleu.log')
        # Seve the model
        torch.save(model.state_dict(), Path(args.out) / 'epoch_{}.m'.format(epoch))

if __name__ == '__main__':
    main()
