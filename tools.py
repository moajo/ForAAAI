from datetime import datetime
from nltk.translate import bleu_score

def strip_token(labels, sos_id = 2, eos_id = 3):
    start_index = labels.index(sos_id)+1 if sos_id in labels else 0
    end_index = labels.index(eos_id) if eos_id in labels else None
    labels = labels[start_index:end_index]
    return labels

def label2word(labels, vocab, return_str=True, sep=' '):
    words = [vocab[label] if label < len(vocab) else vocab[0] for label in labels]
    if return_str:
        words = sep.join(words)
    return words

def calc_bleu_score(hyp, ref):
    bleu = bleu_score.corpus_bleu(ref, hyp,\
        smoothing_function=bleu_score.SmoothingFunction().method1)
    return bleu

def log(msg, start_time=None, overwrite=False, filename=None):
    current_time = datetime.now()
    current_time_s = current_time.strftime("%Y/%m/%d %H:%M:%S")

    if start_time is None:
        text = '{}|{}'.format(current_time_s, msg)
    else:
        elapsed_time = (current_time - start_time).total_seconds()
        text = '{}|{}, {}[sec]'.format(current_time_s, msg, elapsed_time)
    if filename is not None:
        with open(filename, 'a') as f:
            print(text, file=f)
    else:
        print(text)
