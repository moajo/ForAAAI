import torch
import torch.nn as nn
import torch.nn.functional as F

class Seq2Seq(nn.Module):
    def __init__(self, source_vocab_size, target_vocab_size):
        super(Seq2Seq, self).__init__()
        hidden_size = 256
        self.encoder = EncoderRNN(source_vocab_size, hidden_size)
        self.decoder = AttnDecoderRNN('general', hidden_size, target_vocab_size)

    def forward(self, src, trg, length=60):
        # encoder
        enc_outputs, enc_status = self.encoder(src)
        last_status = enc_status
        # decoder
        labels, dec_outputs, attn_weights = self.decoder(trg, last_status, enc_outputs, src, length)
        return labels, dec_outputs, attn_weights
        
class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, n_layers=1, dropout_p=0.1):
        super(EncoderRNN, self).__init__()

        # Keep parameters
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.dropout_p = dropout_p

        # Define layers
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.lstm_fwd1 = LSTMCell(hidden_size, hidden_size, dropout=dropout_p)
        self.lstm_fwd2 = LSTMCell(hidden_size, hidden_size, dropout=dropout_p)
        self.lstm_bwd1 = LSTMCell(hidden_size, hidden_size, dropout=dropout_p)
        self.lstm_bwd2 = LSTMCell(hidden_size, hidden_size, dropout=dropout_p)
        self.dropout = nn.Dropout(p=dropout_p)

    def forward(self, word_inputs):
        # Note: we run this all at once (over the whole input sequence)
        embedded_words = self.dropout(self.embedding(word_inputs))# L x B x H
        
        fstatus1, bstatus1, fstatus2, bstatus2 = None, None, None, None
        fhs, bhs = [], []
        for i in range(len(embedded_words)):
            # 入力を取得
            fembed, fmask = embedded_words[i], word_inputs[i] == 1
            bembed, bmask = embedded_words[-i-1], word_inputs[-i-1] == 1
            # 順方向LSTM
            finput = fembed
            fstatus1 = self.lstm_fwd1(finput, fstatus1, fmask)
            finput = fstatus1[0]
            fstatus2 = self.lstm_fwd2(finput, fstatus2, fmask)
            # 逆方向LSTM
            binput = bembed
            bstatus1 = self.lstm_bwd1(binput, bstatus1, bmask)
            binput = bstatus1[0]
            bstatus2 = self.lstm_bwd2(binput, bstatus2, bmask)
            # 状態を保存
            fhs.append(fstatus2[0])
            bhs.append(bstatus2[0])

        outputs = []
        for fwd, bwd in zip(fhs, reversed(bhs)):
            outputs.append(fwd + bwd / 2)
        outputs = torch.stack(outputs, dim=0)
        return outputs, (bstatus1, bstatus2) #output:L x B x H

class AttnDecoderRNN(nn.Module):
    def __init__(self, attn_model, hidden_size, output_size, n_layers=1, dropout_p=0.1):
        super(AttnDecoderRNN, self).__init__()
        
        # Keep parameters
        self.attn_model = attn_model
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout_p = dropout_p

        # Define layers
        self.embedding = nn.Embedding(output_size, hidden_size)
        #self.lstm1 = LSTMCell(hidden_size, hidden_size, dropout=dropout_p)
        self.lstm1 = LSTMCell(hidden_size * 2, hidden_size, dropout=dropout_p)
        self.lstm2 = LSTMCell(hidden_size, hidden_size, dropout=dropout_p)
        self.out = nn.Linear(hidden_size * 2, output_size)
        self.dropout = nn.Dropout(p=dropout_p)
        # Choose attention model
        if attn_model is not None:
            self.attn = Attn(attn_model, hidden_size)

    def forward(self, word_inputs, last_status, encoder_outputs, src_inputs, length=60):
        src_inputs = src_inputs.transpose(1, 0).unsqueeze(2)
        l = len(word_inputs) if word_inputs is not None else length
        if word_inputs is None:
            sos_id = 2
            batch_size = src_inputs.size()[0]
            device = encoder_outputs.device
            sos = torch.full((batch_size,), sos_id, dtype=torch.int64, device=device)
            word_output = sos
        status1, status2 = last_status
        context = None
        labels, outputs, attn_weights = [], [], []
        for i in range(l):
            word_input = word_inputs[i] if word_inputs is not None else word_output
            output, context, status1, status2, attn = self.forward_one(
                word_input, context, status1, status2 , encoder_outputs, src_inputs)
            outputs.append(output)
            attn_weights.append(attn)
            word_output = torch.argmax(output, dim=1)
            labels.append(word_output)
        attn_weights = torch.stack(attn_weights, dim=0)
        outputs      = torch.stack(outputs, dim=0)
        labels       = torch.stack(labels, dim=0)
        return labels, outputs, attn_weights

    def forward_one(self, word_input, last_context, status1, status2, encoder_outputs, src_inputs):
        # Get the embedding of the current input word (last output word)
        word_embedded = self.embedding(word_input) # B x H

        # Combine embedded input word and last context, run through RNN
        if last_context is None:
            last_context = torch.zeros_like(word_embedded)
        
        rnn_input = self.dropout(torch.cat((word_embedded, last_context), 1))
        status1 = self.lstm1(rnn_input, status1)
        rnn_input = status1[0]
        status2 = self.lstm2(rnn_input, status2)
        rnn_output = torch.unsqueeze(status2[0], dim=0) # S=1 x B x H

        # Calculate attention from current RNN state and all encoder outputs; apply to encoder outputs
        attn_weights = self.attn(rnn_output, encoder_outputs, src_inputs)
        attn_weights = attn_weights.transpose(1, 2)           # B x L x S=1 -> B x S=1 x L
        encoder_outputs = encoder_outputs.transpose(0, 1)     # L x B x H -> B x L x H
        context = torch.matmul(attn_weights, encoder_outputs) # B x S=1 x H

        # Final output layer (next word prediction) using the RNN hidden state and context vector
        rnn_output = rnn_output.squeeze(0) # S=1 x B x H -> B x H
        context = context.squeeze(1)       # B x S=1 x H -> B x H
        output = self.out(self.dropout(torch.cat((rnn_output, context), 1)))

        # Return final output, hidden state, and attention weights (for visualization)
        return output, context, status1, status2, attn_weights

class Attn(nn.Module):
    def __init__(self, method, hidden_size):
        super(Attn, self).__init__()
        
        self.method = method
        self.hidden_size = hidden_size

    def forward(self, hidden, encoder_outputs, src_inputs):
        hidden = hidden.transpose(0, 1) # 1 x B x H -> B x 1 x H 
        hidden = hidden.transpose(1, 2) # B x 1 x H -> B x H x 1
        encoder_outputs = encoder_outputs.transpose(0, 1) # L x B x H -> B x L x H
        score = torch.matmul(encoder_outputs, hidden)     # B x L x 1
        # Decrease score of padding
        score = torch.where(src_inputs == 1, torch.full_like(score, -700), score)
        # Normalize score to weights in range 0 to 1
        weight = F.softmax(score, dim=1)
        return weight

class LSTMCell(nn.LSTMCell):
    def __init__(self, input_size, hidden_size, dropout=None, **kwargs):
        super(LSTMCell, self).__init__(input_size, hidden_size, **kwargs)
        self.dropout = dropout
        self.dropout = nn.Dropout(p=dropout)
    
    def __call__(self, input, hx, mask=None):
        input = self.dropout(input)
        if hx is None:
            device = input.device
            batch_size = input.size()[0]
            zeros = torch.zeros((batch_size, self.hidden_size), device=device) # B x H
            hx = (zeros, zeros)
        old_hx = hx
        hx = super(LSTMCell, self).__call__(input, hx)
        if mask is not None:
            mask = torch.unsqueeze(mask, dim=1)
            hx = (torch.where(mask, old_hx[0], hx[0]),
                  torch.where(mask, old_hx[1], hx[1]))
        return hx
