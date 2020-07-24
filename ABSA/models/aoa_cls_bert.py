from layers.dynamic_rnn import DynamicLSTM
from layers.attention import Attention, NoQueryAttention
from layers.point_wise_feed_forward import PositionwiseFeedForward
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class AOA_CLS_BERT(nn.Module):
    def __init__(self, bert, opt):
        super(AOA_CLS_BERT, self).__init__()
        self.opt = opt
        self.bert = bert
        self.dropout = nn.Dropout(opt.dropout)

        self.dense = nn.Linear(2 * opt.bert_dim, opt.polarities_dim)

    def forward(self, inputs):
        text_bert_indices, bert_segments_ids = inputs[0], inputs[1] # batch_size x seq_len
        ctx_len = torch.sum(text_bert_indices[:, :-5] != 0, dim=1)
        asp_len = torch.sum(text_bert_indices[:, -5:] != 0, dim=1)
        bert_out, pooled_output = self.bert(text_bert_indices, bert_segments_ids)
        ctx, asp = torch.split(bert_out, self.opt.max_seq_len+1, dim=1)
        ctx = self.dropout(ctx) # batch_size x (ctx) seq_len x bert_dim
        asp = self.dropout(asp) # batch_size x (asp) seq_len x bert_dim

        interaction_mat = torch.matmul(ctx, torch.transpose(asp, 1, 2)) # batch_size x (ctx) seq_len x (asp) seq_len
        alpha = F.softmax(interaction_mat, dim=1) # col-wise, batch_size x (ctx) seq_len x (asp) seq_len
        beta = F.softmax(interaction_mat, dim=2) # row-wise, batch_size x (ctx) seq_len x (asp) seq_len
        beta_avg = beta.mean(dim=1, keepdim=True) # batch_size x 1 x (asp) seq_len
        gamma = torch.matmul(alpha, beta_avg.transpose(1, 2)) # batch_size x (ctx) seq_len x 1
        weighted_sum = torch.matmul(torch.transpose(ctx, 1, 2), gamma).squeeze(-1) # batch_size x 2*hidden_dim
        out = torch.cat((weighted_sum, pooled_output), dim=1)

        logits = self.dense(out) # batch_size x polarity_dim

        return logits