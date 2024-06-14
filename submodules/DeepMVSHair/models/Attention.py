import torch
import torch.nn as nn
import torch.nn.functional as F

class Attention(nn.Module):

    def __init__(self, feat_dim, embed_dim):
        super(Attention, self).__init__()

        self.mat_Q = nn.Conv1d(feat_dim, embed_dim, kernel_size=(1,))
        self.mat_K = nn.Conv1d(feat_dim, embed_dim, kernel_size=(1,))
        self.mat_V = nn.Conv1d(feat_dim, embed_dim, kernel_size=(1,))

        self.multi_attn = nn.MultiheadAttention(embed_dim, num_heads=16)

    def forward(self, feat):
        '''

        :param feat: [N, C_f, V]
        :return:
        '''

        # [N, C_e, V] -> [N, V, C_e] -> [V, N, C_e]
        # V: views, equal to sequence length
        Q = self.mat_Q(feat).transpose(1, 2).transpose(0, 1)
        K = self.mat_K(feat).transpose(1, 2).transpose(0, 1)
        V = self.mat_V(feat).transpose(1, 2).transpose(0, 1)

        attn_output, _ = self.multi_attn(Q, K, V)

        return attn_output
