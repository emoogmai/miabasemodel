import torch
import torch.nn as nn
import numpy as np

class MiaSelfAttention(nn.Module):
    def __init__(self, input_dim, output_dim, context_length, dropout_rate, qkv_bias=False) -> None:
        super().__init__()
        self.sqrt_headdim = np.sqrt(output_dim)
        self.output_dim = output_dim
        #The attention mechanism is an algorithm of 4 steps, basically computes a formula, here we create
        #required weight matrix needed in such formula
        self.Q_W = nn.Linear(input_dim, output_dim, qkv_bias)
        self.K_W = nn.Linear(input_dim, output_dim, qkv_bias)
        self.V_W = nn.Linear(input_dim, output_dim, qkv_bias)
        # Helper layers to have stable trainning
        self.dropout_layer = nn.Dropout(dropout_rate)
        self.register_buffer('mask', torch.triu(torch.ones(context_length, context_length), diagonal=1))

    def forward(self, input_x):
        # This is required to mask those tokens we don't want network sees
        batch_size, total_tokens, input_dim = input_x.shape
        # A pre-requisite step is to obtain Q, K, and V matrix by multiplying input X (embeddings of each token in context)
        Q = self.Q_W(input_x)
        K = self.K_W(input_x)
        V = self.V_W(input_x)

        # Fist step is compute the dot product between Q and K transpose matrix. This is equivalent to compare
        # each token (input word) with all other ones in the provided context
        attention = Q @ K.transpose(1,2) # only transpose the sequence no the batch dimension !!!
        attention.masked_fill_(self.mask.bool()[:total_tokens, :total_tokens], -torch.inf) #masked input

        # Second step is divide computed attention scores between square root of the dimensionality of the model
        # embeddings.
        # Third step is normalize similarity scores (attention matrix) using softmax function. Value obtained from
        # this computation should be between 0 and 1 and sum out 1
        #norm_attention_weights = torch.softmax(attention / K.shape[-1]**0.5, dim=-1)
        norm_attention_weights = torch.softmax(attention / self.sqrt_headdim, dim=-1)

        norm_attention_weights = self.dropout_layer(norm_attention_weights)

        # Four and last step is compute the attention matrix Z
        Z = norm_attention_weights @ V

        return Z

class MiaMultiHeadAttention(nn.Module):
    def __init__(self, input_dim, output_dim, context_length, total_heads, dropout_rate, qkv_bias=False) -> None:
        super().__init__()

        assert output_dim % total_heads == 0, "output_dim % total_heads must be zero. Values does not match"

        head_dim = output_dim // total_heads

        self.attention_heads = nn.ModuleList([MiaSelfAttention(input_dim, head_dim , context_length, dropout_rate, qkv_bias)
                                                 for _ in range(total_heads)])

        self.out_proj = nn.Linear(head_dim*total_heads, head_dim*total_heads)
                                         

    def forward(self, input_x):
        Z = torch.cat([cur_att_head(input_x) for cur_att_head in self.attention_heads], dim=-1)                                             

        return self.out_proj(Z)


