# import torch
# import torch.nn as nn
#
# from generate.layers import SelfAttention
#
# class Transformer(nn.Module):
#
#     def __init__(self, src_vocab_dim, trg_vocab_dim, src_pad_idx, trg_pad_idx, embed_dim=256, num_layers=6, num_heads=8, forward_expansion=4, dropout=0.1, max_length=100, device='cpu'):
#         super(Transformer, self).__init__()
#         self.encoder = TEncoder(src_vocab_dim, embed_dim, num_layers, num_heads, forward_expansion, dropout, max_length, device)
#         self.decoder = TDecoder(trg_vocab_dim, embed_dim, num_layers, num_heads, forward_expansion, dropout, max_length, device)
#         self.src_pad_idx = src_pad_idx
#         self.trg_pad_idx = trg_pad_idx
#         self.device = device
#
#     def _make_src_mask(self, src):
#         src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2) # (N, 1, 1, |src|)
#         return src_mask.to(self.device)
#
#     def _make_trg_mask(self, trg):
#         N, trg_len = trg.shape
#         # Need triangular matrix
#         trg_mask = torch.tril(torch.ones((trg_len, trg_len))).expand(N, 1, trg_len, trg_len)
#         return trg_mask.to(self.device)
#
#     def forward(self, src, trg):
#         src_mask = self._make_src_mask(src)
#         trg_mask = self._make_trg_mask(trg)
#         enc_src = self.encoder(src, src_mask)
#         out = self.decoder(trg, enc_src, src_mask, trg_mask)
#         return out
#
# class TEncoder(nn.Module):
#
#     def __init__(self, src_vocab_dim, embed_dim, num_layers, num_heads, forward_expansion, dropout, max_length, device):
#         super(TEncoder, self).__init__()
#         self.embed_dim = embed_dim
#         self.device = device
#         self.word_embedding = nn.Embedding(src_vocab_dim, embed_dim)
#         self.position_embedding = nn.Embedding(max_length, embed_dim)
#         self.layers = nn.ModuleList([
#             TransformerBlock(embed_dim, num_heads, forward_expansion, dropout)
#             for _ in range(num_layers)
#         ])
#         self.dropout = nn.Dropout(dropout)
#
#     def forward(self, x, mask):
#         N, seq_length = x.shape
#         positions = torch.arange(0, seq_length).expand(N, seq_length).to(self.device)
#         out = self.dropout(self.word_embedding(x) + self.position_embedding(positions))
#         for layer in self.layers:
#             out = layer(out, out, out, mask)
#         return out
#
# class TDecoder(nn.Module):
#
#     def __init__(self, trg_vocab_dim, embed_dim, num_layers, num_heads, forward_expansion, dropout, max_length, device):
#         super(TDecoder, self).__init__()
#         self.device = device
#         self.word_embedding = nn.Embedding(trg_vocab_dim, embed_dim)
#         self.position_embedding = nn.Embedding(max_length, embed_dim)
#         self.layers = nn.ModuleList([
#             TDecoderBlock(embed_dim, num_heads, forward_expansion, dropout, device)
#             for _ in range(num_layers)
#         ])
#         self.fc_out = nn.Linear(embed_dim, trg_vocab_dim)
#         self.dropout = nn.Dropout(dropout)
#
#     def forward(self, x, enc_out, src_mask, trg_mask):
#         N, seq_length = x.shape
#         positions = torch.arange(0, seq_length).expand(N, seq_length).to(self.device)
#         x = self.dropout(self.word_embedding(x) + self.position_embedding(positions))
#         for layer in self.layers:
#             x = layer(x, enc_out, enc_out, src_mask, trg_mask)
#         out = self.fc_out(x)
#         return out
#
# class TDecoderBlock(nn.Module):
#
#     def __init__(self, embed_dim, num_heads, forward_expansion, dropout, device):
#         super(TDecoderBlock, self).__init__()
#         self.attention = SelfAttention(embed_dim, num_heads)
#         self.norm = nn.LayerNorm(embed_dim)
#         self.transformer_block = TransformerBlock(embed_dim, num_heads, forward_expansion, dropout)
#         self.dropout = nn.Dropout(dropout)
#
#     def forward(self, x, value, key, src_mask, trg_mask):
#         attention = self.attention(x, x, x, mask=trg_mask)
#         query = self.dropout(self.norm(attention + x))
#         out = self.transformer_block(value, key, query, src_mask)
#         return out
#
# class TransformerBlock(nn.Module):
#
#     def __init__(self, embed_dim, num_heads, forward_expansion, dropout):
#         super(TransformerBlock, self).__init__()
#         self.attention = SelfAttention(embed_dim, num_heads)
#         self.norm_1 = nn.LayerNorm(embed_dim)
#         self.feed_forward = nn.Sequential(nn.Linear(embed_dim, forward_expansion*embed_dim),
#                                           nn.ReLU(),
#                                           nn.Linear(forward_expansion*embed_dim, embed_dim))
#         self.norm_2 = nn.LayerNorm(embed_dim)
#         self.dropout = nn.Dropout(dropout)
#
#     def forward(self, value, key, query, mask):
#         attention = self.attention(value, key, query, mask=mask)
#         x = self.dropout(self.norm_1(attention + query))
#         forward = self.feed_forward(x)
#         out = self.dropout(self.norm_2(forward + x))
#         return out