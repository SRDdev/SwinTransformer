"""
This file contains the detailed explantion and implementation of SWMSA


The idea of shifted window attention is that on alternating attention computation layers we shift our windows so that our shifted windows overlap over the previous layers windows to allow cross window communication for the model. We can achieve this efficiently by a cyclic shift as depicted in the image above. PyTorch has a function torch.roll we can use that will perform our cyclic shift of size window_size/2 on our input.
This is great! However one issue will come from using the cyclic shift to perform shifted window attention. Because we are now computing attention on the new shifted image, the model will be confused on where sections A,B,C in the above example actually belong in the image. We can solve this issue by masking the communication between tokens that should not be next to each other in the original image.
Because we will shift our image size by window_size/2, only half of the last row and column of windows will be affected. For this reason, we can apply the left mask shown below to all of the last rows and the mask on the right to all of the last columns of the image.
"""

import torch
from torch import nn
import math
from einops import rearrange
from torch.nn import functional as F
from utils import RelativeEmbeddings

class ShiftedWindowMSA(nn.Module):
    """
    Shifted window self attention module.
    Detailed description can be found in the 'Notes' section under SWMSA.
    """
    def __init__(self, embed_dim, num_heads, window_size=7, mask=False):
        super(ShiftedWindowMSA, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.mask = mask

        # Projection layer for keys, queries, and values
        self.proj1 = nn.Linear(in_features=embed_dim, out_features=embed_dim*3)
        # Projection layer for output
        self.proj2 = nn.Linear(in_features=embed_dim, out_features=embed_dim)
        self.embeddings = RelativeEmbeddings()

    def forward(self, x):
        # Calculate height and width of input feature map
        height = width = int(math.sqrt(x.shape[1]))
        # Project input to keys, queries, and values
        x = self.proj1(x)
        # Rearrange input for multi-head self attention
        x = rearrange(x, 'b (h w) (c K) -> b h w c K', K=3, h=height, w=width)

        # Shifted window masking
        if self.mask:
            x = torch.roll(x, (-self.window_size//2, -self.window_size//2), dims=(1,2))

        # Further rearrange for applying shifted window self attention
        x = rearrange(x, 'b (h m1) (w m2) (H E) K -> b H h w (m1 m2) E K', H=self.num_heads, m1=self.window_size, m2=self.window_size)
        Q, K, V = x.chunk(3, dim=6)
        Q, K, V = Q.squeeze(-1), K.squeeze(-1), V.squeeze(-1)
        # Calculate attention scores
        att_scores = (Q @ K.transpose(4,5)) / math.sqrt(self.embed_dim / self.num_heads)

        # Apply row and column masks for the last rows and columns of windows
        '''
          shape of att_scores = (b, H, h, w, (m1*m2), (m1*m2))
          we simply have to generate our row/column masks and apply them
          to the last row and columns of windows which are [:,:,-1,:] and [:,:,:,-1]
        '''
        if self.mask:
            row_mask = torch.zeros((self.window_size**2, self.window_size**2)).cuda()
            row_mask[-self.window_size * (self.window_size//2):, 0:-self.window_size * (self.window_size//2)] = float('-inf')
            row_mask[0:-self.window_size * (self.window_size//2), -self.window_size * (self.window_size//2):] = float('-inf')
            column_mask = rearrange(row_mask, '(r w1) (c w2) -> (w1 r) (w2 c)', w1=self.window_size, w2=self.window_size).cuda()
            att_scores[:, :, -1, :] += row_mask
            att_scores[:, :, :, -1] += column_mask

        # Apply softmax to get attention weights and multiply with values
        att = F.softmax(att_scores, dim=-1) @ V
        # Rearrange back to original shape
        x = rearrange(att, 'b H h w (m1 m2) E -> b (h m1) (w m2) (H E)', m1=self.window_size, m2=self.window_size)

        # Reverse shifting if masking was applied
        if self.mask:
            x = torch.roll(x, (self.window_size//2, self.window_size//2), (1,2))

        # Rearrange back to original shape and project to final output
        x = rearrange(x, 'b h w c -> b (h w) c')
        return self.proj2(x)