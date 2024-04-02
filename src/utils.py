"""
This file contains details about the helper functions required to build a Swin-Transformer.
"""
import torch 
import torch.nn as nn
import torch.nn.functional as F
import math
from einops import rearrange
#------------------------------------------------------------------------------------------------------#
class SwinEmbedding(nn.Module):
    """
    Acts as an Embedding function to convert input RGB images into patches.
    Args:
        Input : RGB image with 96*96 image dimensions (size: b,c,h,w)
        Output: Patche vectors with dimensions (size: b,(h/4 * w/4),c)
    """
    def __init__(self,patch_size=4,C=96):
        super().__init__()
        self.linear_embedding = nn.Conv2d(in_channels=3,out_channels=C,kernel_size=patch_size,stride=patch_size)
        self.layer_norm = nn.LayerNorm(C)
        self.relu = nn.ReLU()

    def forward(self,x):
        x = self.linear_embedding(x)
        x = rearrange(x,'b c h w -> b (h w) c')
        x = self.relu(self.layer_norm(x))
        return x
    
#------------------------------------------------------------------------------------------------------#
class PatchMerging(nn.Module):
    """
    Acts as the merging block to reduce the number of patches in each deeper layer.
    Args:
        Input: Patch (b, (h*w), C)
        Output: Patch (b, (h/2 * w/2), C*2)
    """
    def __init__(self,C):
        super().__init__()
        self.linear = nn.Linear(in_features=4*C,out_features=2*C)
        self.layer_norm = nn.LayerNorm(2*C)

    def forward(self,x):
        height = width = int(math.sqrt(x.shape[1])/2)
        x = rearrange(x, 'b (h s1 w s2) c -> b (h w) (s2 s1 c)',s1=2, s2=2, h=height, w=width)
        x = self.layer_norm(self.linear(x))
        return x
    
#-------------------------------------------------------------------------------------------------------#
class RelativeEmbeddings(nn.Module):
    """
    Module to incorporate relative position bias into self-attention computation.
    The relative position bias (B) is added to each head in computing similarity:
    Attention(Q, K, V) = SoftMax((QK^T) / sqrt(d) + B) * V,
    and M is the number of patches in a window.

    Since the relative position along each axis lies in the range [-M + 1, M - 1],
    we parameterize a smaller-sized bias matrix B_hat, and values in B are taken from B_hat.
    """
    def __init__(self, window_size=7):
        super().__init__()
        # Initialize a learnable parameter B
        B = nn.Parameter(torch.randn(2 * window_size - 1, 2 * window_size - 1))
        # Generate relative positions along x and y axes
        x = torch.arange(1, window_size + 1, 1 / window_size)
        x = (x[None, :] - x[:, None]).int()
        y = torch.cat([torch.arange(1, window_size + 1)] * window_size)
        y = (y[None, :] - y[:, None])
        # Take values from B_hat based on relative positions
        self.embeddings = nn.Parameter(B[x[:, :], y[:, :]], requires_grad=False)

    def forward(self, x):
        # Add relative position bias to input
        return x + self.embeddings
    
#-------------------------------------------------------------------------------------------------------#
class SwinEncoderBlock(nn.Module):
    def __init__(self,embed_dim, num_heads, window_size, mask):
        super().__init__()

        self.layer_norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(0.1)
        self.WMSA = ShiftedWindowMSA(embed_dim=embed_dim, num_heads=num_heads, 
                                     window_size=window_size, mask=mask)
        self.MLP1 = nn.Sequential(
            nn.Linear(embed_dim, embed_dim*4),
            nn.GELU(),
            nn.Linear(embed_dim*4, embed_dim)
        )

    def forward(self, x):
        height, width = x.shape[1:3]
        res1 = self.dropout(self.WMSA(self.layer_norm(x)) + x)
        x = self.layer_norm(res1)
        x = self.MLP1(x)
        return self.dropout(x + res1)

#-------------------------------------------------------------------------------------------------------#    
class AlternatingEncoderBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, window_size=7):
        super().__init__()
        self.WSA = SwinEncoderBlock(embed_dim=embed_dim, num_heads=num_heads, window_size=window_size, mask=False)
        self.SWSA = SwinEncoderBlock(embed_dim=embed_dim, num_heads=num_heads, window_size=window_size, mask=True)
    
    def forward(self, x):
        return self.SWSA(self.WSA(x))
    
#-------------------------------------------------------------------------------------------------------#   
"""
The idea of shifted window attention is that on alternating attention computation layers we shift our windows so that our shifted windows overlap over the previous layers windows to allow cross window communication for the model. We can achieve this efficiently by a cyclic shift as depicted in the image above. PyTorch has a function torch.roll we can use that will perform our cyclic shift of size window_size/2 on our input.
This is great! However one issue will come from using the cyclic shift to perform shifted window attention. Because we are now computing attention on the new shifted image, the model will be confused on where sections A,B,C in the above example actually belong in the image. We can solve this issue by masking the communication between tokens that should not be next to each other in the original image.
Because we will shift our image size by window_size/2, only half of the last row and column of windows will be affected. For this reason, we can apply the left mask shown below to all of the last rows and the mask on the right to all of the last columns of the image.
"""
class ShiftedWindowMSA(nn.Module):
    """
    Shifted window self attention module.
    Detailed description can be found in the 'Notes' section under SWMSA.
    """
    def __init__(self, embed_dim, num_heads, window_size=7, mask=False):
        super().__init__()
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