# Swin-Transformer Notes

1. The model starts by splitting an image into p x p non-overlapping patches with a linear embedding exactly like ViT.
2. Our image transforms from (h,w,c) to (h/p,w/p,c*p**2) from patch partitioning.
3. Then to (h/p * w/p, C) after the linear projection
4. We treat the h*w patches as the tokens of the transformer sequence and C as our embedding dimension.


### Shifted Window Multihead Self Attention.
1. We start by initializing our parameters embed_dim, num_heads, and and window_size and defining two linear projections. 
2. The first is our projection from inputs to Queries, Keys, and Values which we do in one parallel projection so the output size is set to 3*C.
3. The second projection is a linear projection applied after the attention computation. This projection is for communication between the concatenated parallel multi-headed attention units.
4. We then do our Q,K,V projection on our input of shape ((h*w), c) to ((h*w), 3C).
5. Our next step is in two parts where we will rearrange our input ((h*w), C*3) into windows and parallel attention heads for our attention computation.

The idea of shifted window attention is that on alternating attention computation layers we shift our windows so that our shifted windows overlap over the previous layers windows to allow cross window communication for the model. We can achieve this efficiently by a cyclic shift as depicted in the image above. PyTorch has a function torch.roll we can use that will perform our cyclic shift of size window_size/2 on our input.

This is great! However one issue will come from using the cyclic shift to perform shifted window attention. Because we are now computing attention on the new shifted image, the model will be confused on where sections A,B,C in the above example actually belong in the image. We can solve this issue by masking the communication between tokens that should not be next to each other in the original image.

Because we will shift our image size by window_size/2, only half of the last row and column of windows will be affected. For this reason, we can apply the left mask shown below to all of the last rows and the mask on the right to all of the last columns of the image.