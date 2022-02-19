import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.functional import conv2d

from src.config import w, h, CHANNEL_N, ALIVE_THRESHOLD

class Net(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.device = device
        self.fire_rate = 0.5
        self.conv1 = nn.Conv2d(CHANNEL_N*3, 128, 1)
        self.conv2 = nn.Conv2d(128, 16, 1)
        
        n_trainable_params =  sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f'Number of trainable parameters: {n_trainable_params}')
    
    def forward(self, x):
        pre_life_mask = get_living_mask(x)
        y = self.perceive(x)
        dx = self.cnn(y)
        update_mask = (torch.rand(1,CHANNEL_N,w,h) > self.fire_rate).to(self.device)
        x = x + dx * update_mask
        post_life_mask = get_living_mask(x)
        life_mask = pre_life_mask & post_life_mask
        return x * life_mask

    def cnn(self, x):
        x = F.relu(self.conv1(x))
        x = self.conv2(x)
        return x
    
    def perceive(self, x):
        kernel_h = torch.tensor([[-1., 0., 1.],
                                 [-2., 0., 2.],
                                 [-1., 0., 1.]]).to(self.device)
        kernel_h /= 8
        kernel_v = kernel_h.T.to(self.device)
        #x_pad = torch.nn.functional.pad(x, [1, 1, 1, 1], 'circular')
        # depth-wise 2D convolution (depth-wise because groups = CHANNEL_N)
        grad_h = conv2d(x, kernel_h.expand(16, 1, 3, 3), padding='same', groups = CHANNEL_N)
        grad_v = conv2d(x, kernel_v.expand(16, 1, 3, 3), padding='same', groups = CHANNEL_N)
        perception_grid = torch.cat((x, grad_h, grad_v),1)
        return perception_grid
        
    
    def call(self, x):
        pre_life_mask = get_living_mask(x)
        y = self.perceive(x)
        dx = self(y)
        update_mask = (torch.rand(1,CHANNEL_N,w,h) > self.fire_rate).to(self.device)
        x = x + dx * update_mask
        
        post_life_mask = get_living_mask(x)
        life_mask = pre_life_mask & post_life_mask
        return x * life_mask
    
    
def get_living_mask(x):
    """
    Set all channels of empty cells to zero.
    A cell is 'empty' when there is no mature 
    (alpha > 0.1) cell in its 3x3 neighborhood.
      * Alpha is at position 3 of the state *
    """
    living_mask = F.max_pool2d(x[:,3,:,:], 3, stride=1, padding=1) >= ALIVE_THRESHOLD
    # repeat along the state axis
    living_mask = torch.repeat_interleave(living_mask[:,None,:,:], CHANNEL_N, 1)
    return living_mask