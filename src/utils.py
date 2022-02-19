import PIL.Image
import requests
import io
import numpy as np
import torch

from src.config import w, h, TARGET_SIZE, CHANNEL_N, POOL_SIZE, BATCH_SIZE

def load_image(url, max_size=TARGET_SIZE):
    r = requests.get(url)
    img = PIL.Image.open(io.BytesIO(r.content))
    img.thumbnail((max_size, max_size), PIL.Image.ANTIALIAS)
    img = np.float32(img)/255.0
    # premultiply RGB by Alpha
    img[..., :3] *= img[..., 3:]
    return img

def load_emoji(emoji):
    code = hex(ord(emoji))[2:].lower()
    url = 'https://github.com/googlefonts/noto-emoji/blob/main/png/128/emoji_u%s.png?raw=true'%code
    return load_image(url)

def pad_img(img):
    wi, hi, _ = img.shape
    # floor and ceil to handle odd numbers
    dw_r = (w - wi)//2
    dw_l = -(-(w - wi)//2)
    dh_t = (h - hi)//2
    dh_b = -(-(h - hi)//2)
    
    return np.pad(img, ((dw_l,dw_r),(dh_t,dh_b), (0,0)))

def get_alpha(x):
    return torch.clamp(x[:,3,:,:], 0.0, 1.0)

def get_rgba(x):
    return x[:,:4,:,:].T#.cpu().detach().numpy()

def get_rgb(x):
    # Assume rgb premultiplied by alpha
    rgb, a = x[:,:3,:,:], get_alpha(x)
    return (1.0-a+rgb).T

def process_img(x, zoom = 5):
    clipped_x = np.clip(x[0,:4,...].T.detach().cpu().clone().numpy(),0,1)
    clipped_x = np.uint8(clipped_x*255)
    clipped_x = clipped_x.repeat(zoom, 0).repeat(zoom, 1)
    return clipped_x


def init_grid(device):
    # create empty grid
    x = torch.zeros((1, CHANNEL_N, w, h))
    # Set all state values of the center pixel to 1,
    # except RGB to 0, so that the pixels is black
    x[:,3:, w//2, h//2] = 1
    return x.to(device)

def damage_masks(n, device):
    x = torch.linspace(-1.0, 1.0, w)[None, None, :]
    y = torch.linspace(-1.0, 1.0, h)[None, :, None]
    center = torch.rand(2,n,1,1)-0.5
    r = (0.1-0.4)*torch.rand(n, 1, 1)+0.4
    x, y = (x-center[0])/r, (y-center[1])/r
    mask = x*x+y*y < 1.0
    return mask[:,None,...].to(device)

class SamplePool:
    def __init__(self, x):
        self.x = x
    
    def sample(self):
        return self.x[torch.randint(0, POOL_SIZE, (BATCH_SIZE,))]