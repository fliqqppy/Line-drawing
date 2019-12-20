import torch
import torch.nn as nn
from torch.nn import functional as F
from torchvision import transforms

import time
import numbers
import math
import numpy as np
import argparse
from PIL import Image
from rcf import RCF
from utils import prepare_image_PIL
from ecn import ECN, BasicBlock, Bottleneck
from utils import gaussian_smooth


device = 'cuda' if torch.cuda.is_available() else 'cpu'


class GaussianSmoothing(nn.Module):
    def __init__(self, channels, kernel_size, sigma):
        super(GaussianSmoothing, self).__init__()
        if isinstance(kernel_size, numbers.Number):
            kernel_size = [kernel_size] * 2
        if isinstance(sigma, numbers.Number):
            sigma = [sigma] * 2

        kernel = 1
        meshgrids = torch.meshgrid(
            [
                torch.arange(size, dtype=torch.float32)
                for size in kernel_size
            ]
        )
        for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
            mean = (size - 1) / 2
            kernel *= 1 / (std * math.sqrt(2 * math.pi)) * torch.exp(-((mgrid - mean) / std) ** 2 / 2)

        # Make sure sum of values in gaussian kernel equals 1.
        kernel = kernel / torch.sum(kernel)

        # Reshape to depthwise convolutional weight
        kernel = kernel.view(1, 1, *kernel.size())
        kernel = kernel.repeat(channels, *[1] * (kernel.dim() - 1))

        self.register_buffer('weight', kernel)
        self.groups = channels
        self.conv = F.conv2d

    def forward(self, input):
        return self.conv(input, weight=self.weight, groups=self.groups)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--rcf', type=str, default='models/RCFcheckpoint_epoch12.pth',
                        help='rcf model file')
    parser.add_argument('--ecn', type=str, default='models/ecn_rcf_tl.tar',
                        help='ecn model file')
    parser.add_argument('--image', type=str, default='test/1.jpg', help='test image file')
    parser.add_argument('--sigma', type=float, default=1.5, help='sigma for Gaussian filter')
    parser.add_argument('--save_edge', type=bool, default=True, help='whether save edge map')
    parser.add_argument('--edge_name', type=str, default='results/1_edge.png', help='rcf output name')
    parser.add_argument('--output', type=str, default='results/1_t_line.png', help='final output name')
    args = parser.parse_args()

    # load models
    rcf = RCF().to(device)
    checkpoint = torch.load(args.rcf)
    rcf.load_state_dict(checkpoint['state_dict'])
    rcf.eval()

    ecn = ECN(en_block=Bottleneck, de_block=BasicBlock, zero_init_residual=False).to(device)
    model = torch.load(args.ecn)
    state = model['model_state_dict']
    ecn.load_state_dict(state)
    ecn.eval()

    # load image
    image = Image.open(args.image).convert('RGB')
    r_w = image.size[0] - image.size[0] % 16
    r_h = image.size[1] - image.size[1] % 16
    image = transforms.CenterCrop((r_h, r_w))(image)
    image.save(args.image)
    image = np.array(image, dtype='float32')
    image = prepare_image_PIL(image)
    image = torch.tensor(image).unsqueeze(0).to(device)

    # load Gaussian filter
    smooth = GaussianSmoothing(1, 5, args.sigma).to(device)

    # run
    start = time.time()
    rcf_outputs = rcf(image)
    rcf_fin = torch.squeeze(rcf_outputs[-1].detach())
    blank = torch.ones(rcf_fin.shape[0], rcf_fin.shape[1]).to(device)
    edge_map = ((blank - rcf_fin) * 255).type(torch.uint8)

    if args.save_edge:
        edge_map = edge_map.cpu().numpy()
        edge_map = Image.fromarray(edge_map)
        edge_map.save(args.edge_name)

        smoothed_edge = gaussian_smooth(edge_map, size=5, sigma=args.sigma)
        in_edge = transforms.ToTensor()(smoothed_edge)
        in_edge = in_edge.unsqueeze(0).to(device)
    else:
        edge_map = edge_map.float().expand(1, 1, -1, -1)
        smoothed_edge = smooth(edge_map)
        in_edge = smoothed_edge / 255

    line = ecn(in_edge)
    line_fin = line.squeeze(0).detach().cpu()
    result_fin = (line_fin.clamp(0.0, 1.0).numpy().transpose(1, 2, 0)[:, :, 0] * 255.0).astype(np.uint8)
    out_image = Image.fromarray(result_fin)
    # out_image.show()
    out_image.save(args.output)

    end = time.time()
    print('time cost:' + str(end-start))





