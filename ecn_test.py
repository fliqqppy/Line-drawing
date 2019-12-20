from PIL import Image, ImageOps
import numpy
import torch
from torchvision import transforms, utils
from ecn import ECN, BasicBlock, Bottleneck
from utils import gaussian_smooth


def main():
    # load model
    ecn = ECN(en_block=Bottleneck, de_block=BasicBlock, zero_init_residual=False).cuda()
    model = torch.load('checkpoints/ecn_rcf_tl_ckpt_4.tar')
    state = model['model_state_dict']
    ecn.load_state_dict(state)
    ecn.eval()

    # for i in range(1, 6):
    image = Image.open('test/1.png').convert('L')
    image = gaussian_smooth(image, size=5, sigma=1.5)

    image = transforms.ToTensor()(image)
    image = image.unsqueeze(0).cuda()

    result = ecn(image)
    result_fin = result.squeeze(0).detach().cpu()
    # utils.save_image(result_fin, config['save_path'])
    result_fin = (result_fin.clamp(0.0, 1.0).numpy().transpose(1, 2, 0)[:, :, 0] * 255.0).astype(numpy.uint8)
    out_image = Image.fromarray(result_fin)
    out_image.show()
    # out_image.save(config['save_path'])


if __name__ == '__main__':
    main()
