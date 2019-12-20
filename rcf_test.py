from PIL import Image
import numpy as np
import torch
import torchvision
from rcf import RCF
from utils import prepare_image_PIL


if __name__ == '__main__':
    rcf = RCF().cuda()
    checkpoint = torch.load('models/RCFcheckpoint_epoch12.pth')
    rcf.load_state_dict(checkpoint['state_dict'])
    rcf.eval()

    # for i in range(1, 6):
    image = Image.open('rcf-data/ann.png').convert('RGB')
    image = np.array(image, dtype='float32')
    image = prepare_image_PIL(image)
    image = torch.tensor(image).unsqueeze(0).cuda()
    _, _, H, W = image.shape

    results = rcf(image)
    result = torch.squeeze(results[-1].detach()).cpu().numpy()

    # results_all = torch.zeros((len(results), 1, H, W))
    # for i in range(len(results)):
    #     results_all[i, 0, :, :] = results[i]
    # torchvision.utils.save_image(1 - results_all, 'results/bear_out_r35.png')

    result = Image.fromarray(((1 - result) * 255).astype(np.uint8))
    result.save('rcf-data/rcf.png')




