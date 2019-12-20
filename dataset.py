from torch.utils import data
from torch.autograd import Variable
from torchvision import transforms

import pickle
from PIL import Image

from utils import add_gauss_noise, add_pepper_noise


transform = transforms.Compose([transforms.ToTensor()])


class ImageSet(data.Dataset):
    def __init__(self, root, i, trans=transform, noise=False):
        self.line_root = root + '/cropped_lines_{}/'.format(i)
        self.ann_root = root + '/cropped_anns_{}/'.format(i)
        self.noise = noise
        self.transform = trans
        self.line_list = pickle.load(open(root + '/line_names.pkl', 'rb'))
        self.ann_list = pickle.load(open(root + '/ann_names.pkl', 'rb'))

    def __getitem__(self, idx):
        line = Image.open(self.line_root + self.line_list[idx] + '.png').convert('L')
        if self.noise:
            line = add_pepper_noise(line)
        line_tensor = self.transform(line)

        ann = Image.open(self.ann_root + self.ann_list[idx] + '.png').convert('L')
        ann_tensor = self.transform(ann)

        return Variable(line_tensor), Variable(ann_tensor)

    def __len__(self):
        return len(self.line_list)


