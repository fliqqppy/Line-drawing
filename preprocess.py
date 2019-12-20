import os
import pickle
from PIL import Image
from torchvision import transforms


def create_name_list(list_file, path):
    names = []
    for root, dirs, files in os.walk(path):
        for file in files:
            names.append(os.path.splitext(file)[0])
    with open(list_file, 'wb') as f:
        pickle.dump(names, f)


def image_crop(data_root):
    sizes = [800, 600, 400, 224]
    step = len(sizes)
    ext = '.png'
    line_path = data_root
    ann_path = data_root

    for i in range(step):
        if i > 0:
            create_name_list(list_file=data_root + 'line_names.pkl', path=line_path)
            create_name_list(list_file=data_root + 'ann_names.pkl', path=ann_path)
        line_names = pickle.load(open(data_root + 'line_names.pkl', 'rb'))
        ann_names = pickle.load(open(data_root + 'ann_names.pkl', 'rb'))

        crop = transforms.TenCrop(sizes[i])
        # if i < 2:
        #     crop = transforms.TenCrop(sizes[i])
        # else:
        #     crop = transforms.FiveCrop(sizes[i])

        line_crop_path = data_root + 'cropped_lines_{}/'.format(i)
        ann_crop_path = data_root + 'cropped_anns_{}/'.format(i)

        if not os.path.exists(line_crop_path):
            os.makedirs(line_crop_path)
        if not os.path.exists(ann_crop_path):
            os.makedirs(ann_crop_path)

        for file_name in line_names:
            image = Image.open(line_path + file_name + ext)
            crop_images = crop(image)
            for n, crop_image in enumerate(crop_images):
                crop_image.save(line_crop_path + file_name + '-{}.png'.format(n))

        for file_name in ann_names:
            image = Image.open(ann_path + file_name + ext)
            crop_images = crop(image)
            for n, crop_image in enumerate(crop_images):
                crop_image.save(ann_crop_path + file_name + '-{}.png'.format(n))

        line_path = line_crop_path
        ann_path = ann_crop_path

    return ann_path, line_path


if __name__ == '__main__':
    # init
    root = 'tl-data/'
    line_name = ['rcf']
    ann_name = ['ann']
    with open(root + 'line_names.pkl', 'wb') as f:
        pickle.dump(line_name, f)
    with open(root + 'ann_names.pkl', 'wb') as f:
        pickle.dump(ann_name, f)

    # create data set
    fin_ann_path, fin_line_path = image_crop(root)

    # create new name list after add blank images manually (if needed)
    create_name_list(list_file=root + 'line_names.pkl', path=fin_line_path)
    create_name_list(list_file=root + 'ann_names.pkl', path=fin_ann_path)
