import os
from PIL import Image
from torch.utils.data import Dataset


class BaseSemanticDataset(Dataset):
    '''
    image: input image
    label: semantic mask with classes
    '''

    def __init__(self):
        pass

    def __getitem__(self, item):
        pass


class VOCSemanticDataset(Dataset):
    def __init__(self, root_dir, domain, transform, with_id=False, with_mask=False):
        super(VOCSemanticDataset, self).__init__()
        self.root_dir = root_dir

        self.image_dir = self.root_dir + 'JPEGImages/'
        self.xml_dir = self.root_dir + 'Annotations/'
        self.mask_dir = self.root_dir + 'SegmentationClass/'

        self.image_id_list = [image_id.strip() for image_id in open('./data/%s.txt' % domain).readlines()]
        self.transform = transform
        self.with_id = with_id
        self.with_mask = with_mask
        self.class_names = ['bicycle', 'bird', 'boat', 'bottle',
                            'bus', 'car', 'cat', 'chair', 'cow',
                            'diningtable', 'dog', 'horse', 'motorbike', 'person',
                            'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']

    def __len__(self):
        return len(self.image_id_list)

    def get_image(self, image_id):
        image = Image.open(self.image_dir + image_id + '.jpg').convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        return image

    def get_mask(self, image_id):
        mask_path = self.mask_dir + image_id + '.png'
        if os.path.isfile(mask_path):
            mask = Image.open(mask_path)
        else:
            mask = None
        return mask

    def __getitem__(self, index):
        image_id = self.image_id_list[index]

        data_list = [self.get_image(image_id)]

        if self.with_id:
            data_list.append(image_id)

        if self.with_mask:
            data_list.append(self.get_mask(image_id))

        return data_list
