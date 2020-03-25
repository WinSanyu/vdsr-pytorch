from os import listdir
from os.path import join

import torch.utils.data as data
from PIL import Image
from torchvision.transforms import ToTensor

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg", ".bmp"])


class TrainDatasetFromFolder(data.Dataset):
    def __init__(self):
        super(TrainDatasetFromFolder, self).__init__()
        self.input_dir = '../data/Train_sub_bic'
        self.label_dir = '../data/Train_sub'
		
        self.image_filenames = listdir(self.input_dir)

    def __getitem__(self, index):
        input_image = Image.open( join(self.input_dir ,self.image_filenames[index]) )
        label_image = Image.open( join(self.label_dir ,self.image_filenames[index]) )

        return ToTensor()(input_image), ToTensor()(label_image)

    def __len__(self):
        return len(self.image_filenames)
        
class TestDatasetFromFolder(data.Dataset):
    def __init__(self):
        super(TestDatasetFromFolder, self).__init__()
        self.input_dir = 'data/Test_sub_bic'
        self.label_dir = 'data/Test_sub'
		
        self.image_filenames = listdir(self.input_dir)

    def __getitem__(self, index):
        input_image = Image.open( join(self.input_dir ,self.image_filenames[index]) )
        label_image = Image.open( join(self.label_dir ,self.image_filenames[index]) )

        return ToTensor()(input_image), ToTensor()(label_image)

    def __len__(self):
        return len(self.image_filenames)