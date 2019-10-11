import os
import cv2
import skimage.io as io
from PIL import Image

import torch
from torch.utils.data.dataset import Dataset
from torchvision import transforms


class Classifier(Dataset):
    def __init__(self, root_dir, height, width, grayscale):
        if grayscale:
            self.grayscale = True
        self.root_dir = root_dir
        self.height = height
        self.width = width
        self.image_paths, self.labels = self.readDataset()
        # print(self.image_paths)
        self.transforms = transforms.Compose([transforms.ToPILImage(),
                                            transforms.Resize((self.width, self.height)),
                                              transforms.ToTensor()])

    def __getitem__(self, index):
        # image = io.imread(self.image_paths[index], as_gray=self.grayscale)
        # image = Image.fromarray(image)
        try:
            image=cv2.imread(self.image_paths[index],cv2.IMREAD_GRAYSCALE)
            label = self.labels[index]
            image = self.transforms(image)
        except:
            print('Error in file ',self.image_paths[index])
            if index==0:
                index=index+1
            else:
                index=index-1
            image=cv2.imread(self.image_paths[index],cv2.IMREAD_GRAYSCALE)
            label = self.labels[index]
            image = self.transforms(image)
        # print(image.shape)
        # print(type(image))
        return (image, label)

    def __len__(self):
        return len(self.labels)

    def readDataset(self):
        image_paths = []
        labels = []
        print(self.root_dir)
        # print(os.walk(self.root_dir))
        for (dirpath, dirnames, filenames) in os.walk(self.root_dir):
            # print(dirpath,dirnames,filenames)
            filenames = [f for f in filenames if not f[0] == '.']
            dirnames[:] = [d for d in dirnames if not d[0] == '.']

            if not dirnames:

                for file in filenames:
                    if file.lower().endswith('.jpg') or file.lower().endswith('.png'):
                        image_paths.append(os.path.join(dirpath, file))
                        labels.append(int(dirpath.split("/")[-1]))

        return image_paths, labels


# transformations = transforms.Compose([transforms.ToTensor()])

if __name__ == "__main__":
    root_dir = './data/'
    width = 30
    height = 40
    grayscale = True
    digit_dataset = Classifier(root_dir=root_dir, width=width, height=height, grayscale=grayscale)
    data_loader = torch.utils.data.DataLoader(dataset=digit_dataset,
                                              batch_size=10,
                                              shuffle=False)
    for images, labels in data_loader:
        print(images[0])
        print(images.shape)
        break

# def load_dataset():
#     data_path = 'data/train/'
#     train_dataset = torchvision.datasets.ImageFolder(
#         root=data_path,
#         transform=torchvision.transforms.ToTensor()
#     )
#     train_loader = torch.utils.data.DataLoader(
#         train_dataset,
#         batch_size=64,
#         num_workers=0,
#         shuffle=True
#     )
#     return train_loader
#
# for batch_idx, (data, target) in enumerate(load_dataset()):
