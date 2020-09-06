import os
import pandas as pd
import torch
import numpy as np
from torch.utils.data import Dataset
from skimage import io
from PIL import Image

class CatsAndDogsDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.annotations = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
    
    def __len__(self):
        #return length of the passed dataset
        return len(self.annotations)
    
    def __getitem__(self, index):
        #return an image along with its target
        img_path = os.path.join(self.root_dir,
                                self.annotations.iloc[index,0])
        im = io.imread(img_path)
        #plt.imshow(im)
        #im = im.astype(float)/255.0
        image = Image.fromarray(im)
        y_label = torch.tensor(int(self.annotations.iloc[index,1]))
        
        if self.transform:
            image = self.transform(image)
        
        return (image, y_label)