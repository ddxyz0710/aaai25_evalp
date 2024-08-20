
import os
import pickle
import numpy as np
import torch.utils.data
import PIL

from torchvision import datasets

from torch.utils import data

import PIL





class ImageFolderDataset(data.Dataset):
    def __init__(self, root, imageSize, 
                 num_images=None,
                 transforms=None):
        self.root = root
        self.imageSize = imageSize
        self.num_images = num_images #TODO: add subset of dataset with first num_images samples
        self.imgList = os.listdir(root)
        self.transform = transforms
    
    def __len__(self):
        return len(self.imgList)
    
    def __getitem__(self, idx):
        img = PIL.Image.open(os.path.join(self.root, self.imgList[idx]))
        # img.save(f"test{idx}.jpg")
        img = self.transform(img)
        return img 

class ImageMTDataset(data.Dataset):
    def __init__(self, root, cache, num_images=None, transforms=None, 
                 workers=32, protocol=None,
                 normalize_to_0_1=False):
        self.normalize_to_0_1 = normalize_to_0_1
        self.to_range_0_1 = lambda x: (x + 1.) / 2.

        if cache is not None and os.path.exists(cache):
            print(f"Loadidng data from the cache {cache} ... ")
            with open(cache, 'rb') as f:
                self.images = pickle.load(f)
        else:
            self.transform = transforms if not transforms is None else lambda x: x
            self.images = []

            def split_seq(seq, size):
                newseq = []
                splitsize = 1.0 / size * len(seq)
                for i in range(size):
                    newseq.append(seq[int(round(i * splitsize)):int(round((i + 1) * splitsize))])
                return newseq
            
            def map(path_imgs):
                imgs_0 = [self.transform(np.array(PIL.Image.open(os.path.join(root, p_i)))) for p_i in path_imgs]
                imgs_1 = [self.compress(img) for img in imgs_0]

                print('.')
                return imgs_1

            path_imgs = os.listdir(root)
            n_splits = len(path_imgs) // 1000
            path_imgs_splits = split_seq(path_imgs, n_splits)

            from multiprocessing.dummy import Pool as ThreadPool
            pool = ThreadPool(workers)
            results = pool.map(map, path_imgs_splits)
            pool.close()
            pool.join()

            for r in results:
                self.images.extend(r)

            if cache is not None:
                with open(cache, 'wb') as f:
                    pickle.dump(self.images, f, protocol=protocol)

        print('Total number of images {}'.format(len(self.images)))


    def __getitem__(self, item):
        if self.normalize_to_0_1:
            return self.to_range_0_1(self.decompress(self.images[item]))
        else:
            return self.decompress(self.images[item])

    def __len__(self):
        return len(self.images)
    
    @staticmethod
    def compress(img):
        return img

    @staticmethod
    def decompress(output):
        return output
    
class ImageListDataset(data.Dataset):
    '''
    Create a dataset fromlist of image files. 
    Used in FID calculation for pre-generated list of samples. 
    '''
    def __init__(self, imgList):
        self.imgList = imgList
    
    def __len__(self):
        return len(self.imgList)
    def __getitem__(self, idx):
        img = self.imgList[idx]
        return img