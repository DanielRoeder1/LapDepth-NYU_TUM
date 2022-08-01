from torch.utils.data import Dataset, DataLoader
import os
import h5py
import numpy as np
from PIL import Image
from transform_list import RandomCropNumpy,EnhancedCompose,RandomColor,RandomHorizontalFlip,ArrayToTensorNumpy,Normalize
from torchvision import transforms

class Transformer(object):
    def __init__(self, crop_dim):
      self.train_transform = EnhancedCompose([
          RandomCropNumpy(crop_dim),
          RandomHorizontalFlip(),
          [RandomColor(multiplier_range=(0.8, 1.2),brightness_mult_range=(0.75, 1.25)), None],
          ArrayToTensorNumpy(),
          [transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), None]
      ])
      self.test_transform = EnhancedCompose([
          ArrayToTensorNumpy(),
          [transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), None]
      ])
    def __call__(self, images, train=True):
        if train is True:
            return self.train_transform(images)
        else:
            return self.test_transform(images)


class LapDepth_Dataset(Dataset):
  def __init__(self, nyu_path, train = True, depth_scale = 5000):
    self.nyu_file_paths = [os.path.join(root, file_name) for root, _, filenames in os.walk(nyu_path) for file_name in filenames]
    #self.tum_file_paths = [os.path.join(root, file_name) for root, _, filenames in os.walk(tum_path) for file_name in filenames]
    self.depth_scale = depth_scale
    self.train = train
    self.angle_range = (-2.5, 2.5)
    self.crop_params = (20,20,620,460)
    self.max_depth = 10
    self.transform = Transformer((416,544))
    
  def __getraw__(self, idx):
    """
    Returns rgb and depth in PIL.Image format
    NYU depths are scaled to allow transforming to PIL.Image
    """
    h5f = h5py.File(self.nyu_file_paths[idx], "r")
    rgb = np.array(h5f['rgb'])
    depth = np.array(h5f['depth'])

    if depth.dtype != 'int32':
      rgb = np.transpose(rgb, (1, 2, 0))
      depth = (np.around(depth, 3) * self.depth_scale).astype(np.int32)
  
    rgb = Image.fromarray(rgb)
    depth = Image.fromarray(depth)

    return rgb, depth
    
  
  def __getitem__(self,idx):
    """
    Apply transformation and return rgbs, depth pairs
    Crop only 20 pixel from each side to remove white barrier
    """
    rgb, depth = self.__getraw__(idx)

    # Rotate
    angle = np.random.uniform(self.angle_range[0], self.angle_range[1])
    rgb = rgb.rotate(angle, resample=Image.BILINEAR)
    depth = depth.rotate(angle, resample=Image.NEAREST)

    # Crop
    rgb = rgb.crop(self.crop_params)
    depth = depth.crop(self.crop_params)

    # Standardize / Scale
    rgb = np.asarray(rgb, dtype=np.float32)/255.0
    depth = (np.asarray(depth, dtype=np.float32))/self.depth_scale
    depth = np.expand_dims(depth, axis=2)
    depth = np.clip(depth, 0, self.max_depth)

    # Transform
    rgb, depth = self.transform([rgb] + [depth], self.train)

    # Return depth twice to use it as dense depth
    return rgb, depth, depth
  
  def __len__(self):
    return len(self.nyu_file_paths)