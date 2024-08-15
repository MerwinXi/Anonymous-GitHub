import os.path as osp
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import logging
from .process import Process
from mmcv.parallel import DataContainer as DC


class BaseDataset(Dataset):
    """Base dataset class for handling image and mask data, with support for transformations and visualization."""

    def __init__(self, data_root, split, processes=None, cfg=None):
        """
        Initialize the dataset.

        Args:
            data_root (str): Root directory for the data.
            split (str): Dataset split, e.g., 'train' or 'val'.
            processes (list of dicts, optional): List of data processing transformations.
            cfg (object, optional): Configuration object with dataset parameters.
        """
        self.cfg = cfg
        self.logger = logging.getLogger(__name__)
        self.data_root = data_root
        self.training = 'train' in split
        self.processes = Process(processes, cfg)
        self.data_infos = self.load_data_infos()

    def load_data_infos(self):
        """Load dataset information from the root directory."""
        # This method should be implemented to populate self.data_infos with dataset info.
        raise NotImplementedError("Subclasses should implement this method.")

    def view(self, predictions, img_metas):
        """
        Visualize the predictions.

        Args:
            predictions (list): List of lane predictions.
            img_metas (list): Metadata of images.
        """
        img_metas = [item for img_meta in img_metas.data for item in img_meta]
        for lanes, img_meta in zip(predictions, img_metas):
            img_name = img_meta['img_name']
            img_path = osp.join(self.data_root, img_name)
            img = cv2.imread(img_path)
            out_file = osp.join(self.cfg.work_dir, 'visualization', img_name.replace('/', '_'))
            lanes = [lane.to_array(self.cfg) for lane in lanes]
            self.imshow_lanes(img, lanes, out_file=out_file)

    def imshow_lanes(self, img, lanes, out_file):
        """Helper function to display lanes on the image and save the result."""
        # This function should be implemented to visualize lanes on the image.
        raise NotImplementedError("Subclasses should implement this method.")

    def __len__(self):
        return len(self.data_infos)

    def __getitem__(self, idx):
        """
        Get an item from the dataset.

        Args:
            idx (int): Index of the item.

        Returns:
            dict: A dictionary containing the image, mask, and metadata.
        """
        data_info = self.data_infos[idx]
        img = cv2.imread(data_info['img_path'])
        img = img[self.cfg.cut_height:, :, :]
        sample = {'img': img}

        if self.training:
            label = cv2.imread(data_info['mask_path'], cv2.IMREAD_UNCHANGED)
            if len(label.shape) > 2:
                label = label[:, :, 0]
            label = label.squeeze()
            label = label[self.cfg.cut_height:, :]
            sample.update({'mask': label})

            if self.cfg.cut_height != 0:
                sample['lanes'] = [
                    [(p[0], p[1] - self.cfg.cut_height) for p in lane]
                    for lane in data_info.get('lanes', [])
                ]

        sample = self.processes(sample)
        meta = {'full_img_path': data_info['img_path'], 'img_name': data_info['img_name']}
        meta = DC(meta, cpu_only=True)
        sample['meta'] = meta

        return sample
