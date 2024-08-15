import os
import os.path as osp
import numpy as np
import cv2
import pickle as pkl
from tqdm import tqdm
import logging
from .base_dataset import BaseDataset

LIST_FILE = {
    'train': 'list/train_gt.txt',
    'val': 'list/val.txt',
    'test': 'list/test.txt',
}

CATEGORYS = {
    'normal': 'list/test_split/test0_normal.txt',
    'crowd': 'list/test_split/test1_crowd.txt',
    'hlight': 'list/test_split/test2_hlight.txt',
    'shadow': 'list/test_split/test3_shadow.txt',
    'noline': 'list/test_split/test4_noline.txt',
    'arrow': 'list/test_split/test5_arrow.txt',
    'curve': 'list/test_split/test6_curve.txt',
    'cross': 'list/test_split/test7_cross.txt',
    'night': 'list/test_split/test8_night.txt',
}


class CULane(BaseDataset):
    def __init__(self, data_root, split, processes=None, cfg=None):
        """
        Initialize the CULane dataset.

        Args:
            data_root (str): Root directory for the data.
            split (str): Dataset split, e.g., 'train', 'val', or 'test'.
            processes (list of dicts, optional): List of data processing transformations.
            cfg (object, optional): Configuration object with dataset parameters.
        """
        super().__init__(data_root, split, processes=processes, cfg=cfg)
        self.list_path = osp.join(data_root, LIST_FILE[split])
        self.split = split
        self.load_annotations()

    def load_annotations(self):
        """
        Load and cache CULane annotations.
        """
        self.logger.info('Loading CULane annotations...')
        cache_path = 'cache/culane_{}.pkl'.format(self.split)
        os.makedirs('cache', exist_ok=True)

        if os.path.exists(cache_path):
            with open(cache_path, 'rb') as cache_file:
                self.data_infos = pkl.load(cache_file)
                self.max_lanes = max(len(anno['lanes']) for anno in self.data_infos)
                return

        self.data_infos = []
        with open(self.list_path) as list_file:
            for line in list_file:
                infos = self.load_annotation(line.strip().split())
                self.data_infos.append(infos)

        with open(cache_path, 'wb') as cache_file:
            pkl.dump(self.data_infos, cache_file)

    def load_annotation(self, line):
        """
        Parse an annotation line.

        Args:
            line (list): A list of annotation data.

        Returns:
            dict: Parsed annotation information.
        """
        infos = {}
        img_line = line[0].lstrip('/')
        img_path = osp.join(self.data_root, img_line)
        infos['img_name'] = img_line
        infos['img_path'] = img_path

        if len(line) > 1:
            mask_line = line[1].lstrip('/')
            mask_path = osp.join(self.data_root, mask_line)
            infos['mask_path'] = mask_path

        if len(line) > 2:
            exist_list = [int(l) for l in line[2:]]
            infos['lane_exist'] = np.array(exist_list)

        anno_path = img_path.replace('.jpg', '.lines.txt')
        with open(anno_path, 'r') as anno_file:
            data = [list(map(float, line.split())) for line in anno_file]

        lanes = [
            [(lane[i], lane[i + 1]) for i in range(0, len(lane), 2)
             if lane[i] >= 0 and lane[i + 1] >= 0]
            for lane in data
        ]
        lanes = [list(set(lane)) for lane in lanes]
        lanes = [lane for lane in lanes if len(lane) > 2]
        lanes = [sorted(lane, key=lambda x: x[1]) for lane in lanes]
        infos['lanes'] = lanes

        return infos

    def get_prediction_string(self, pred):
        """
        Convert predictions to the required format.

        Args:
            pred (list): List of lane predictions.

        Returns:
            str: Formatted prediction string.
        """
        ys = np.arange(270, 590, 8) / self.cfg.ori_img_h
        out = []
        for lane in pred:
            xs = lane(ys)
            valid_mask = (xs >= 0) & (xs < 1)
            xs = xs * self.cfg.ori_img_w
            lane_xs = xs[valid_mask]
            lane_ys = ys[valid_mask] * self.cfg.ori_img_h
            lane_xs, lane_ys = lane_xs[::-1], lane_ys[::-1]
            lane_str = ' '.join(f'{x:.5f} {y:.5f}' for x, y in zip(lane_xs, lane_ys))
            if lane_str:
                out.append(lane_str)

        return '\n'.join(out)

    def evaluate(self, predictions, output_basedir):
        """
        Evaluate predictions and generate output files.

        Args:
            predictions (list): List of lane predictions.
            output_basedir (str): Directory to save evaluation results.

        Returns:
            float: F1 score of the evaluation.
        """
        os.makedirs(output_basedir, exist_ok=True)
        self.logger.info('Generating prediction output...')
        for idx, pred in enumerate(predictions):
            output_dir = osp.join(output_basedir, osp.dirname(self.data_infos[idx]['img_name']))
            output_filename = osp.basename(self.data_infos[idx]['img_name']).replace('.jpg', '.lines.txt')
            os.makedirs(output_dir, exist_ok=True)
            output = self.get_prediction_string(pred)
            with open(osp.join(output_dir, output_filename), 'w') as out_file:
                out_file.write(output)

        results = {}
        for cate, cate_file in CATEGORYS.items():
            results[cate] = culane_metric.eval_predictions(
                output_basedir,
                self.data_root,
                osp.join(self.data_root, cate_file),
                iou_thresholds=[0.5],
                official=True
            )

        result = culane_metric.eval_predictions(
            output_basedir,
            self.data_root,
            self.list_path,
            iou_thresholds=np.linspace(0.5, 0.95, 10),
            official=True
        )

        return result[0.5]['F1']
