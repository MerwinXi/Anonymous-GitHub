import os
import os.path as osp
import numpy as np
import cv2
import json
import random
import logging
from .base_dataset import BaseDataset

SPLIT_FILES = {
    'trainval': [
        'label_data_0313.json', 'label_data_0601.json', 'label_data_0531.json'
    ],
    'train': [
        'label_data_0313.json', 'label_data_0601.json'
    ],
    'val': [
        'label_data_0531.json'
    ],
    'test': [
        'test_label.json'
    ],
}


class TuSimple(BaseDataset):
    def __init__(self, data_root, split, processes=None, cfg=None):
        """
        Initialize the TuSimple dataset.

        Args:
            data_root (str): Root directory for the data.
            split (str): Dataset split, e.g., 'train', 'val', or 'test'.
            processes (list of dicts, optional): List of data processing transformations.
            cfg (object, optional): Configuration object with dataset parameters.
        """
        super().__init__(data_root, split, processes=processes, cfg=cfg)
        self.anno_files = SPLIT_FILES[split]
        self.h_samples = list(range(160, 720, 10))
        self.load_annotations()

    def load_annotations(self):
        """
        Load annotations from JSON files and prepare dataset.
        """
        self.logger.info('Loading TuSimple annotations...')
        self.data_infos = []
        max_lanes = 0

        for anno_file in self.anno_files:
            anno_file_path = osp.join(self.data_root, anno_file)
            with open(anno_file_path, 'r') as file:
                lines = file.readlines()

            for line in lines:
                data = json.loads(line)
                y_samples = data['h_samples']
                gt_lanes = data['lanes']
                mask_path = data['raw_file'].replace('clips', 'seg_label')[:-3] + 'png'
                lanes = [
                    [(x, y) for (x, y) in zip(lane, y_samples) if x >= 0]
                    for lane in gt_lanes
                ]
                lanes = [lane for lane in lanes if len(lane) > 0]
                max_lanes = max(max_lanes, len(lanes))

                self.data_infos.append({
                    'img_path': osp.join(self.data_root, data['raw_file']),
                    'img_name': data['raw_file'],
                    'mask_path': osp.join(self.data_root, mask_path),
                    'lanes': lanes,
                })

        if self.training:
            random.shuffle(self.data_infos)

        self.max_lanes = max_lanes

    def pred2lanes(self, pred):
        """
        Convert model predictions to lane coordinates.

        Args:
            pred (list): List of predicted lane functions.

        Returns:
            list: List of lanes with coordinates in image space.
        """
        ys = np.array(self.h_samples) / self.cfg.ori_img_h
        lanes = []

        for lane in pred:
            xs = lane(ys)
            invalid_mask = xs < 0
            lane = (xs * self.cfg.ori_img_w).astype(int)
            lane[invalid_mask] = -2
            lanes.append(lane.tolist())

        return lanes

    def pred2tusimpleformat(self, idx, pred, runtime):
        """
        Format predictions into TuSimple format.

        Args:
            idx (int): Index of the current image.
            pred (list): List of predicted lane functions.
            runtime (float): Inference time in seconds.

        Returns:
            str: JSON formatted string of the prediction.
        """
        runtime_ms = runtime * 1000  # Convert seconds to milliseconds
        img_name = self.data_infos[idx]['img_name']
        lanes = self.pred2lanes(pred)
        output = {'raw_file': img_name, 'lanes': lanes, 'run_time': runtime_ms}
        return json.dumps(output)

    def save_tusimple_predictions(self, predictions, filename, runtimes=None):
        """
        Save TuSimple formatted predictions to a file.

        Args:
            predictions (list): List of predictions.
            filename (str): Output file path.
            runtimes (list, optional): List of inference times in seconds.
        """
        if runtimes is None:
            runtimes = np.ones(len(predictions)) * 1.e-3

        lines = [
            self.pred2tusimpleformat(idx, prediction, runtime)
            for idx, (prediction, runtime) in enumerate(zip(predictions, runtimes))
        ]

        with open(filename, 'w') as output_file:
            output_file.write('\n'.join(lines))

    def evaluate(self, predictions, output_basedir, runtimes=None):
        """
        Evaluate predictions and save the results.

        Args:
            predictions (list): List of predictions.
            output_basedir (str): Directory to save evaluation results.
            runtimes (list, optional): List of inference times in seconds.

        Returns:
            float: Accuracy score of the evaluation.
        """
        pred_filename = osp.join(output_basedir, 'tusimple_predictions.json')
        self.save_tusimple_predictions(predictions, pred_filename, runtimes)
        result, acc = LaneEval.bench_one_submit(pred_filename, self.cfg.test_json_file)
        self.logger.info(result)
        return acc
