import math
import numpy as np
import cv2
import imgaug.augmenters as iaa
from imgaug.augmentables.lines import LineString, LineStringsOnImage
from imgaug.augmentables.segmaps import SegmentationMapsOnImage
from scipy.interpolate import InterpolatedUnivariateSpline


class GenerateLaneLine:
    def __init__(self, transforms=None, cfg=None, training=True):
        self.cfg = cfg
        self.img_w, self.img_h = cfg.img_w, cfg.img_h
        self.num_points = cfg.num_points
        self.n_strips = self.num_points - 1
        self.strip_size = self.img_h / self.n_strips
        self.max_lanes = cfg.max_lanes
        self.offsets_ys = np.arange(self.img_h, -1, -self.strip_size)
        self.training = training

        self.transforms = transforms or CLRTransforms(self.img_h, self.img_w)
        self.transform = self._build_transform_pipeline(transforms)

    def _build_transform_pipeline(self, transforms):
        img_transforms = []
        if transforms:
            for aug in transforms:
                p = aug['p']
                if aug['name'] != 'OneOf':
                    img_transforms.append(iaa.Sometimes(p=p, then_list=getattr(iaa, aug['name'])(**aug['parameters'])))
                else:
                    img_transforms.append(
                        iaa.Sometimes(
                            p=p,
                            then_list=iaa.OneOf([
                                getattr(iaa, aug_['name'])(**aug_['parameters'])
                                for aug_ in aug['transforms']
                            ])
                        )
                    )
        return iaa.Sequential(img_transforms)

    def lane_to_linestrings(self, lanes):
        return [LineString(lane) for lane in lanes]

    def sample_lane(self, points, sample_ys):
        points = np.array(points)
        if not np.all(points[1:, 1] < points[:-1, 1]):
            raise ValueError('Annotation points must be sorted in descending Y order')

        x, y = points[:, 0], points[:, 1]
        interp = InterpolatedUnivariateSpline(y[::-1], x[::-1], k=min(3, len(points) - 1))
        sample_ys_inside_domain = sample_ys[(sample_ys >= y.min()) & (sample_ys <= y.max())]

        if len(sample_ys_inside_domain) == 0:
            return np.array([]), np.array([])

        interp_xs = interp(sample_ys_inside_domain)
        extrap_ys = sample_ys[sample_ys > y.max()]
        extrap_xs = np.polyval(np.polyfit(points[:2, 1], points[:2, 0], deg=1), extrap_ys)

        all_xs = np.hstack((extrap_xs, interp_xs))
        inside_mask = (all_xs >= 0) & (all_xs < self.img_w)
        return all_xs[~inside_mask], all_xs[inside_mask]

    def filter_lane(self, lane):
        unique_y = set()
        filtered_lane = [p for p in lane if p[1] not in unique_y and not unique_y.add(p[1])]
        return filtered_lane

    def transform_annotation(self, anno, img_wh=None):
        old_lanes = filter(lambda x: len(x) > 1, anno['lanes'])
        old_lanes = [self.filter_lane(sorted(lane, key=lambda x: -x[1])) for lane in old_lanes]
        old_lanes = [[[x * self.img_w / float(img_wh[0]), y * self.img_h / float(img_wh[1])] for x, y in lane] for lane
                     in old_lanes]

        lanes = np.ones((self.max_lanes, 6 + self.num_points), dtype=np.float32) * -1e5
        lanes[:, :2] = [1, 0]  # Mark lanes as invalid by default
        lanes_endpoints = np.ones((self.max_lanes, 2))

        for lane_idx, lane in enumerate(old_lanes):
            if lane_idx >= self.max_lanes:
                break

            try:
                xs_outside_image, xs_inside_image = self.sample_lane(lane, self.offsets_ys)
            except AssertionError:
                continue

            if len(xs_inside_image) <= 1:
                continue

            all_xs = np.hstack((xs_outside_image, xs_inside_image))
            lanes[lane_idx, 0] = 0
            lanes[lane_idx, 1] = 1
            lanes[lane_idx, 2] = len(xs_outside_image) / self.n_strips
            lanes[lane_idx, 3] = xs_inside_image[0]

            thetas = [math.atan(i * self.strip_size / (xs_inside_image[i] - xs_inside_image[0] + 1e-5)) / math.pi for i
                      in range(1, len(xs_inside_image))]
            theta_far = sum(thetas) / len(thetas)
            lanes[lane_idx, 4] = theta_far
            lanes[lane_idx, 5] = len(xs_inside_image)
            lanes[lane_idx, 6:6 + len(all_xs)] = all_xs
            lanes_endpoints[lane_idx, 0] = (len(all_xs) - 1) / self.n_strips
            lanes_endpoints[lane_idx, 1] = xs_inside_image[-1]

        return {'label': lanes, 'old_anno': anno, 'lane_endpoints': lanes_endpoints}

    def linestrings_to_lanes(self, lines):
        return [line.coords for line in lines]

    def __call__(self, sample):
        img_org = sample['img']
        line_strings_org = LineStringsOnImage(self.lane_to_linestrings(sample['lanes']), shape=img_org.shape)

        for _ in range(30):
            if self.training:
                mask_org = SegmentationMapsOnImage(sample['mask'], shape=img_org.shape)
                img, line_strings, seg = self.transform(image=img_org.copy().astype(np.uint8),
                                                        line_strings=line_strings_org, segmentation_maps=mask_org)
            else:
                img, line_strings = self.transform(image=img_org.copy().astype(np.uint8), line_strings=line_strings_org)

            line_strings.clip_out_of_image_()
            new_anno = {'lanes': self.linestrings_to_lanes(line_strings)}

            try:
                annos = self.transform_annotation(new_anno, img_wh=(self.img_w, self.img_h))
                sample['img'] = img.astype(np.float32) / 255.
                sample['lane_line'] = annos['label']
                sample['lanes_endpoints'] = annos['lane_endpoints']
                sample['gt_points'] = new_anno['lanes']
                sample['seg'] = seg.get_arr() if self.training else np.zeros(img_org.shape)
                return sample
            except Exception as e:
                if _ == 29:
                    raise RuntimeError(f'Transform annotation failed after 30 attempts: {e}')
