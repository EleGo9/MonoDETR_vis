import os
import numpy as np
import torch.utils.data as data
from PIL import Image, ImageFile, ImageDraw
import random
ImageFile.LOAD_TRUNCATED_IMAGES = True

from lib.datasets.utils import angle2class
from lib.datasets.utils import gaussian_radius
from lib.datasets.utils import draw_umich_gaussian, draw_projected_box3d
from lib.datasets.kitti.kitti_utils import get_objects_from_label
from lib.datasets.kitti.kitti_utils import Calibration
from lib.datasets.kitti.kitti_utils import get_affine_transform
from lib.datasets.kitti.kitti_utils import affine_transform
from lib.datasets.kitti.indy_eval_python.eval import get_official_eval_result
from lib.datasets.kitti.indy_eval_python.eval import get_distance_eval_result
import lib.datasets.kitti.indy_eval_python.indy_common as indy
# from lib.datasets.utils import box3d_to_corners, draw_projected_box3d
import copy
from .pd import PhotometricDistort

import sys
import os

# Get the absolute path of the directory containing this script
# current_dir = '/home/elenagovi/repos/MonoDETR/lib/datasets/kitti/'

# # Add the parent directory to sys.path if 'lib' is in the parent directory
# project_root = os.path.abspath(os.path.join(current_dir, "../../../"))  # Adjust if needed
# sys.path.append(parent_dir)


class INDY_Dataset(data.Dataset):
    def __init__(self, split, cfg, root_dir=None):

        # basic configuration
        if root_dir is None:
            self.root_dir = cfg.get('root_dir')
        else:
            self.root_dir = root_dir
        # print(self.root_dir)
        self.split = split
        self.num_classes = 1
        self.max_objs = 10
        self.class_name = [ 'Car']
        self.cls2id = {'Car': 0}
        self.resolution = np.array([1280, 384])  # W * H
        self.use_3d_center = cfg.get('use_3d_center', True)
        self.writelist = cfg.get('writelist', ['Car'])
        # anno: use src annotations as GT, proj: use projected 2d bboxes as GT
        self.bbox2d_type = cfg.get('bbox2d_type', 'anno')
        assert self.bbox2d_type in ['anno', 'proj']
        self.meanshape = cfg.get('meanshape', False)
        self.class_merging = cfg.get('class_merging', False)
        self.use_dontcare = cfg.get('use_dontcare', False)

        if self.class_merging:
            self.writelist.extend(['Van', 'Truck'])
        if self.use_dontcare:
            self.writelist.extend(['DontCare'])

        # data split loading
        assert self.split in ['train', 'val', 'trainval', 'test', 'all']
        self.split_file = os.path.join(self.root_dir, 'ImageSets', self.split + '.txt')
        self.idx_list = [x.strip() for x in open(self.split_file).readlines()]

        # path configuration
        self.data_dir = os.path.join(self.root_dir) #, 'testing' if split == 'test' else 'training')
        self.image_dir = os.path.join(self.data_dir, 'image_2')
        self.calib_dir = os.path.join(self.data_dir, 'calib')
        self.label_dir = os.path.join(self.data_dir, 'label_2')

        # data augmentation configuration
        self.data_augmentation = True if split in ['train', 'trainval'] else False

        self.aug_pd = cfg.get('aug_pd', False)
        self.aug_crop = cfg.get('aug_crop', False)
        self.aug_calib = cfg.get('aug_calib', False)

        self.random_flip = cfg.get('random_flip', 0.5)
        self.random_crop = cfg.get('random_crop', 0.5)
        self.scale = cfg.get('scale', 0.4)
        self.shift = cfg.get('shift', 0.1)

        self.depth_scale = cfg.get('depth_scale', 'normal')

        # statistics
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        
        # self.mean = np.array([1.2, 1.9, 5.0], dtype=np.float32)
        # self.std = np.array([0.1, 0.1, 0.1], dtype=np.float32)
        self.cls_mean_size = np.array([[1.52563191462,1.62856739989, 3.88311640418]])
        if not self.meanshape:
            self.cls_mean_size = np.zeros_like(self.cls_mean_size, dtype=np.float32)

        # others
        self.downsample = 32
        self.pd = PhotometricDistort()
        self.clip_2d = cfg.get('clip_2d', False)

    def get_image(self, idx):
        img_file = os.path.join(self.image_dir, '%06d.png' % idx)
        assert os.path.exists(img_file)
        return Image.open(img_file)    # (H, W, 3) RGB mode

    def get_label(self, idx):
        label_file = os.path.join(self.label_dir, '%06d.txt' % idx)
        assert os.path.exists(label_file)
        return get_objects_from_label(label_file)

    def get_calib(self, idx):
        calib_file = os.path.join(self.calib_dir, '%06d.txt' % idx)
        assert os.path.exists(calib_file)
        return Calibration(calib_file)

    def eval(self, results_dir, logger):
        logger.info("==> Loading detections and GTs...")
        img_ids = [int(id) for id in self.idx_list]
        dt_annos = indy.get_label_annos(results_dir)
        gt_annos = indy.get_label_annos(self.label_dir, img_ids)

        test_id = {'Car': 0}

        logger.info('==> Evaluating (official) ...')
        car_moderate = 0
        for category in self.writelist:
            results_str, results_dict, mAP3d_R40 = get_official_eval_result(gt_annos, dt_annos, test_id[category])
            if category == 'Car':
                car_moderate = mAP3d_R40
            logger.info(results_str)
        return car_moderate

    def __len__(self):
        return self.idx_list.__len__()

    def __getitem__(self, item):
        #  ============================   get inputs   ===========================
        index = int(self.idx_list[item])  # index mapping, get real data id
        # print(self.idx_list)
        # image loading
        # print(index)
        img = self.get_image(index)
        img_size = np.array(img.size)
        features_size = self.resolution // self.downsample    # W * H

        # data augmentation for image
        center = np.array(img_size) / 2
        crop_size, crop_scale = img_size, 1
        random_flip_flag, random_crop_flag = False, False

        if self.data_augmentation:

            if self.aug_pd:
                img = np.array(img).astype(np.float32)
                img = self.pd(img).astype(np.uint8)
                img = Image.fromarray(img)

            if np.random.random() < self.random_flip:
                random_flip_flag = True
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
            
            if self.aug_crop:
                if np.random.random() < self.random_crop:
                    random_crop_flag = True
                    crop_scale = np.clip(np.random.randn() * self.scale + 1, 1 - self.scale, 1 + self.scale)
                    crop_size = img_size * crop_scale
                    center[0] += img_size[0] * np.clip(np.random.randn() * self.shift, -2 * self.shift, 2 * self.shift)
                    center[1] += img_size[1] * np.clip(np.random.randn() * self.shift, -2 * self.shift, 2 * self.shift)

        # add affine transformation for 2d images.
        trans, trans_inv = get_affine_transform(center, crop_size, 0, self.resolution, inv=1)
        img = img.transform(tuple(self.resolution.tolist()),
                            method=Image.AFFINE,
                            data=tuple(trans_inv.reshape(-1).tolist()),
                            resample=Image.BILINEAR)

        # image encoding
        img = np.array(img).astype(np.float32) / 255.0
        img = (img - self.mean) / self.std

        img = img.transpose(2, 0, 1)  # C * H * W

        info = {'img_id': index,
                'img_size': img_size,
                'bbox_downsample_ratio': img_size / features_size}

        if self.split == 'test':
            calib = self.get_calib(index)
            return img, calib.P2, img, info

        #  ============================   get labels   ==============================
        objects = self.get_label(index)
        calib = self.get_calib(index)

        # data augmentation for labels
        if random_flip_flag:
            if self.aug_calib:
                calib.flip(img_size)
            for object in objects:
                [x1, _, x2, _] = object.box2d
                object.box2d[0],  object.box2d[2] = img_size[0] - x2, img_size[0] - x1
                object.alpha = np.pi - object.alpha
                object.ry = np.pi - object.ry
                if self.aug_calib:
                    object.pos[0] *= -1
                if object.alpha > np.pi:  object.alpha -= 2 * np.pi  # check range
                if object.alpha < -np.pi: object.alpha += 2 * np.pi
                if object.ry > np.pi:  object.ry -= 2 * np.pi
                if object.ry < -np.pi: object.ry += 2 * np.pi

        # labels encoding
        calibs = np.zeros((self.max_objs, 3, 4), dtype=np.float32)
        indices = np.zeros((self.max_objs), dtype=np.int64)
        mask_2d = np.zeros((self.max_objs), dtype=bool)
        labels = np.zeros((self.max_objs), dtype=np.int8)
        depth = np.zeros((self.max_objs, 1), dtype=np.float32)
        heading_bin = np.zeros((self.max_objs, 1), dtype=np.int64)
        heading_res = np.zeros((self.max_objs, 1), dtype=np.float32)
        size_2d = np.zeros((self.max_objs, 2), dtype=np.float32) 
        size_3d = np.zeros((self.max_objs, 3), dtype=np.float32)
        src_size_3d = np.zeros((self.max_objs, 3), dtype=np.float32)
        boxes = np.zeros((self.max_objs, 4), dtype=np.float32)
        boxes_3d = np.zeros((self.max_objs, 6), dtype=np.float32)

        object_num = len(objects) if len(objects) < self.max_objs else self.max_objs
        # print('object num', object_num)
        for i in range(object_num):
            # filter objects by writelist
            if objects[i].cls_type not in self.writelist:
                print(f'{objects[i].cls_type} not in write list')
                continue

            # print('Pos[-1]', objects[i].pos[-1])
            # filter inappropriate samples
            # this is only for kitti categorization in easy, moderate or hard
            # if objects[i].level_str == 'UnKnown' or objects[i].pos[-1] < 2:
            #     print(f'{objects[i].level_str}== UnKnown or {objects[i].pos[-1]}  <2 ')
            #     continue

            # ignore the samples beyond the threshold [hard encoding]
            threshold = 120
            
            if objects[i].pos[-1] > threshold:
                print(f'Too far {objects[i].pos[-1]} > {threshold}')
                continue

            # process 2d bbox & get 2d center
            bbox_2d = objects[i].box2d.copy()
            
            # add affine transformation for 2d boxes.
            bbox_2d[:2] = affine_transform(bbox_2d[:2], trans)
            bbox_2d[2:] = affine_transform(bbox_2d[2:], trans)

            # process 3d center
            center_2d = np.array([(bbox_2d[0] + bbox_2d[2]) / 2, (bbox_2d[1] + bbox_2d[3]) / 2], dtype=np.float32)  # W * H
            corner_2d = bbox_2d.copy()

            center_3d = objects[i].pos + [0, -objects[i].h / 2, 0]  # real 3D center in 3D space
            center_3d = center_3d.reshape(-1, 3)  # shape adjustment (N, 3)
            center_3d, _ = calib.rect_to_img(center_3d)  # project 3D center to image plane
            center_3d = center_3d[0]  # shape adjustment
            if random_flip_flag and not self.aug_calib:  # random flip for center3d
                center_3d[0] = img_size[0] - center_3d[0]
            center_3d = affine_transform(center_3d.reshape(-1), trans)

            # filter 3d center out of img
            proj_inside_img = True

            if center_3d[0] < 0 or center_3d[0] >= self.resolution[0]: 
                proj_inside_img = False
            if center_3d[1] < 0 or center_3d[1] >= self.resolution[1]: 
                proj_inside_img = False

            # if proj_inside_img == False:
                # print('3d box outside img', index)
                # continue #TODO: fix this!!

            # class
            cls_id = self.cls2id[objects[i].cls_type]
            labels[i] = cls_id

            # encoding 2d/3d boxes
            w, h = bbox_2d[2] - bbox_2d[0], bbox_2d[3] - bbox_2d[1]
            size_2d[i] = 1. * w, 1. * h

            center_2d_norm = center_2d / self.resolution
            size_2d_norm = size_2d[i] / self.resolution

            corner_2d_norm = corner_2d
            corner_2d_norm[0: 2] = corner_2d[0: 2] / self.resolution
            corner_2d_norm[2: 4] = corner_2d[2: 4] / self.resolution
            center_3d_norm = center_3d / self.resolution

            l, r = center_3d_norm[0] - corner_2d_norm[0], corner_2d_norm[2] - center_3d_norm[0]
            t, b = center_3d_norm[1] - corner_2d_norm[1], corner_2d_norm[3] - center_3d_norm[1]

            if l < 0 or r < 0 or t < 0 or b < 0:
                if self.clip_2d:
                    l = np.clip(l, 0, 1)
                    r = np.clip(r, 0, 1)
                    t = np.clip(t, 0, 1)
                    b = np.clip(b, 0, 1)
                else:
                    print(f' l < 0 or r < 0 or t < 0 or b < 0  {index}')
                    continue		

            boxes[i] = center_2d_norm[0], center_2d_norm[1], size_2d_norm[0], size_2d_norm[1]
            boxes_3d[i] = center_3d_norm[0], center_3d_norm[1], l, r, t, b

            # encoding depth
            if self.depth_scale == 'normal':
                depth[i] = objects[i].pos[-1] * crop_scale
            
            elif self.depth_scale == 'inverse':
                depth[i] = objects[i].pos[-1] / crop_scale
            
            elif self.depth_scale == 'none':
                depth[i] = objects[i].pos[-1]

            # encoding heading angle
            heading_angle = calib.ry2alpha(objects[i].ry, (objects[i].box2d[0] + objects[i].box2d[2]) / 2)
            if heading_angle > np.pi:  heading_angle -= 2 * np.pi  # check range
            if heading_angle < -np.pi: heading_angle += 2 * np.pi
            heading_bin[i], heading_res[i] = angle2class(heading_angle)

            # encoding size_3d
            src_size_3d[i] = np.array([objects[i].h, objects[i].w, objects[i].l], dtype=np.float32)
            mean_size = self.cls_mean_size[self.cls2id[objects[i].cls_type]]
            size_3d[i] = src_size_3d[i] - mean_size

            if objects[i].trucation <= 0.5 and objects[i].occlusion <= 2:
                mask_2d[i] = 1

            calibs[i] = calib.P2

        # collect return data
        inputs = img
        targets = {
                   'calibs': calibs,
                   'indices': indices,
                   'img_size': img_size,
                   'labels': labels,
                   'boxes': boxes,
                   'boxes_3d': boxes_3d,
                   'depth': depth,
                   'size_2d': size_2d,
                   'size_3d': size_3d,
                   'src_size_3d': src_size_3d,
                   'heading_bin': heading_bin,
                   'heading_res': heading_res,
                   'mask_2d': mask_2d}

        info = {'img_id': index,
                'img_size': img_size,
                'bbox_downsample_ratio': img_size / features_size}
        return inputs, calib.P2, targets, info





if __name__ == '__main__':
    from torch.utils.data import DataLoader
    import matplotlib.pyplot as plt

    cfg = {'root_dir': '/media/franco/hdd/dataset/dataset_3d/20250104_lvms_run02_virginia/camera_fc_cropped',
           'random_flip': 0.0, 'random_crop': 1.0, 'scale': 0.8, 'shift': 0.1, 'use_dontcare': False,
           'class_merging': False, 'writelist':['Car'], 'use_3d_center':False}
    dataset = INDY_Dataset('train', cfg)
    dataloader = DataLoader(dataset=dataset, batch_size=1)
    print(dataset.writelist)

    for batch_idx, (inputs, P2, targets, info) in enumerate(dataloader):
        # test image
        img = inputs[0].numpy().transpose(1, 2, 0)
        img = (img * dataset.std + dataset.mean) * 255
        img = Image.fromarray(img.astype(np.uint8))
        img.show()
        # print('3D labels', targets['boxes_3d']) #cx, cy, l, r, t, b
        # print(P2.shape)
        P2 = P2[0]

        # # print(targets['size_3d'][0][0])
        # print('3D labels', targets['boxes_3d'].shape)
        # image = draw_projected_box3d(img, np.array(targets['boxes_3d'][0]))
        # test heatmap
        # heatmap = targets['heatmap'][0]  # image id
        # heatmap = Image.fromarray(heatmap[0].numpy() * 255)  # cats id
        # heatmap.show()
            
        # For each object in the image
        for i in range(targets['labels'].shape[1]):
            cx, cy, l, r, t, b = targets['boxes_3d'][0, i].numpy()
            # h, w, l = targets['src_size_3d'][0, i].numpy()
            depth_val = float(targets['depth'][0, i])
            bin_id, res = int(targets['heading_bin'][0, i]), float(targets['heading_res'][0, i])
            heading_angle = class2angle(bin_id, res)
            resolution = np.array(img.size)
            
            # # Reconstruct corner points
            cx, cy = cx * resolution[0], cy * resolution[1]
            l, r = l * resolution[0], r * resolution[0]
            t, b = t * resolution[1], b * resolution[1]

            # fx = P2[0, 0]
            # fy = P2[1, 1]
            # cx_cam = P2[0, 2]
            # cy_cam = P2[1, 2]

            # # Back-project from image to 3D
            # x3d = depth_val * (cx - cx_cam) / fx
            # y3d = depth_val * (cy - cy_cam) / fy
            # z3d = depth_val

            # center_3d = [x3d, y3d, z3d]
            # dimensions = [h, w, l]
            # corners_3d = box3d_to_corners(center_3d, dimensions, heading_angle)
            # # Project corners to image
            # print(corners_3d.shape)
            # corners_2d, depth = project_to_image(corners_3d, P2)
            # # Draw the box
            # img = draw_projected_box3d(img, corners_2d, color=(255, 0, 0))
        

                
            
            # Draw a simplified 3D box (this is approximate)
            # For accurate 3D box, you'd need to properly project using calibration
            corners = np.array([
                [cx - l, cy - t],
                [cx + r, cy - t],
                [cx + r, cy + b],
                [cx - l, cy + b]
            ])
            
            draw = ImageDraw.Draw(img)
            for j in range(4):
                draw.line([corners[j, 0], corners[j, 1], corners[(j+1)%4, 0], corners[(j+1)%4, 1]], 
                            fill=(255, 0, 0), width=2)



            
    
        # Display the image
        plt.figure(figsize=(10, 8))
        plt.imshow(np.array(img))
        plt.title(f"Sample {batch_idx}, Image ID: {info['img_id'].item()}")
        plt.axis('off')
        #     # if targets['labels'][0, i] > 0:  # If there's an object
        #         # Get the 3D box information
        #     box_3d = targets['boxes_3d'][0, i].numpy()
        #     h, w, l = targets['src_size_3d'][0, i].numpy()
        #     # Get depth
        #     depth_val = float(targets['depth'][0, i])
        #     # Get heading angle
        #     bin_id, res = int(targets['heading_bin'][0, i]), float(targets['heading_res'][0, i])
        #     heading_angle = class2angle(bin_id, res)
            
        #     # Calculate 3D position from boxes_3d and depth
        #     cx, cy, left, right, top, bottom = targets['boxes_3d'][0, i].numpy()
        #     img_size = np.array(img.size)
        #     cx, cy = cx * img_size[0], cy * img_size[1]
        #     # box_corner = box_center_to_corner(box_3d)
        #     # print('box_corner', box_corner)
        #     # img = draw_projected_box3d(img, box_corner)
                
        #     # Draw the box
        #     img = draw_3d_box_from_labels(img, box_3d, color=(255, 0, 0), thickness=2)
        
        # # Display the image
        # plt.figure(figsize=(10, 8))
        # plt.imshow(np.array(img))
        # plt.title(f"Sample {batch_idx}, Image ID: {info['img_id'].item()}")
        # plt.axis('off')
        img_id = info['img_id'].item()
        output_dir = '/home/elenagovi/repos/MonoDETR/lib/datasets/kitti/debug/'
        save_path = os.path.join(output_dir, f"sample_{batch_idx}_id_{img_id}.png")
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()
        if batch_idx>50:
            break

    # print ground truth fisrt
    objects = dataset.get_label(0)
    for object in objects:
        print(object.to_kitti_format())
