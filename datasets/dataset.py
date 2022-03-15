#!/usr/bin/python3
# -*- coding: utf-8 -*-

import numpy as np
import numpy.ma as ma
import open3d as o3d
import cv2
import random
import os
import re
import json
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from IPython import embed


class BOPDataset(data.Dataset):
    def __init__(self, root, dataset_name, mode='test'):
        assert mode in ['test', 'train']
        self.root = root
        self.dataset_name = dataset_name
        self.mode = mode
        print('Init {} dataset in {} mode.'.format(dataset_name, self.mode))

        # load model and config info
        # model is open3d.geometry.TriangleMesh object
        self.model_info = self.__init_model_info()
        self.model_names = []
        for key in self.model_info:
            self.model_names.append(key)

        self.mask_combined = False
        self.data_info = self.__init_data_info(self.mode)

        self.test_targets = None
        if self.mode == 'test':
            self.test_targets = self.__init_test_targets()

    def get_camera_info(self):
        """ Return camera info of the dataset

        Returns:
            List[List]: List contains (height, width, fx, fy, cx, cy, depth_scale)
        """
        path = os.path.join(self.root, self.dataset_name)
        files = os.listdir(path)
        camera_infos = []
        for name in files:
            if name.startswith('camera'):
                f = open(os.path.join(path, name))
                camera_info = json.load(f)
                f.close()

                camera = []
                camera.append(camera_info['height'])
                camera.append(camera_info['width'])
                camera.append(camera_info['fx'])
                camera.append(camera_info['fy'])
                camera.append(camera_info['cx'])
                camera.append(camera_info['cy'])
                camera.append(camera_info['depth_scale'])
                camera_infos.append(camera)
        return camera_infos

    def __init_test_targets(self):
        try:
            f = open(
                os.path.join(self.root, self.dataset_name,
                             'test_targets.json'))
            targets = json.load(f)
            f.close()
        except:
            targets = None
            print('Load test targets json fail.')
        return targets

    def __init_model_info(self):
        model_path = os.path.join(self.root, self.dataset_name, 'models')
        model_list = {}
        try:
            f = open(os.path.join(model_path, 'models_info.json'))
            model_list = json.load(f)
            str_list = os.listdir(model_path)
            model_names = [
                os.path.join(model_path, s) for s in str_list
                if s.endswith('.ply')
            ]
            model_names.sort(key=lambda x: int(re.findall(r'\d+', x)[-1]))
            print('Load {} models.'.format(len(model_names)))
            self.num_models = len(model_names)
            f.close()

            for i, (key, val) in enumerate(model_list.items()):
                mesh = o3d.io.read_triangle_mesh(model_names[i])
                mesh = mesh.scale(0.001, np.array([0, 0, 0]))
                model_list[key]['mesh'] = mesh

                cloud = o3d.io.read_point_cloud(model_names[i])
                cloud = cloud.scale(0.001, np.array([0, 0, 0]))
                model_list[key]['cloud'] = cloud

        except:
            print('Load model info json fail.')

        return model_list

    def __init_data_info(self, mode='test'):
        path = os.path.join(self.root, self.dataset_name, mode)
        lists = os.listdir(path)
        lists.sort(key=lambda x: int(re.findall(r'\d+', x)[-1]))

        data_dict = {}
        for test_id in lists:
            d = {}
            d['rgb'] = os.path.join(path, test_id, 'rgb')
            d['depth'] = os.path.join(path, test_id, 'depth')
            if os.path.exists(os.path.join(path, test_id,
                                           'mask_visib')) is False:
                d['mask'] = os.path.join(path, test_id, 'mask')
            else:
                d['mask'] = os.path.join(path, test_id, 'mask_visib')
            f = open(os.path.join(path, test_id, 'scene_camera.json'))
            d['scene_camera'] = json.load(f)
            f.close()
            f = open(os.path.join(path, test_id, 'scene_gt.json'))
            d['scene_gt'] = json.load(f)
            f.close()
            f = open(os.path.join(path, test_id, 'scene_gt_info.json'))
            d['scene_gt_info'] = json.load(f)
            f.close()
            data_dict[test_id] = d
            print('Scene {} has {} {} data'.format(test_id,
                                                   len(d['scene_camera']),
                                                   mode))
        sample = os.listdir(data_dict[test_id]['mask'])[0]
        if '_' not in sample:
            self.mask_combined = True

        return data_dict

    def get_test_targets(self):
        """Return the test targets info

        Returns:
            [List[Dict]]: 
        """
        return self.test_targets

    def get_model_names(self):
        """Return the names of the models.

        Returns:
            [List[string]]: 
        """
        return self.model_names

    def get_model(self, model_idx='1', model_type='cloud'):
        """ Return a model given id

        Args:
            model_idx (string, optional): name of the model. Defaults to '1'.
            model_type (string, optional): type of the model. Defaults to 'cloud'.

        Returns:
            [open3d.geometry.TriangleMesh or open3d.geometry.PointCloud]: model
        """
        assert model_type in ['cloud', 'mesh']

        if len(self.model_info) == 0:
            print('No model in {} dataset.'.format(self.dataset_name))
            return None

        if model_idx not in self.model_info:
            print('The name of model is not in the model config.')
            return None

        model = self.model_info[model_idx][model_type]
        return model

    def get_single_data(self, scene, name):
        """ Return single test data given id and data name 

        Args:
            scene ([string]): scene name 
            name ([string]): data name

        Returns:
            [np.ndarray, np.ndarray, tuple[5], List]: 
            (rgb, depth, (fx, fy, cx, cy, depth_scale), obj_infos)
        """
        name_no_zero = str(int(name))
        length = name.__len__()
        if name_no_zero not in self.data_info[scene][
                'scene_camera'] or scene not in self.data_info:
            print('No data found.')
            return None, None, None, None, None

        scene_data_dict = self.data_info[scene]
        data_name = name + '.png'

        img = cv2.imread(os.path.join(scene_data_dict['rgb'], data_name))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        depth = cv2.imread(os.path.join(scene_data_dict['depth'], data_name),
                           cv2.IMREAD_ANYDEPTH)

        shape = img.shape

        # check mask is combined or separated
        mask_combined = None
        if self.mask_combined:
            mask_combined = cv2.imread(
                os.path.join(scene_data_dict['mask'], data_name),
                cv2.IMREAD_ANYDEPTH)
            # check mask data type
            if str(mask_combined.dtype) != 'uint16':
                print('Mask is not uint16.')
                return None, None, None, None, None

        K = scene_data_dict['scene_camera'][name_no_zero]['cam_K']
        depth_scale = 1.0

        if self.dataset_name == 'ycbv':
            depth_scale *= 10000
        else:
            depth_scale *= 1000

        intrin = (K[0], K[4], K[2], K[5], depth_scale)

        obj_infos = []
        class_id = {}
        for obj in range(len(scene_data_dict['scene_gt_info'][name_no_zero])):
            gt_info = scene_data_dict['scene_gt_info'][name_no_zero][obj]
            gt = scene_data_dict['scene_gt'][name_no_zero][obj]
            obj_info = {}

            if 'bbox_visib' not in gt_info:
                bbox = gt['bbox_obj']
            else:
                bbox = gt_info['bbox_visib']
            for i, val in enumerate(bbox):
                bbox[i] = max(0, val)

            bbox_out = (bbox[0], bbox[1], bbox[2] + bbox[0], bbox[3] + bbox[1])
            obj_info['bbox'] = bbox_out

            R = gt['cam_R_m2c']
            t = gt['cam_t_m2c']
            gt_pose = np.identity(4)
            gt_pose[:3, :3] = np.array(R).reshape((3, 3))
            gt_pose[:3, 3] = np.array(t) / 1000
            obj_info['gt_pose'] = gt_pose
            obj_id = gt['obj_id']

            if obj_id in class_id:
                class_id[obj_id] += 1
            else:
                class_id[obj_id] = 1
            obj_info['obj_id'] = obj_id
            obj_info['inst_count'] = class_id[obj_id]

            if self.mask_combined is False:
                mask_name = os.path.join(
                    scene_data_dict['mask'],
                    name + '_' + str(obj).zfill(length) + '.png')
                m = cv2.imread(mask_name, cv2.IMREAD_ANYDEPTH)
                indice = np.where(m == 255)
                indice_list = [
                    indice[0][i] * shape[1] + indice[1][i]
                    for i in range(len(indice[0]))
                ]
                obj_info['mask'] = indice_list
            obj_infos.append(obj_info)

        return img, depth, intrin, obj_infos

    def get_scene_models(self, scene, model_type='cloud'):
        """ Return all models in current scene

        Args:
            scene (str): name of scene folder
            model_type (str, optional): _description_. Defaults to 'cloud'.

        Returns:
            Dict: _description_
        """
        data_info_gt = self.data_info[scene]['scene_gt']
        model_names = []
        for sample in data_info_gt:
            for instance in data_info_gt[sample]:
                if not self.mask_combined:
                    model_names.append(instance['obj_id'])
                else:
                    pass
        model_names = set(model_names)
        models = {}
        for name in model_names:
            model = self.get_model(str(name), model_type)
            models[name] = model

        return models


class DenseFusionDataset(BOPDataset):
    def __init__(self,
                 root,
                 dataset_name,
                 mode='test',
                 points_num=(500, 500, 500),
                 noise_t=None,
                 sym_list=[],
                 refine=True):
        super(DenseFusionDataset, self).__init__(root, dataset_name, mode)
        self.noise_t = noise_t if noise_t is not None else 0.0
        self.norm = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        self.points_num = points_num[0]
        self.small_model_num = points_num[1]
        self.large_model_num = points_num[2]
        self.sym_list = sym_list
        self.refine = refine

        self.xmap = np.array([[j for i in range(640)] for j in range(480)])
        self.ymap = np.array([[i for i in range(640)] for j in range(480)])

        self.border_list = [
            -1, 40, 80, 120, 160, 200, 240, 280, 320, 360, 400, 440, 480, 520,
            560, 600, 640, 680
        ]

        self.list_rgb = []
        self.list_depth = []
        self.list_mask = []
        self.list_obj = []
        self.list_data_name = []
        self.list_gt = []

        self.item_count = 0

        for scene_name in self.data_info:
            scene_data_dict = self.data_info[scene_name]
            for sample in scene_data_dict['scene_gt']:
                sample_name = sample.zfill(6) + '.png'
                gt_info_dict = scene_data_dict['scene_gt_info'][sample]
                camera_dict = scene_data_dict['scene_camera'][sample]
                for i, instance in enumerate(
                        scene_data_dict['scene_gt'][sample]):
                    self.list_rgb.append(
                        os.path.join(scene_data_dict['rgb'], sample_name))
                    self.list_depth.append(
                        os.path.join(scene_data_dict['depth'], sample_name))
                    if self.mask_combined:
                        # TODO: should be implemented based on final structure
                        self.list_mask.append(
                            os.path.join(scene_data_dict['mask'], sample_name))
                        obj_id = instance['inst_id']
                        self.list_obj.append(obj_id)
                    else:
                        obj_id = instance['obj_id']
                        mask_name = sample.zfill(6) + '_' + str(i).zfill(
                            6) + '.png'
                        self.list_mask.append(
                            os.path.join(scene_data_dict['mask'], mask_name))
                        self.list_obj.append(obj_id)

                    gt = instance
                    if 'bbox_visib' in gt_info_dict[i]:
                        gt['bbox'] = gt_info_dict[i]['bbox_visib']
                    else:
                        gt['bbox'] = gt_info_dict[i]['bbox_obj']
                    K = camera_dict['cam_K']
                    gt['cam_K'] = [K[0], K[4], K[2], K[5]]

                    self.list_gt.append(gt)

                    self.item_count += 1
                    self.list_data_name.append(sample)

    def __getitem__(self, index):
        img = cv2.imread(self.list_rgb[index])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        depth = cv2.imread(self.list_depth[index], cv2.IMREAD_ANYDEPTH)
        mask = cv2.imread(self.list_mask[index], cv2.IMREAD_ANYDEPTH)
        obj = self.list_obj[index]
        data_name = self.list_data_name[index]
        gt = self.list_gt[index]

        mask_depth = ma.getmaskarray(ma.masked_not_equal(depth, 0))
        mask_label = ma.getmaskarray(ma.masked_equal(mask, np.array(255)))

        valid_mask = mask_depth * mask_label

        img_masked = np.transpose(img, (2, 0, 1))
        rmin, rmax, cmin, cmax = self.get_bbox(gt['bbox'], self.border_list)
        img_masked = img_masked[:, rmin:rmax, cmin:cmax]

        target_r = np.resize(np.array(gt['cam_R_m2c']), (3, 3))
        target_t = np.array(gt['cam_t_m2c']) / 1000.0
        noise_t = np.array(
            [random.uniform(-self.noise_t, self.noise_t) for i in range(3)])

        choose = valid_mask[rmin:rmax, cmin:cmax].flatten().nonzero()[0]
        if len(choose) == 0:
            cc = torch.LongTensor([0])
            return (cc, cc, cc, cc, cc, cc)

        if len(choose) > self.points_num:
            c_mask = np.zeros(len(choose), dtype=int)
            c_mask[:self.points_num] = 1
            np.random.shuffle(c_mask)
            choose = choose[c_mask.nonzero()]
        else:
            choose = np.pad(choose, (0, self.points_num - len(choose)), 'wrap')
        depth_masked = depth[rmin:rmax,
                             cmin:cmax].flatten()[choose][:,
                                                          np.newaxis].astype(
                                                              np.float32)

        xmap_masked = self.xmap[
            rmin:rmax,
            cmin:cmax].flatten()[choose][:, np.newaxis].astype(np.float32)
        ymap_masked = self.ymap[
            rmin:rmax,
            cmin:cmax].flatten()[choose][:, np.newaxis].astype(np.float32)
        choose = np.array([choose])

        fx, fy, cx, cy = gt['cam_K']
        depth_scale = 10000.0 if self.dataset_name == 'ycbv' else 1000.0
        pt2 = depth_masked / depth_scale
        pt0 = (ymap_masked - cx) * pt2 / fx
        pt1 = (xmap_masked - cy) * pt2 / fy
        cloud = np.concatenate((pt0, pt1, pt2), axis=1)
        if self.mode == 'train':
            cloud = np.add(cloud, noise_t)

        model_pcd = self.get_model(str(obj))
        total_num = len(model_pcd.points)
        dellist = [j for j in range(0, total_num)]
        # TODO: use fps here
        if self.refine:
            ratio = self.large_model_num / len(model_pcd.points)
            dellist = random.sample(dellist, total_num - self.large_model_num)
        else:
            dellist = random.sample(dellist, total_num - self.small_model_num)
        model_points = np.delete(np.asarray(model_pcd.points), dellist, axis=0)
        # model_pcd = model_pcd.get_farthest_point_sample(self.points_num)

        target = np.dot(model_points, target_r.T)
        if self.mode == 'train':
            target = np.add(target, target_t + noise_t)
        else:
            target = np.add(target, target_t)

        return torch.from_numpy(cloud.astype(np.float32)), \
               torch.LongTensor(choose.astype(np.int32)), \
               self.norm(torch.from_numpy(img_masked.astype(np.float32))), \
               torch.from_numpy(target.astype(np.float32)), \
               torch.from_numpy(model_points.astype(np.float32)), \
               torch.LongTensor([obj - 1])

    def __len__(self):
        return self.item_count

    def get_sym_list(self):
        return self.sym_list

    def get_num_points_model(self):
        if self.refine:
            return self.large_model_num
        else:
            return self.small_model_num

    @staticmethod
    def get_bbox(bbox, border_list):
        bbx = [bbox[1], bbox[1] + bbox[3], bbox[0], bbox[0] + bbox[2]]
        if bbx[0] < 0:
            bbx[0] = 0
        if bbx[1] >= 480:
            bbx[1] = 479
        if bbx[2] < 0:
            bbx[2] = 0
        if bbx[3] >= 640:
            bbx[3] = 639
        rmin, rmax, cmin, cmax = bbx[0], bbx[1], bbx[2], bbx[3]
        r_b = rmax - rmin
        for tt in range(len(border_list)):
            if r_b > border_list[tt] and r_b < border_list[tt + 1]:
                r_b = border_list[tt + 1]
                break
        c_b = cmax - cmin
        for tt in range(len(border_list)):
            if c_b > border_list[tt] and c_b < border_list[tt + 1]:
                c_b = border_list[tt + 1]
                break
        center = [int((rmin + rmax) / 2), int((cmin + cmax) / 2)]
        rmin = center[0] - int(r_b / 2)
        rmax = center[0] + int(r_b / 2)
        cmin = center[1] - int(c_b / 2)
        cmax = center[1] + int(c_b / 2)
        if rmin < 0:
            delt = -rmin
            rmin = 0
            rmax += delt
        if cmin < 0:
            delt = -cmin
            cmin = 0
            cmax += delt
        if rmax > 480:
            delt = rmax - 480
            rmax = 480
            rmin -= delt
        if cmax > 640:
            delt = cmax - 640
            cmax = 640
            cmin -= delt
        return rmin, rmax, cmin, cmax


if __name__ == '__main__':
    from IPython import embed
    def np2o3d(xyz, normals=None):
        """convert numpy ndarray to open3D point cloud 

        Args:
            xyz ([np.ndarray]): [description]
            normals ([np.ndarray], optional): [description]. Defaults to None.

        Returns:
            [type]: [description]
        """

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(xyz)
        if normals is not None:
            pcd.normals = o3d.utility.Vector3dVector(normals)
        return pcd

    dataset = DenseFusionDataset('/home/yuecideng/WorkSpace/Data/BOPDataset',
                                 'lm',
                                 'train',
                                 points_num=(500, 500, 500))
    loader = torch.utils.data.DataLoader(dataset,
                                         batch_size=1,
                                         shuffle=False,
                                         num_workers=8)
    for i, data in enumerate(loader, 0):
        points, choose, img, target, model_points, idx = data

        if len(choose[0][0]) == 1:
            print('no data')
            continue
        img = np.squeeze(img.numpy())
        img = np.transpose(img, (1, 2, 0))

        points = np.squeeze(points.numpy())
        pcd = np2o3d(points)
        pcd.paint_uniform_color([0, 1, 0])
        target = np2o3d(np.squeeze(target.numpy()))
        target.paint_uniform_color([0.5, 0.5, 0.5])
        o3d.visualization.draw_geometries([pcd, target])