import glob
import os
import yaml

import cv2
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from utils.noise import NOISE_GENERATORS, NOISE_POSTPROCESSORS, NoiseSelector

mean_train = [0.485, 0.456, 0.406]
std_train = [0.229, 0.224, 0.225]
MAX_TRY_NUMBER = 200

with open('config.yaml', 'r') as f:
    conf = yaml.safe_load(f)
    DATA_ROOT = conf.get('DATA_ROOT','../datasets/')
    MVTEC2D_DIR = conf.get('MVTEC2D_DIR', 'mvtec_anomaly_detection')
    MVTEC3D_DIR = conf.get('MVTEC3D_DIR', 'mvtec_3d_anomaly_detection')

mvtec2d_classes = ['carpet', 'grid', 'leather', 'tile', 'wood',
                   'bottle', 'cable', 'capsule', 'hazelnut', 'metal_nut', 'pill',
                   'screw', 'toothbrush', 'transistor', 'zipper']

mvtec3d_classes = ['bagel', 'cable_gland', 'carrot', 'cookie', 'dowel',
                   'peach', 'potato', 'rope', 'tire', 'foam']

dataset_classes = {
    'mvtec2d': mvtec2d_classes,
    'mvtec3d': mvtec3d_classes
}

SKIP_BACKGROUND = {
    'mvtec2d': {
        'carpet': False,
        'grid': False,
        'leather': False,
        'tile': False,
        'wood': False,
        'bottle': True,
        'cable': False,
        'capsule': False,
        'hazelnut': True,
        'metal_nut': True,
        'pill': True,
        'screw': True,
        'toothbrush': True,
        'transistor': False,
        'zipper': True
    },

    'mvtec3d': {
        'bagel': False,
        'cable_gland': False,
        'carrot': False,
        'cookie': False,
        'dowel': False,
        'peach': False,
        'potato': False,
        'rope': False,
        'tire': False,
        'foam': False
    },
}


def load_mvtec2d(category):
    assert category in mvtec2d_classes

    test_img_path = os.path.join(DATA_ROOT, MVTEC2D_DIR, category, 'test')
    train_img_path = os.path.join(DATA_ROOT, MVTEC2D_DIR, category, 'train')
    ground_truth_path = os.path.join(DATA_ROOT, MVTEC2D_DIR, category, 'ground_truth')

    def load_mvtec2d_phase(root_path, gt_path):
        img_tot_paths = []
        gt_tot_paths = []
        tot_labels = []
        tot_types = []

        defect_types = os.listdir(root_path)

        for defect_type in defect_types:
            if defect_type == 'good':
                img_paths = glob.glob(os.path.join(root_path, defect_type) + "/*.png")
                img_tot_paths.extend(img_paths)
                gt_tot_paths.extend([0] * len(img_paths))
                tot_labels.extend([0] * len(img_paths))
                tot_types.extend(['good'] * len(img_paths))
            else:
                img_paths = glob.glob(os.path.join(root_path, defect_type) + "/*.png")
                gt_paths = [os.path.join(gt_path, defect_type, os.path.basename(s)[:-4] + '_mask.png') for s in
                            img_paths]
                img_paths.sort()
                gt_paths.sort()
                img_tot_paths.extend(img_paths)
                gt_tot_paths.extend(gt_paths)
                tot_labels.extend([1] * len(img_paths))
                tot_types.extend([defect_type] * len(img_paths))

        assert len(img_tot_paths) == len(gt_tot_paths), "Something wrong with test and ground truth pair!"

        return img_tot_paths, gt_tot_paths, tot_labels, tot_types

    return load_mvtec2d_phase(train_img_path, ground_truth_path), load_mvtec2d_phase(test_img_path, ground_truth_path)


def load_mvtec3d(category):
    assert category in mvtec3d_classes

    test_img_path = os.path.join(DATA_ROOT, MVTEC3D_DIR, category, 'test')
    train_img_path = os.path.join(DATA_ROOT, MVTEC3D_DIR, category, 'train')

    def load_mvtec3d_phase(root_path):
        img_tot_paths = []
        gt_tot_paths = []
        tot_labels = []
        tot_types = []

        defect_types = os.listdir(root_path)

        for defect_type in defect_types:
            if defect_type == 'good':
                img_paths = glob.glob(os.path.join(root_path, defect_type, 'rgb') + "/*.*")
                img_tot_paths.extend(img_paths)
                gt_tot_paths.extend([0] * len(img_paths))
                tot_labels.extend([0] * len(img_paths))
                tot_types.extend(['good'] * len(img_paths))
            else:
                img_paths = glob.glob(os.path.join(root_path, defect_type, 'rgb') + "/*.*")
                gt_paths = glob.glob(os.path.join(root_path, defect_type, 'gt') + "/*.*")
                img_paths.sort()
                gt_paths.sort()
                img_tot_paths.extend(img_paths)
                gt_tot_paths.extend(gt_paths)
                tot_labels.extend([1] * len(img_paths))
                tot_types.extend([defect_type] * len(img_paths))

        assert len(img_tot_paths) == len(gt_tot_paths), "Something wrong with test and ground truth pair!"

        return img_tot_paths, gt_tot_paths, tot_labels, tot_types

    return load_mvtec3d_phase(train_img_path), load_mvtec3d_phase(test_img_path)


load_function_dict = {
    'mvtec2d': load_mvtec2d,
    'mvtec3d': load_mvtec3d,
}


class CDODataset(Dataset):
    epoch_ratio = 0
    augm_red_type = ""

    def __init__(self, dataset_name, category, input_size, phase,
                 load_memory=False, perturbed=False, augm_red = {"normal": {}}):

        assert dataset_name in list(load_function_dict.keys())

        self.load_function = load_function_dict[dataset_name]
        self.skip_bkg = SKIP_BACKGROUND[dataset_name][category]
        self.phase = phase
        self.perturbed = perturbed
        self.augm_red = augm_red
        self.NoiseSelector = "NoiseSelector" in self.augm_red

        if phase == 'test':
            self.perturbed = False

        self.transform = transforms.Compose([
            transforms.Resize((input_size, input_size), Image.LANCZOS),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean_train,
                                 std=std_train)])

        self.gt_transform = transforms.Compose([
            transforms.Resize((input_size, input_size), Image.NEAREST),
            transforms.CenterCrop(input_size),
            transforms.ToTensor()])

        self.load_memory = load_memory
        self.category = category
        self.resize_shape = (input_size, input_size)
        self.h = input_size
        self.w = input_size

        # noise generators
        if self.NoiseSelector:
            NoiseSelector.reset(size=self.resize_shape, ch=3)
        else:
            self.augm_red_type = set(augm_red.keys())
            gen_str = self.augm_red_type.intersection(set(NOISE_GENERATORS.keys()))
            gen = gen_str.pop() if len(gen_str) > 0 else "normal"
            self.noise_gen_name = gen
            self.noise_gen = NOISE_GENERATORS[gen](size=self.resize_shape, ch=3)
            if "controller" in self.augm_red.get(gen, ""):
                controller_args = self.augm_red[gen]["controller_args"]
                self.noise_gen.setController(self.augm_red[gen]["controller"](*controller_args))

            noise_post_str = self.augm_red_type.intersection(set(NOISE_POSTPROCESSORS.keys()))
            self.noise_postprocessor = {noise_post: NOISE_POSTPROCESSORS[noise_post](*self.augm_red[noise_post]["args"]) for noise_post in noise_post_str}
            for noise_post in self.noise_postprocessor:
                if "controller" in self.augm_red.get(noise_post, ""):
                    controller_args = self.augm_red[noise_post]["controller_args"]
                    self.noise_postprocessor[noise_post].setController(self.augm_red[noise_post]["controller"](*controller_args))

        # load datasets
        self.img_paths, self.gt_paths, self.labels, self.types = self.load_dataset()  # self.labels => good : 0, anomaly : 1

        if self.load_memory:
            self.load_dataset_to_memory()

    def get_size(self):
        return self.h, self.w

    def estimate_background(self, image, thr_low=30, thr_high=220):

        gray_image = np.mean(image * 255, axis=2)

        bkg_msk_high = np.where(gray_image > thr_high, np.ones_like(gray_image), np.zeros_like(gray_image))
        bkg_msk_low = np.where(gray_image < thr_low, np.ones_like(gray_image), np.zeros_like(gray_image))

        bkg_msk = np.bitwise_or(bkg_msk_low.astype(np.uint8), bkg_msk_high.astype(np.uint8))
        bkg_msk = cv2.medianBlur(bkg_msk, 7)
        kernel = np.ones((19, 19), np.uint8)
        bkg_msk = cv2.dilate(bkg_msk, kernel)

        bkg_msk = bkg_msk.astype(np.float32)
        return bkg_msk

    def patch_mask(self, bkg_msk, shape):
        patch_mask = np.zeros(shape[:2], dtype=np.float32)
        patch_number = np.random.randint(0, 5)

        for i in range(patch_number):
            try_count = 0
            coor_min_dim1 = 0
            coor_min_dim2 = 0

            coor_max_dim1 = 0
            coor_max_dim2 = 0
            while try_count < MAX_TRY_NUMBER:
                try_count += 1

                patch_dim1 = np.random.randint(self.h // 40, self.h // 10)
                patch_dim2 = np.random.randint(self.w // 40, self.w // 10)

                center_dim1 = np.random.randint(patch_dim1, shape[0] - patch_dim1)
                center_dim2 = np.random.randint(patch_dim2, shape[1] - patch_dim2)

                if self.skip_bkg:
                    if bkg_msk[center_dim1, center_dim2] > 0:
                        continue

                coor_min_dim1 = np.clip(center_dim1 - patch_dim1, 0, shape[0])
                coor_min_dim2 = np.clip(center_dim2 - patch_dim2, 0, shape[1])

                coor_max_dim1 = np.clip(center_dim1 + patch_dim1, 0, shape[0])
                coor_max_dim2 = np.clip(center_dim2 + patch_dim2, 0, shape[1])

                break

            patch_mask[coor_min_dim1:coor_max_dim1, coor_min_dim2:coor_max_dim2] = 1.0
        
        return patch_mask

    def augment_image(self, image):

        # get bkg mask
        bkg_msk = self.estimate_background(image)

        # generate random mask
        patch_mask = self.patch_mask(bkg_msk, image.shape)

        if self.NoiseSelector:
            augmented_image = NoiseSelector.augment(image, patch_mask)
        else:
            # generate noise image
            noise_image = self.noise_gen(self.epoch_ratio)

            augmented_image = image

            if "alpha" in self.augm_red_type:
                blended_noise = self.noise_postprocessor["alpha"](self.epoch_ratio, noise_image[patch_mask > 0], augmented_image[patch_mask > 0])
                augmented_image[patch_mask > 0] = blended_noise
            else:
                augmented_image[patch_mask > 0] = noise_image[patch_mask > 0]

        patch_mask = patch_mask[:, :, np.newaxis]

        if patch_mask.max() > 0:
            has_anomaly = 1.0
        else:
            has_anomaly = 0.0

        return augmented_image, patch_mask, np.array([has_anomaly], dtype=np.float32)

    def transform_image(self, image):
        image = np.array(image).reshape((image.shape[0], image.shape[1], image.shape[2])).astype(np.float32) / 255.0
        augmented_image, anomaly_mask, has_anomaly = self.augment_image(image)

        image = (image * 255.0).astype(np.uint8)
        augmented_image = (augmented_image * 255.0).astype(np.uint8)
        anomaly_mask = (anomaly_mask[:, :, 0] * 255.0).astype(np.uint8)

        return image, augmented_image, anomaly_mask, has_anomaly

    def load_dataset(self):

        (train_img_tot_paths, train_gt_tot_paths, train_tot_labels, train_tot_types), \
        (test_img_tot_paths, test_gt_tot_paths, test_tot_labels, test_tot_types) = self.load_function(self.category)

        if self.phase == 'train':
            return train_img_tot_paths, train_gt_tot_paths, train_tot_labels, train_tot_types
        else:
            return test_img_tot_paths, test_gt_tot_paths, test_tot_labels, test_tot_types

    def __len__(self):
        return len(self.img_paths)

    def load_dataset_to_memory(self):

        self.img = []
        self.gt = []
        self.img_name = []
        self.img_type = []

        for idx in range(self.__len__()):
            img_path, gt, label, img_type = self.img_paths[idx], self.gt_paths[idx], self.labels[idx], self.types[idx]
            img = Image.open(img_path).convert('RGB')
            img = np.array(img)
            img = cv2.resize(img, dsize=(self.resize_shape[1], self.resize_shape[0]))
            if gt == 0:
                gt = np.zeros([img.shape[0], img.shape[0]])
            else:
                gt = Image.open(gt)
                gt = np.array(gt)
                gt[gt > 0] = 255
                gt = cv2.resize(gt, dsize=(self.resize_shape[1], self.resize_shape[0]), interpolation=cv2.INTER_NEAREST)

            self.img.append(img)
            self.gt.append(gt)
            self.img_name.append(f'{self.category}_{img_type}_{os.path.basename(img_path[:-4])}')
            self.img_type.append(img_type)

    def __getitem__(self, idx):

        if self.load_memory:
            img = self.img[idx]
            gt = self.gt[idx]
            label = self.labels[idx]
            name = self.img_name[idx]

        else:
            img_path, gt, label, img_type = self.img_paths[idx], self.gt_paths[idx], self.labels[idx], self.types[idx]

            img = Image.open(img_path).convert('RGB')
            img = np.array(img)
            img = cv2.resize(img, dsize=(self.resize_shape[1], self.resize_shape[0]))
            if gt == 0:
                gt = np.zeros([img.shape[0], img.shape[1]])
            else:
                gt = Image.open(gt)
                gt = np.array(gt)
                gt[gt > 0] = 255
                gt = cv2.resize(gt, dsize=(self.resize_shape[1], self.resize_shape[0]), interpolation=cv2.INTER_NEAREST)

            name = f'{self.category}_{img_type}_{os.path.basename(img_path[:-4])}'

        if self.perturbed:
            img, augmented_image, anomaly_mask, _ = self.transform_image(img)
        else:
            augmented_image = np.zeros_like(img)
            anomaly_mask = np.zeros((img.shape[0], img.shape[1]))

        img = Image.fromarray(img)
        augmented_image = Image.fromarray(augmented_image)
        anomaly_mask = Image.fromarray(anomaly_mask)
        gt = Image.fromarray(gt)

        img = self.transform(img)
        augmented_image = self.transform(augmented_image)
        anomaly_mask = self.gt_transform(anomaly_mask)
        gt = self.gt_transform(gt)

        if self.perturbed:
            img = augmented_image
            gt = anomaly_mask

        return img, gt, label, name
