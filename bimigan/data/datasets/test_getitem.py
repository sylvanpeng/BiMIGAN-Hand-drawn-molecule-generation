# -*- coding:utf-8 -*-
"""
author：Sylvan Peng
date�?024�?3�?1�?
version�?.0
"""
import os
import numpy as np
from torchvision.datasets.folder import default_loader
from torch.utils.data import Dataset
from torchvision.datasets.folder import IMG_EXTENSIONS
from funcs import load_images, sample_image, load_smiles


def apply_if_not_none(fn, x):
    if x is None:
        return None

    return fn(x)

def load_images(paths, transform = None):
    result = [ apply_if_not_none(default_loader, x) for x in paths ]

    if transform is not None:
        result = [ apply_if_not_none(transform, x) for x in result ]

    return result

def load_smiles_mapping(self, path):
    smiles_mapping = {}
    with open(path, 'r') as file:
        for line in file:
            parts = line.strip().split(',')  # 或使用其他分隔符，如空格：split(' ')
            if len(parts) == 2:
                image_path, smiles = parts
                smiles_mapping[image_path] = smiles
    return smiles_mapping


def load_smiles(paths, smiles_paths):
    # 分别加载两个SMILES映射文件
    smiles_mapping_a = load_smiles_mapping(smiles_paths[0])
    smiles_mapping_b = load_smiles_mapping(smiles_paths[1])

    # 初始化结果列�?
    result = []

    # 为paths[0]的图像路径查找SMILES，使用smiles_mapping_a映射
    image_name_a = os.path.basename(paths[0])
    smiles_a = smiles_mapping_a.get(image_name_a)
    if smiles_a:
        result.append(smiles_a)
    else:
        result.append(None)  # 如果没有找到匹配的SMILES，可以添加None或相应的占位�?

    # 为paths[1]的图像路径查找SMILES，使用smiles_mapping_b映射
    image_name_b = os.path.basename(paths[1])
    smiles_b = smiles_mapping_b.get(image_name_b)
    if smiles_b:
        result.append(smiles_b)
    else:
        result.append(None)  # 如果没有找到匹配的SMILES，可以添加None或相应的占位�?

    # 返回包含两张图片对应的SMILES字符串的列表，顺序与paths参数一�?
    return result


def find_images_in_dir(path):
    extensions = set(IMG_EXTENSIONS)

    result = []
    for fname in os.listdir(path):
        fullpath = os.path.join(path, fname)

        if not os.path.isfile(fullpath):
            continue

        ext = os.path.splitext(fname)[1]
        if ext not in extensions:
            continue

        result.append(fullpath)

    result.sort()
    return result


def _sample_image(self, images, index):
    randomize = (self._is_train and (not self._align_train))
    return sample_image(images, index, self._prg, randomize)



def main():
    _path_a = r"D:\workspace\datasets\test_fold\trainA"
    _path_b = r"D:\workspace\datasets\test_fold\trainB"
    _imgs_a = find_images_in_dir(_path_a)
    _imgs_b = find_images_in_dir(_path_b)
    _transform = None
    for index in range[0:2]:
        path_a = _sample_image(_imgs_a,index)
        path_b = _sample_image(_imgs_b,index)
        images = load_images([path_a, path_b], _transform)
        smiles_path_a = os.path.join(_path_a, "smiles.txt")
        smiles_path_b = os.path.join(_path_a, "smiles.txt")
        # 假设self._smiles_a和self._smiles_b是与图像对应的SMILES数据
        smiles = load_smiles([path_a, path_b], [smiles_path_a, smiles_path_b])

        sample = {'images': images, 'smiles': smiles}
        print(sample)


if __name__ == "__main__":
    main()
