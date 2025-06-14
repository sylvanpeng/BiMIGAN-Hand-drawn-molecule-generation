import os
from torchvision.datasets.folder import default_loader
import csv

def sample_image(images, index, prg, randomize = False):
    if randomize:
        return prg.choice(images, size = 1)[0]

    if index >= len(images):
        return None

    return images[index]

def apply_if_not_none(fn, x):
    if x is None:
        return None

    return fn(x)

def load_images(paths, transform = None):
    result = [ apply_if_not_none(default_loader, x) for x in paths ]
    csv_path = os.path.expanduser('~/image_dimensions_origin.csv')
    with open(csv_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        # 写入表头
        writer.writerow(['Width', 'Height', 'Channels'])

        # 对于每个图像，获取其尺寸和通道数并写入CSV
        for image in result:
            width, height = image.size
            channels = len(image.getbands())
            writer.writerow([width, height, channels])
    if transform is not None:
        result = [ apply_if_not_none(transform, x) for x in result ]

    # 打开一个CSV文件准备写入
    # csv_path = os.path.expanduser('~/image_dimensions_after.csv')
    # with open(csv_path, mode='w', newline='') as file:
    #     writer = csv.writer(file)
    #     # 写入表头
    #     writer.writerow(['Width', 'Height', 'Channels'])
    #
    #     # 对于每个图像，获取其尺寸和通道数并写入CSV
    #     for image in result:
    #         width, height = image.size
    #         channels = len(image.getbands())
    #         writer.writerow([width, height, channels])
    return result
#测试版本
# def load_images(paths, transform=None):
#     # 加载图片
#     result = [apply_if_not_none(default_loader, x) for x in paths]
#
#     # 使用绝对路径
#     csv_path = os.path.expanduser('~/image_dimensions_origin.csv')
#
#     # 打开一个CSV文件准备写入
#     with open(csv_path, mode='w', newline='') as file:
#         writer = csv.writer(file)
#         # 写入表头
#         writer.writerow(['Image Path', 'Original Width', 'Original Height', 'Original Channels', 'Transformed Width', 'Transformed Height', 'Transformed Channels'])
#
#         # 对于每个图像，获取其尺寸和通道数并写入CSV
#         for i, image in enumerate(result):
#             original_width, original_height = image.size
#             original_channels = len(image.getbands())
#
#             # 应用变换（如果有的话�?
#             if transform is not None:
#                 transformed_image = apply_if_not_none(transform, image)
#                 transformed_width, transformed_height = transformed_image.size
#                 transformed_channels = len(transformed_image.getbands())
#             else:
#                 transformed_image = image
#                 transformed_width, transformed_height = original_width, original_height
#                 transformed_channels = original_channels
#
#             # 写入CSV
#             writer.writerow([paths[i], original_width, original_height, original_channels, transformed_width, transformed_height, transformed_channels])
#
#             # 更新结果列表
#             result[i] = transformed_image
#
#     return result

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



