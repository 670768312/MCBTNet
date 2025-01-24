import os
import logging
import sys
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt

from networks.bra_unet_for_attenmap import BRAUnet
import cv2
from synapse_train_test.utils import test_single_volume
import global_attenmap


def load_model(snapshot_path):
    net = BRAUnet(img_size=224, in_chans=3, num_classes=9, n_win=7).cuda()
    net.load_state_dict(torch.load(snapshot_path))
    net.eval()
    return net


def predict(model, img_path, save_path=None):
    image = Image.open(img_path).convert("RGB")
    img_size = 224
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image = transform(image).unsqueeze(0).cuda()

    model.eval()
    with torch.no_grad():
        prediction,r_idx,atten_weight = model(image)
    return prediction,r_idx,atten_weight

def visualize_segmentation(prediction):
    # Apply softmax to get probability distribution
    probabilities = torch.softmax(prediction, dim=1)
    # Get the class predictions by taking the argmax along the channel dimension
    # 定义颜色类别
    colors = ['black', 'red', 'green', 'purple', 'cyan', 'pink', 'yellow', 'burlywood', 'blue']

    # 根据类别预测生成的分割掩码
    class_predictions = torch.argmax(probabilities, dim=1).squeeze().cpu().numpy()

    # 叠加掩码在CT图像上
    for i in range(len(colors)):
        mask_class = (class_predictions == i)
        plt.imshow(np.ma.masked_where(mask_class == 0, mask_class), cmap=plt.cm.colors.ListedColormap([colors[i]]),
                   alpha=0.5)

    plt.show()
def generate_ImageWithPoint(background_image_path,x_coordinate, y_coordinate):
    background_image = cv2.imread(background_image_path)
    resized_image = cv2.resize(background_image, (224, 224))
    cv2.circle(resized_image, (x_coordinate, y_coordinate), radius=3, color=(0, 0, 255), thickness=-1)
    return resized_image

def generate_heatmap(background_image_path,r_idx, atten_weight,x_coordinate, y_coordinate):
    #生成背景图
    background_image = cv2.imread(background_image_path)
    resized_image = cv2.resize(background_image, (224, 224))
    cv2.circle(resized_image, (x_coordinate, y_coordinate), radius=3, color=(0, 0, 255), thickness=-1)

    # 创建一个14x14的图像，初始为白色
    image = np.zeros((14, 14))

    # 对每个patch进行处理
    for i, patch_index in enumerate(r_idx):
        # 计算当前 patch 的索引
        row_index = patch_index // 7
        col_index = patch_index % 7

        min_val = np.min(atten_weight)
        max_val = np.max(atten_weight)
        # 将张量的值归一化到 [0, 1] 范围内
        atten_weight = (atten_weight - min_val) / (max_val - min_val)

        # 获取当前 patch 块内的热力图值
        patch_heatmap_values = atten_weight[i * 4: (i + 1) * 4]

        # 将热力图值映射到颜色，并分配给当前 patch 块内的四个像素
        for j, heatmap_value in enumerate(patch_heatmap_values):
            color = heatmap_value
            # 在图像上着色
            image[row_index * 2 + j // 2, col_index * 2 + j % 2] = color

    heatmap = cv2.applyColorMap((image*255).astype(np.uint8), cv2.COLORMAP_JET)

    # 创建一个空白的 224x224 画布
    resized_heatmap = np.zeros((224, 224, 3), dtype=np.uint8)

    # 将 heatmap 像素块重复放大为 224x224
    for i in range(14):
        for j in range(14):
            resized_heatmap[i * 16:(i + 1) * 16, j * 16:(j + 1) * 16] = heatmap[i, j]
    # 将 heatmap 叠加到 background_image 上
    alpha = 0.5  # 控制 heatmap 的透明度
    result = cv2.addWeighted(resized_image, 1 - alpha, resized_heatmap, alpha, 0)
    return result
    # 显示结果
    cv2.namedWindow('Result', cv2.WINDOW_NORMAL)  # 使用可调整大小的窗口
    cv2.imshow('Result', result)
    # cv2.imshow('Result', resized_heatmap)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def generate_regionmap(background_image_path,r_idx,x_coordinate, y_coordinate):
    #生成背景图
    background_image = cv2.imread(background_image_path)
    resized_image = cv2.resize(background_image, (224, 224))
    cv2.circle(resized_image, (x_coordinate, y_coordinate), radius=3, color=(0, 0, 255), thickness=-1)

    # 创建一个14x14的图像，初始为黑色
    region_image = np.zeros((14, 14), dtype=np.uint8)

    for patch_index in r_idx:
        # 计算当前 patch 的索引
        row_index = patch_index // 7
        col_index = patch_index % 7

        for j in range(4):
            color = 255
            # 在图像上着色
            region_image[row_index * 2 + j // 2, col_index * 2 + j % 2] = color

    resized_regionmap = np.zeros((224, 224,3), dtype=np.uint8)

    # 将 region 像素块重复放大为 224x224
    for i in range(14):
        for j in range(14):
            resized_regionmap[i * 16:(i + 1) * 16, j * 16:(j + 1) * 16] = region_image[i,j]
    # 将 regionmap 叠加到 background_image 上
    alpha = 0.5  # 控制 heatmap 的透明度
    result = cv2.addWeighted(resized_image, 1 - alpha, resized_regionmap, alpha, 0)
    return result
    # 显示结果
    cv2.namedWindow('Result', cv2.WINDOW_NORMAL)  # 使用可调整大小的窗口
    cv2.imshow('Result', result)
    # cv2.imshow('Result', resized_heatmap)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def get_map(r_idx,atten_weight,patch_number,pixel_index,num_head=0):
    R_idx=r_idx[11][0,patch_number,:]
    Atten_weight=atten_weight[11][patch_number,num_head,pixel_index,:]
    # 将 PyTorch 张量从 CUDA 设备移到 CPU 上
    R_idx = R_idx.cpu().numpy()
    Atten_weight = Atten_weight.cpu().numpy()
    return R_idx, Atten_weight


def SelectPixel(image_path):
    def preprocess_image(image):
        resized_image = cv2.resize(image, (224, 224))
        return resized_image

    # 图像压缩为14x14
    def compress_image(image):
        compressed_image = cv2.resize(image, (14, 14))
        return compressed_image

    # 显示图像并获取鼠标点击位置
    def get_mouse_click(event, x, y, flags, param):
        global clicked_point
        if event == cv2.EVENT_LBUTTONDOWN:
            clicked_point = (x, y)

    def get_patch_number_and_pixel_index(x, y):
        # 计算x坐标所在的列数和在patch中的像素列索引
        patch_col = x // 2
        pixel_col_index = x % 2

        # 计算y坐标所在的行数和在patch中的像素行索引
        patch_row = y // 2
        pixel_row_index = y % 2

        # 计算patch序号
        patch_index = patch_row * 7 + patch_col

        # 计算像素在patch内的索引（从左到右，从上到下）
        pixel_index_in_patch = pixel_row_index * 2 + pixel_col_index

        return patch_index, pixel_index_in_patch

    image = cv2.imread(image_path)

    # 处理图像大小为224x224
    resized_image = preprocess_image(image)

    # 显示原始图像并让用户选择像素点
    plt.imshow(cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB))
    plt.title('Select a pixel on the original image')
    point = plt.ginput(1)  # 用户点击一个像素点
    plt.close()

    # 获取用户点击的像素点在原始图像中的位置
    clicked_x, clicked_y = int(point[0][0]), int(point[0][1])

    # 压缩图像为14x14
    compressed_image = compress_image(resized_image)

    # 计算点击位置在压缩图像中的坐标
    compressed_x = int(clicked_x / 224 * 14)
    compressed_y = int(clicked_y / 224 * 14)

    patch_number, pixel_index = get_patch_number_and_pixel_index(compressed_x, compressed_y)

    print("Coordinate (x, y):", (compressed_x, compressed_y))
    print("Patch Number:", patch_number)
    print("Pixel Index in Patch:", pixel_index)
    return patch_number,pixel_index,clicked_x, clicked_y


if __name__ == "__main__":
    snapshot_path = r"D:\python_code\BRAU-Netplusplus\synapse_train_test\save_models\Synapse_epoch_359.pth"
    model = load_model(snapshot_path)
    img_path = r"D:\python_code\BRAU-Netplusplus\synapse_train_test\save_models\predictions_sy9\png\case0022_slice58_img.png"
    prediction,r_idx,atten_weight = predict(model, img_path)
    patch_index,pixel_index,clicked_x, clicked_y = SelectPixel(img_path)
    R_idx, Atten_weight = get_map(r_idx,atten_weight,patch_index,pixel_index,num_head=3)
    ImageWithPoint = generate_ImageWithPoint(img_path,clicked_x,clicked_y)
    Regionmap = generate_regionmap(img_path, R_idx, clicked_x, clicked_y)
    Heatmap = generate_heatmap(img_path,R_idx, Atten_weight,clicked_x, clicked_y)

    output_directory = r"D:\python_code\BRAU-Netplusplus\Attention_map\Synapse"
    cv2.imwrite(os.path.join(output_directory, "ImageWithPoint.png"), ImageWithPoint)
    cv2.imwrite(os.path.join(output_directory, "Regionmap.png"), Regionmap)
    cv2.imwrite(os.path.join(output_directory, "Heatmap.png"), Heatmap)

    # 水平合并三张图片
    merged_image = cv2.hconcat([ImageWithPoint, Regionmap, Heatmap])

    # 显示合并后的结果
    cv2.namedWindow('Merged Result', cv2.WINDOW_NORMAL)  # 使用可调整大小的窗口
    cv2.imshow('Merged Result', merged_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    print(f"IdxSize:",r_idx[11].shape,f"AttenWeightSize",atten_weight[11].shape)
    visualize_segmentation(prediction)
