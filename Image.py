#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ------------------------------------------------------------------------------
# Purpose:
# Author:      lianjiaxin
# History:
#
# Created:     2017-02-13
# Copyright:   (c) Hikvision.com 2017
# ------------------------------------------------------------------------------
'''
图像测试库主要用于图像属性分析和多图像效果对比。
目前已实现亮度，饱和度，锐度，灰度，对比度的测评；
另外支持色偏检测、图像过曝/过暗检测、噪点检测、镜像分析、走廊模式分析、图片相似性分析、图片模糊虚焦识别、黑白或彩色图片识别。
'''
import os
import sys
import math
import logging
# import time
import gc
import numpy as np
from numpy.matlib import repmat
import cv2
from PIL import Image
from PIL import ImageFile
from PIL import ImageChops
import ssim
import imagehash
ImageFile.LOAD_TRUNCATED_IMAGES = True
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    stream=sys.stdout)


rosolution_abbreviation_mapping = {
    "DCIF": "528*384/528*320",
    "CIF": "352*288/352*240",
    "QCIF": "176*144/176*120",
    "4CIF": "704*576/704*480",
    "2CIF": "704*288/704*240",
    "QVGA": "320*240",
    "QQVGA": "160*120",
    "VGA": "640*480",
    "UXGA": "1600*1200",
    "SVGA": "800*600",
    "HD720P": "1280*720",
    "XVGA": "1280*960",
    "HD900P": "1600*900",
    "XGA": "1024*768",
    "SXGA": "1280*1024",
    "WD1": "960*576/960*480",
    "1080i": "1920*1080",
    "1080I": "1920*1080",
    "HD1080P": "1920*1080",
    "HD_F": "1920*1080/1280*720",
    "HD_H": "1920*540/1280*360",
    "HD_Q": "960*540/630*360",
}

MAX_PIXELS_FOR_NUMPY = 1000000  # Numpy运算时，图像像素不宜超过100W像素


def cv2_img_read(filepath):
    """
    读取图像，解决imread不能读取中文路径的问题
    参数：图片路径
    返回值：图像数据数组
    """
    img_array = cv2.imdecode(np.fromfile(filepath, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    return img_array


class ImageAnalysis(object):
    '''
    图像测试库主要用于图像属性分析和多图像效果对比。
    目前已实现亮度，饱和度，锐度，灰度，对比度的测评；
    另外支持色偏检测、图像过曝/过暗检测、噪点检测、镜像分析、走廊模式分析、图片相似性分析、图片模糊虚焦识别、黑白或彩色图片识别。
    详细实例请访问:https://10.1.14.79/testdev/HITA/branches/testcases/CaseExample/08_图像测试库调用实例
    '''
    ROBOT_LIBRARY_SCOPE = 'TEST SUITE'
    version = '1.1.0'

    def __init__(self):
        Image.init()
        self.file_extension = ['.jpg', '.jpeg', '.bmp', '.png']

    def _check_file(self, filepath):
        '''
        判断文件是否存在，及文件类型
        '''
        if not os.path.exists(filepath):
            raise Exception(
                'OSError: [Errno 2] No such file or directory: %s' % filepath)
        filename, file_extension = os.path.splitext(filepath)
        if file_extension.lower() not in self.file_extension:
            raise Exception(
                "file extension is not available. The available file extensions include: %s" % ', '.join(self.file_extension))

    def get_available_file_extension(self):
        '''
        获取有效的图像扩展名
        返回值：
            有效的图像扩展名列表
        作者：
            lianjiaxin
        '''
        return self.file_extension

    def get_picture_size(self, filepath):
        '''
        获取图片的像素大小。
        作者：
            sumuzhong
        '''
        self._check_file(filepath)
        picture = Image.open(filepath)
        return picture.size

    def get_picture_resolution(self, filepath):
        '''
        获取图片的分辨率。
        作者：
            sumuzhong
        '''
        self._check_file(filepath)
        picture = Image.open(filepath)
        (width, height) = picture.size
        if width > height:
            return str(width) + "*" + str(height)
        else:
            return str(height) + "*" + str(width)

    def estimate_image_luminance(self, filepath):
        '''
        评估图片亮度
        参数:
            filepath:图片路径
        返回值:
            返回图像亮度评估数值，数值越大，亮度越高；范围：[0, 100]
        作者：
            lianjiaxin
        '''
        self._check_file(filepath)
        img_array = cv2_img_read(filepath)
        hsv = cv2.cvtColor(img_array, cv2.COLOR_BGR2HSV)
        hue, saturation, value = cv2.split(hsv)
        luminance = float(value.sum()) / (hsv.shape[0] * hsv.shape[1])
        return round(luminance, 2)

    def estimate_image_saturation(self, filepath):
        '''
        评估图片饱和度
        参数:
            filepath:图片路径
        返回值:
            返回图像饱和度评估数值，数值越大，饱和度越高；范围：[0, 255]
        作者：
            lianjiaxin
        '''
        self._check_file(filepath)
        img_array = cv2_img_read(filepath)
        hsv = cv2.cvtColor(img_array, cv2.COLOR_BGR2HSV)
        hue, saturation, value = cv2.split(hsv)
        saturation_value = float(saturation.sum()) / \
            (hsv.shape[0] * hsv.shape[1])
        return round(saturation_value, 2)

    def estimate_image_gray(self, filepath):
        '''
        评估图片灰度
        参数:
            filepath:图片路径
        返回值:
            返回图像灰度评估数值，数值越大，灰度越高；范围：[0, 255]
        作者：
            lianjiaxin
        '''
        self._check_file(filepath)
        img_array = cv2_img_read(filepath)
        gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
        logging.info(gray.shape)
        gray_value = float(gray.sum()) / (gray.shape[0] * gray.shape[1])
        return round(gray_value, 2)

    def estimate_image_contrast(self, filepath):
        '''
        评估图片对比度
        参数:
            filepath:图片路径
        返回值:
            返回图像对比度评估数值，数值越大，对比度越高。
        作者：
            lianjiaxin
        '''
        # 【算法】
        # I=im2double(imread('E1.jpg'));   %首先把图像数据类型转换为双精度浮点类型
        # I=rgb2gray(I);%将图像转换成二维
        # [Nx,Ny] = size(I);
        # Ng=256;
        # G=double(I);
        # %计算对比度
        # [counts,graylevels]=imhist(I);
        # PI=counts/(Nx*Ny);
        # averagevalue=sum(graylevels.*PI);
        # u4=sum((graylevels-repmat(averagevalue,[256,1])).^4.*PI);
        # standarddeviation=sum((graylevels-repmat(averagevalue,[256,1])).^2.*PI);
        # alpha4=u4/standarddeviation^2;
        # Fcontrast=sqrt(standarddeviation)/alpha4.^(1/4);
        # disp('对比度：');display(Fcontrast)

        self._check_file(filepath)
        img_array = cv2_img_read(filepath)
        bins = np.arange(257)
        gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
        hist, bins = np.histogram(gray, bins)
        bins = np.arange(256)
        pi = hist * 1.0 / (img_array.shape[0] * img_array.shape[1])
        aver_b = sum(bins * pi)
        ufouth_array = np.power(
            bins * 1.0 - repmat(aver_b, 1, 256), 4) * pi
        ufouth = ufouth_array.sum()
        std_array = np.power(
            bins * 1.0 - repmat(aver_b, 1, 256), 2) * pi
        std = std_array.sum()
        alpha_fouth = ufouth / (std * std)
        contrast = np.sqrt(std) / math.sqrt(math.sqrt(alpha_fouth))
        return int(contrast)

    def estimate_image_sharpness(self, filepath):
        '''
        评估图片锐度
        参数:
            filepath:图片路径
        返回值：
            返回锐度评估值，数值越大，锐度越高
        作者：
            lianjiaxin
        '''
        self._check_file(filepath)
        img_array = cv2_img_read(filepath)
        height, width = img_array.shape[0], img_array.shape[1]

        # 判断图像是否超过numpy支持最大像素
        src_times = self._times_of_max_numpy_support_pixels(img_array)
        if src_times > 1:
            new_width = width / (int(math.sqrt(src_times)) + 1)
            new_height = height / (int(math.sqrt(src_times)) + 1)
            resized_src = cv2.resize(
                img_array, (new_width, new_height), cv2.INTER_NEAREST)
            img_array = resized_src
        edges = cv2.Laplacian(img_array, cv2.CV_64F)
        std = round(edges.std(), 2)
        del img_array
        gc.collect()
        return std

        # estimate_image_sharpness(self, filepath, minVal=50, maxVal=150, aperture_size=3):
        # edges = cv2.Canny(img, int(minVal), int(maxVal), int(aperture_size))
        # edges_count = edges.sum() / 255  # canny算法处理后，黑色为0,边缘点即白色为255
        # return edges_count
        # edge_percent = float(edges_count) / edges.size * 100
        # return round(edge_percent, 2)

    def is_image_colorful(self, filepath, rgb_range=15):
        '''
        判断图片是否为彩色
        参数:
            filepath:图片路径
            rgb_range:像素R,G,B通道之间的绝对值误差范围，rgb_range越小，黑白图片识别率越低；rgb_range越大，彩色图片识别率越低
        返回值:
            若图片为彩色，则返回True；否则返回False
        作者：
            lianjiaxin
        '''
        # 原理：彩色像素RGB三个通道的强度值差异很大
        if not str(rgb_range).isdigit():
            raise Exception("rgb_range value should be an integer")
        rgb_range = int(rgb_range)
        self._check_file(filepath)
        img_array = cv2_img_read(filepath)
        if img_array.std() == 0:  # 全白或者全黑图像
            avg_l = img_array.sum() / img_array.size
            return avg_l >= 128

        blue, green, red = cv2.split(img_array)
        detal_rg = red.astype(int) - green.astype(int)
        detal_rb = red.astype(int) - blue.astype(int)
        detal_gb = green.astype(int) - blue.astype(int)
        # 选择绝对值最大的数
        max_rg = detal_rg.max() if detal_rg.max() > abs(
            detal_rg.min()) else abs(detal_rg.min())
        max_rb = detal_rb.max() if detal_rb.max() > abs(
            detal_rb.min()) else abs(detal_rb.min())
        max_gb = detal_gb.max() if detal_gb.max() > abs(
            detal_gb.min()) else abs(detal_gb.min())

        return not (max_rg < rgb_range and max_rb < rgb_range and max_gb < rgb_range)

    def detect_image_blur(self, filepath, threshold=100):
        '''
        检测图像是否模糊（虚焦）
        PS：【景深过深的场景判定为会判断为模糊吗？待确认；若判断有误，那么将图像拆分成M*N块进行分析，是否可以解决？待验证】
        参数：
            filepath:图片路径
            threshold:图像模糊、虚焦判断的阈值，默认值100。分析结果高于threshold则认定图像清晰，否则认定图像模糊、虚焦.
        返回值：
            图像清晰返回True，否则返回False
        作者：
            lianjiaxin
        '''
        self._check_file(filepath)
        img_array = cv2_img_read(filepath)
        height, width = img_array.shape[0], img_array.shape[1]

        # 判断图像是否超过numpy支持最大像素
        src_times = self._times_of_max_numpy_support_pixels(img_array)
        if src_times > 1:
            new_width = width / (int(math.sqrt(src_times)) + 1)
            new_height = height / (int(math.sqrt(src_times)) + 1)
            resized_src = cv2.resize(
                img_array, (new_width, new_height), cv2.INTER_NEAREST)
            img_array = resized_src
        variance = cv2.Laplacian(img_array, cv2.CV_64F).var()
        logging.info("Laplacian variance: {}".format(variance))
        del img_array
        gc.collect()
        return variance >= float(threshold)

    # **********************图像哈希算法计算两张图片的相似性****************************

    def calculate_similar_images_by_imghash(self, srcfile, dstfile, **offset):
        '''
        通过图像哈希算法计算两张图片的相似性【图片相似性计算推荐使用该方法】
        参数：
            srcfile:对比图片1路径
            dstfile:对比图片2路径
            **offset: srcfile的偏移量（可变长参数），非必填项。
                表示图片2与图片1（左上角为坐标原点）偏移(x, y)后的区域图像进行相似度比较
                x: 水平宽度偏移量，默认为0(坐标中心(0,0)为图片1的左上角)
                y: 垂直高度偏移量，默认为0(坐标中心(0,0)为图片1的左上角)
        返回值：
            两张图片哈希字符串不相同的数据位个数。如果不相同的数据位不超过5，说明图像很相似；
            如果介于5~10，说明两张图像比较相似；通常认为距离>10 就是两张完全不同的图片
        作者：
            lianjiaxin
        '''
        offset_x = int(offset.get("x", 0))
        offset_y = int(offset.get("y", 0))
        self._check_file(srcfile)
        self._check_file(dstfile)
        src_img = cv2_img_read(srcfile)
        dst_img = cv2_img_read(dstfile)
        height, width = src_img.shape[0], src_img.shape[1]
        if offset_x > 0 or offset_y > 0:
            src_img = src_img[offset_x:, offset_y:]

        # 判断图像是否超过numpy支持最大像素
        src_times = self._times_of_max_numpy_support_pixels(src_img)
        dst_times = self._times_of_max_numpy_support_pixels(dst_img)
        if src_times > 1:
            new_width = width / (int(math.sqrt(src_times)) + 1)
            new_height = height / (int(math.sqrt(src_times)) + 1)
            resized_src = cv2.resize(
                src_img, (new_width, new_height), cv2.INTER_NEAREST)
            srcimg_array = Image.fromarray(resized_src)
        else:
            srcimg_array = Image.fromarray(src_img)
        if dst_times > 1:
            new_width = width / (int(math.sqrt(src_times)) + 1)
            new_height = height / (int(math.sqrt(src_times)) + 1)
            resized_dst = cv2.resize(
                dst_img, (new_width, new_height), cv2.INTER_NEAREST)
            dstimg_array = Image.fromarray(resized_dst)
        else:
            dstimg_array = Image.fromarray(dst_img)

        src_hash = imagehash.phash(srcimg_array)
        dst_hash = imagehash.phash(dstimg_array)
        del srcimg_array, dstimg_array, src_img, dst_img
        gc.collect()
        return src_hash - dst_hash

    # **********************图像颜色直方图算法计算两张图片的相似性****************************

    def __make_regalur_image(self, img, size=(256, 256)):
        '''
        把图片都统一到特别的规格(分辨率)
        参数：
            img:图像数组
            size:重新设置图像的规格大小
        '''
        return img.resize(size).convert('RGB')

    def __hist_similar(self, src_hist, dst_hist):
        '''
        计算直方图相似性
        参数：
            src_hist, dst_hist为直方图数据
        返回值：
            直方图相似度
        '''
        return sum(1 - (0 if l == r else float(abs(l - r)) / max(l, r)) for l, r in zip(src_hist, dst_hist)) / len(src_hist)

    def _calc_similar(self, src_img, dst_img):
        '''
        计算图片相似性
        参数：
            src_img:图片1的数组
            dst_img:图片2的数组
        返回值：
            两张图片的相似度
        '''
        if src_img.size[0] > 64 and src_img.size[1] > 64:
            part_size = (64, 64)
        else:
            part_size = (src_img.size[0], src_img.size[1])
        # img_divide_num = (
        #     src_img.size[0] * src_img.size[1]) / (part_size[0] * part_size[1])  # 计算图像划分个数
        # 计算图像划分个数  向上取整
        img_divide_num = math.ceil(float(
            src_img.size[0]) / part_size[0]) * math.ceil(float(src_img.size[1]) / part_size[1])
        img_divide_num = int(img_divide_num)
        return sum(
            self.__hist_similar(
                l.histogram(), r.histogram()) for l, r in zip(
                    self.__split_image(
                        src_img, part_size), self.__split_image(dst_img, part_size))) / img_divide_num  # 16.0

    def __split_image(self, img, part_size=(64, 64)):
        '''
        分割图片
        参数：
            img:图像数组
            part_size:每个子图片的规格大小
        返回值：
            分割后的图片列表
        '''
        width, height = img.size
        pwidth, pheight = part_size
        tem_img_list = []

        for i in range(0, width, pwidth):
            for j in range(0, height, pheight):
                temp_img = img.crop((i, j, i + pwidth, j + pheight)).copy()
                tem_img_list.append(temp_img)
        return tem_img_list

    def calculate_similar_images_by_histogram(self, srcfile, dstfile, **offset):
        '''
        通过图像颜色直方图算法计算两张图片的相似性
        参数：
            srcfile:对比图片1路径
            dstfile:对比图片2路径
            **offset: srcfile的偏移量（可变长参数），非必填项。
                表示图片2与图片1（左上角为坐标原点）偏移(x, y)后的区域图像进行相似度比较
                x: 水平宽度偏移量，默认为0(坐标中心(0,0)为图片1的左上角)
                y: 垂直高度偏移量，默认为0(坐标中心(0,0)为图片1的左上角)
        返回值：
            返回两张图片相似度（百分比）
        作者：
            lianjiaxin
        '''
        offset_x = int(offset.get("x", 0))
        offset_y = int(offset.get("y", 0))
        self._check_file(srcfile)
        self._check_file(dstfile)
        src_img = cv2_img_read(srcfile)
        dst_img = cv2_img_read(dstfile)
        if offset_x > 0 or offset_y > 0:
            src_img = src_img[offset_x:, offset_y:]
        srcimg_array = Image.fromarray(src_img)
        dstimg_array = Image.fromarray(dst_img)

        resize_value = srcimg_array.size if (
            srcimg_array.size[0] * srcimg_array.size[1]) < (
                dstimg_array.size[0] * dstimg_array.size[1]) else dstimg_array.size
        src_img = self.__make_regalur_image(srcimg_array, resize_value)
        dst_img = self.__make_regalur_image(dstimg_array, resize_value)
        return round(self._calc_similar(src_img, dst_img) * 100, 2)

    # **********************SSIM结构相似性算法计算两张图片的相似性****************************

    def calculate_similar_images_by_ssim(self, srcfile, dstfile, **offset):
        '''
        通过SSIM结构相似性算法计算两张图片的相似性【图片相似性计算推荐使用该方法】
        参数：
            srcfile:对比图片1路径
            dstfile:对比图片2路径
            **offset: srcfile偏移量（可变长参数），非必填项。
                表示图片2与图片1的（左上角为坐标原点）偏移(x, y)后的区域图像进行相似度比较
                x: 水平宽度偏移量，默认为0(坐标中心(0,0)为图片1的左上角)
                y: 垂直高度偏移量，默认为0(坐标中心(0,0)为图片1的左上角)
        返回值：
            返回两张图片相似度（百分比）
        作者：
            lianjiaxin
        '''
        offset_x = int(offset.get("x", 0))
        offset_y = int(offset.get("y", 0))
        self._check_file(srcfile)
        self._check_file(dstfile)
        src_img = cv2_img_read(srcfile)
        dst_img = cv2_img_read(dstfile)
        height, width = src_img.shape[0], src_img.shape[1]
        if offset_x > 0 or offset_y > 0:
            src_img = src_img[offset_x:, offset_y:]

        # 判断图像是否超过numpy支持最大像素
        src_times = self._times_of_max_numpy_support_pixels(src_img)
        dst_times = self._times_of_max_numpy_support_pixels(dst_img)
        if src_times > 1:
            new_width = width / (int(math.sqrt(src_times)) + 1)
            new_height = height / (int(math.sqrt(src_times)) + 1)
            resized_src = cv2.resize(
                src_img, (new_width, new_height), cv2.INTER_NEAREST)
            srcimg_array = Image.fromarray(resized_src)
        else:
            srcimg_array = Image.fromarray(src_img)
        if dst_times > 1:
            new_width = width / (int(math.sqrt(src_times)) + 1)
            new_height = height / (int(math.sqrt(src_times)) + 1)
            resized_dst = cv2.resize(
                dst_img, (new_width, new_height), cv2.INTER_NEAREST)
            dstimg_array = Image.fromarray(resized_dst)
        else:
            dstimg_array = Image.fromarray(dst_img)
        del src_img, dst_img
        gc.collect()
        return round(ssim.compute_ssim(srcimg_array, dstimg_array) * 100, 2)

    def detect_image_corridor(self, srcfile, dstfile, similarity=80):
        '''
        检测图像走廊模式开启后的效果，将源图像srcfile进行走廊模式处理，再与目的图像dstfile进行相似性比较。
        若相似度大于similarity，则判定走廊模式生效；否则走廊模式不生效
        参数：
            srcfile:未开启走廊模式的图片路径
            dstfile:开启走廊模式的图片路径
            similarity:图片相似度阈值，若源图像处理后，相似度大于similarity，则判定走廊模式生效；否则走廊模式不生效
        返回值：
            分析结果，走廊模式生效返回True；否则返回False
        作者：
            lianjiaxin
        '''
        if not str(similarity).isdigit():
            raise Exception("similarity value should be an integer")
        self._check_file(srcfile)
        self._check_file(dstfile)
        srcimg_array = Image.open(srcfile)
        dstimg_array = Image.open(dstfile)
        similarity = int(similarity)
        temp = srcimg_array.transpose(Image.ROTATE_270).rotate(
            180)  # .transpose(Image.FLIP_LEFT_RIGHT)
        similar_rate1 = round(self._calc_similar(temp, dstimg_array) * 100, 2)
        similar_rate2 = ssim.compute_ssim(temp, dstimg_array) * 100
        similar_rate = (similar_rate1 + similar_rate2) / 2
        logging.info('ssim similarity= {}'.format(round(similar_rate, 2)))
        return similar_rate > similarity

    def detect_image_mirror(self, srcfile, dstfile, mirror='left-right', similarity=80):
        '''
        检测图像镜像模式开启后的效果，将源图像srcfile进行镜像处理，再与目的图像dstfile进行相似性比较。
        若相似度大于similarity，则判定走廊模式生效；否则走廊模式不生效
        参数：
            srcfile:未开启镜像模式的图片路径
            dstfile:开启镜像模式的图片路径
            mirror:镜像模式，支持left-right（左右镜像）,center（中心镜像）,top-bottom（中心镜像）
            similarity:图片相似度阈值，若源图像处理后，相似度大于similarity，则判定镜像模式生效；否则镜像模式不生效
        返回值：
            分析结果，镜像模式生效返回True；否则返回False
        作者：
            lianjiaxin
        '''
        if not str(similarity).isdigit():
            raise Exception("similarity value should be an integer")
        self._check_file(srcfile)
        self._check_file(dstfile)
        src_img = cv2_img_read(srcfile)
        dst_img = cv2_img_read(dstfile)
        height, width = src_img.shape[0], src_img.shape[1]

        # 判断图像是否超过numpy支持最大像素
        src_times = self._times_of_max_numpy_support_pixels(src_img)
        dst_times = self._times_of_max_numpy_support_pixels(dst_img)
        if src_times > 1:
            new_width = width / (int(math.sqrt(src_times)) + 1)
            new_height = height / (int(math.sqrt(src_times)) + 1)
            resized_src = cv2.resize(
                src_img, (new_width, new_height), cv2.INTER_NEAREST)
            srcimg_array = Image.fromarray(resized_src)
        else:
            srcimg_array = Image.fromarray(src_img)
        if dst_times > 1:
            new_width = width / (int(math.sqrt(src_times)) + 1)
            new_height = height / (int(math.sqrt(src_times)) + 1)
            resized_dst = cv2.resize(
                dst_img, (new_width, new_height), cv2.INTER_NEAREST)
            dstimg_array = Image.fromarray(resized_dst)
        else:
            dstimg_array = Image.fromarray(dst_img)

        if mirror.lower() == 'left-right':
            mirror_value = Image.FLIP_LEFT_RIGHT
        elif mirror.lower() == 'center':
            mirror_value = Image.ROTATE_180
        elif mirror.lower() == 'top-bottom':
            mirror_value = Image.FLIP_TOP_BOTTOM
        else:
            raise Exception("mirror ValueError, not support value:%s" % mirror)
        temp = srcimg_array.transpose(mirror_value)
        similar_rate1 = round(self._calc_similar(temp, dstimg_array) * 100, 2)
        similar_rate2 = ssim.compute_ssim(temp, dstimg_array) * 100
        similar_rate = (similar_rate1 + similar_rate2) / 2
        similarity = int(similarity)
        logging.info('ssim similarity= {}'.format(round(similar_rate, 2)))
        del dstimg_array, temp
        gc.collect()
        return similar_rate > similarity

    def slice_image(self, filepath, dst_dir=None, horizontal=4, vertical=4):
        '''
        将图片分割成horizontal*vertical张图片，并保存到dst_dir目录
        参数：
            filepath:待分割图片路径
            dst_dir:保存图片目录
            horizontal:水平方向上切割个数
            vertical:垂直方向上切割个数
            备注：源图像命名，如"Test.jpg", 那么分割后的图像命名为"Test_1.jpg", "Test_2.jpg", "Test_3.jpg", ...
        返回值：
            图像分割后的各个子图的路径列表
        '''
        self._check_file(filepath)
        img = cv2_img_read(filepath)
        filepath_dir = os.path.dirname(filepath)
        basename = os.path.basename(filepath)
        basename_only, extensions = os.path.splitext(basename)
        if not dst_dir or dst_dir == 'None':
            dst_dir = filepath_dir
        else:
            if not os.path.exists(dst_dir):
                os.makedirs(dst_dir)
        array_list = get_all_slice_from_array(
            img, int(horizontal), int(vertical))
        sub_image_list = []
        for index in range(1, len(array_list) + 1):
            filename = basename_only + '_%d' % index + extensions
            temp_path = os.path.join(dst_dir, filename)
            logging.info(temp_path)
            cv2.imwrite(temp_path, array_list[index - 1])
            sub_image_list.append(temp_path)
        return sub_image_list

    def estimate_noise(self, filepath):
        '''
        检测图像噪点。
        参数：
            filepath:图片路径
        返回值：
            返回噪点指数，指数越大，图像噪点越大
        作者：
            lianjiaxin
        '''
        import scipy
        self._check_file(filepath)
        img = cv2_img_read(filepath)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        height, width = gray.shape

        meangaussian = [[1, -2, 1],  # 0均值高斯滤波
                        [-2, 4, -2],
                        [1, -2, 1]]
        sigma = np.sum(np.absolute(
            scipy.signal.convolve2d(gray, meangaussian)))
        sigma = sigma * math.sqrt(0.5 * math.pi) / \
            (6 * (width - 2) * (height - 2))
        return sigma

    def detect_image_color_cast(self, filepath):
        '''
        检测图像色偏（仅适用于彩色图片）
        参数：
            filepath:图片路径
        返回值：
            返回色偏指数字典：
            kernel：色偏指数，综合来说，kernel值小于1.5，可认为图像是正常的，否则，可认为存在色偏（参考值可能需要和实际情况结合进行调整）
            Da:红/绿色偏估计值，da大于0，表示偏红；da小于0表示偏绿
            Db:黄/蓝色偏估计值，db大于0，表示偏黄；db小于0表示偏蓝
        作者：
            lianjiaxin
        '''
        # RGB颜色空间是最简单的一种颜色空间，但是RGB颜色空间最大的局限性在于当用欧氏距离来刻画两种颜色之间的差异时，
        # 所计算出的两种颜色之间的距无法正确表征人们实际所感知到的这两种颜色之间的真实差异。采用CIE Lab颜色空间，
        # 此空间所计算出来的颜色之间的距离与实际感知上的差别基本一致。其直方图可以客观的反映图像色偏程度，
        # 在CIE Lab下进行偏色图像的自动检测更为合理。
        # 将RGB图像转变到CIE L*a*b*空间，其中L*表示图像亮度，a*表示图像红/绿分量，b*表示图像黄/蓝分量。
        # 通常存在色偏的图像，在a*和b*分量上的均值会偏离原点很远，方差也会偏小；
        # 通过计算图像在a*和b*分量上的均值和方差，就可评估图像是否存在色偏。
        self._check_file(filepath)
        img = cv2_img_read(filepath)
        colorful = self.is_image_colorful(filepath, 10)
        if not colorful:
            raise Exception(
                "%s: The picture seems to be a gray picture but a colorful picture is excepted." % filepath)
        lab_img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        area = lab_img.shape[0] * lab_img.shape[1]
        axis_l, axis_a, axis_b = cv2.split(lab_img)  # 取通道0~2中的1和2通道
        area = axis_a.size
        # * da          红/绿色偏估计值，da大于0，表示偏红；da小于0表示偏绿
        # * db          黄/蓝色偏估计值，db大于0，表示偏黄；db小于0表示偏蓝
        Da = float(axis_a.sum()) / area - 128  # 必须归一化到[-128，,127]范围内
        Db = float(axis_b.sum()) / area - 128

        hist_a, bins_a = np.histogram(axis_a, range(256))
        hist_b, bins_b = np.histogram(axis_b, range(256))
        # 平均色度
        avg_huv = math.sqrt(Da * Da + Db * Db)
        measure_a = 0.0
        measure_b = 0.0
        for i in range(0, len(hist_a)):
            measure_a += abs(i - 128 - Da) * hist_a[i]  # //计算范围-128～127
            measure_b += abs(i - 128 - Db) * hist_b[i]
        measure_a /= area
        measure_b /= area
        center_distance = math.sqrt(
            measure_a * measure_a + measure_b * measure_b)
        if center_distance > 0:
            kernel = avg_huv / center_distance
        else:
            kernel = 100
        color_cast_dict = {}
        color_cast_dict['kernel'] = kernel
        color_cast_dict['Da'] = Da
        color_cast_dict['Db'] = Db

        return color_cast_dict

    def detect_image_luminance(self, filepath):
        '''
        检测图像亮度情况。
        filepath:图片路径
        返回值：
            返回图像亮度检测结果：（以下为经验值）
                1)结果在[-1, 1]之间，亮度正常
                2)结果在[1, 2]之间，亮度较高
                3)结果大于2，亮度过曝
                4)结果在[-1, -1.5]，亮度较低
                5)结果小于-1.5，亮度太暗
        作者：
            lianjiaxin
        '''
        self._check_file(filepath)
        img = cv2_img_read(filepath)
        gray_array = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        area = gray_array.shape[0] * gray_array.shape[1]
        hist, bins = np.histogram(gray_array, range(256))
        dlum = float(gray_array.sum()) / area - 128
        measure_a = 0.0
        for i in range(len(hist)):
            measure_a += abs(i - 128 - dlum) * hist[i]
        measure_a /= area
        cast = dlum / measure_a
        return cast

    def get_ridecase_examples(self):
        '''
        图像分析库脚本示例：
        | #图片相似度 | | | |
        | ${similar_rate}= | Calculate Similar Images By Ssim | F:\\1.jpg | F:\\2.jpg |
        | #虚假检测 | | | |
        | ${is_clear}= | Detect Image Blur | F:\\1.jpg | 150 |
        | Should Be True | ${is_clear} | | |
        | #噪声对比 | | | |
        | ${noise1}= | Estimate Noise | F:\\1.jpg | |
        | ${noise2}= | Estimate Noise | F:\\2.jpg | |
        | Should Be True | ${noise1}>${noise2} | | |
        更多示例请访问：https://10.1.14.79/testdev/HITA/branches/testcases/CaseExample/08_图像测试库调用实例
        '''
        return None

    def resoluion_match(self, abbreviation_rosolution, picture_resoluion):
        '''
        检查分辨率(简写的) 和图片的大小是否匹配。
        picture_resoluion 可以通过 关键字 Get Picture Resolution 获得。
        返回值：
            abbreviation_rosolution等于picture_resoluion，则返回True
            abbreviation_rosolution不等于picture_resoluion，则返回False
        作者：
            sumuzhong
        '''
        if "*" in abbreviation_rosolution:
            if abbreviation_rosolution == picture_resoluion:
                return True
            else:
                return False
        else:
            rosolution = rosolution_abbreviation_mapping[
                abbreviation_rosolution]
            if picture_resoluion in rosolution:
                return True
            else:
                return False

    def _deal_with_2_images(self, img1, img2):
        '''
        处理两幅图，得到灰度图
        '''
        if img1.size != img2.size:
            logging.error("Error: images size differ")
            raise SystemExit
        img3 = ImageChops.invert(img1)
        img = Image.blend(img2, img3, 0.5)
        img = img.convert('L')

        return img

    def _get_binary_image(self, img, path):
        '''
        图像二值化
        '''
        table = []
        for i in range(256):
            if i > 100:
                table.append(0)
            else:
                table.append(1)

        binary_img = img.point(table, "1")
        binary_img.save(path)
        return binary_img

    def _expend(self, img, expend=(3, 3)):
        '''
        膨胀处理
        '''
        kernel = cv2.getStructuringElement(
            cv2.MORPH_CROSS, expend)  # 结构 (3, 3), (7, 7)
        for i in range(8):
            img = cv2.dilate(img, kernel)
        return img

    def _corrosion(self, expend_img):
        '''
        腐蚀操作--去散点
        '''
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))  # 腐蚀操作--去散点
        corrosion_img = cv2.erode(expend_img, kernel)
        return corrosion_img

    def _get_region(self, img1, corrosion_img, detectpix=20):
        '''
        获取图片不同点的区域
        参数：
            img1：model图片，即从该图片进行截取
            corrosion_img：进过腐蚀去散处理的图片
            detectpix：检测的最小区域，单位像素，默认20
        返回：返回区域编码的矩形box
        '''
        img1_a = cv2_img_read(img1)
        img2_closed = cv2_img_read(corrosion_img)
        gray = cv2.cvtColor(img2_closed, cv2.COLOR_BGR2GRAY)
        ret, binary = cv2.threshold(
            gray, 0.5, 1, cv2.THRESH_BINARY)  # 阈值（0.5，1）
        find_image, contours, hierarchy = cv2.findContours(
            binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)  # contours 边缘， hierarchy
        cv2.drawContours(img1_a, contours, -1, (0, 0, 255), 3)
        ContourArea = []
        for cnt in contours:
            Area = cv2.contourArea(cnt)
            ContourArea.append(Area)
        ContourArea = np.array(ContourArea)
        # 找到大于detectpix的区域
        ContourArea = ContourArea[ContourArea > detectpix]
        # 取出区域的中心坐标
        region = []
        for cnt in contours:
            Area = cv2.contourArea(cnt)
            if Area > detectpix:
                region.append(cv2.boundingRect(cnt))   # 画出边缘的矩形区域
        return region

    def _corp_image(self, img_file, box, path):
        '''
        根据box截取图片并保存
        参数：
            img_file： 图像文件
            box：需要截取的box,格式：(左上xy，右下xy)。如：(333,333,666,666)
            path: 截取后图片的保存的名称
        返回：成功返回True，失败抛出异常
        作者：sumuzhong
        '''
        try:
            img = Image.open(r'%s' % img_file)
            new_img = img.crop(box)
            new_img.save(path)
            return True
        except Exception as e:
            raise Exception("Save error %s" % e.message)

    def get_different_area_number_of_images(self, model, image, is_save=False, detectpix=20):
        '''
        比较图片:
        参数:
            model: 图片模版(图片),图片路径(如：D:\\test\\Resolution_1080P.jpeg)
            image: 待比较图片,图片路径(如：D:\\test\\Resolution_1080P.jpeg)
            is_save: 是否保存不同点图片，默认不保存
                若置为True,保存为模版所在路径，图片名称在模版名称后加上不同点序号(如：D:\\test\\Resolution_1080P_1.jpeg)
            detectpix: 检测的最小区域，单位像素，默认20(作为本地菜单的检查，20pix足够检查出不同点)
        返回值：
            两张图片的不同区域个数
        作者：sumuzhong
        '''
        self._check_file(model)
        self._check_file(image)
        (filepath, filename) = os.path.split(model)
        name, ext = os.path.splitext(filename)
        img1 = Image.open(model)
        img2 = Image.open(image)
        hug_img = self._deal_with_2_images(img1, img2)
        binary_path = os.path.join(filepath, "binary_img.jpeg")
        binary_img = self._get_binary_image(hug_img, binary_path)

        binary_img = cv2_img_read(binary_path, 0)
        expend_img = self._expend(binary_img)

        corrosion_img = self._corrosion(expend_img)
        closed_path = os.path.join(filepath, "img_closed.jpeg")
        cv2.imwrite(closed_path, corrosion_img)

        box = self._get_region(model,
                               closed_path, int(detectpix))

        if os.path.exists(closed_path):
            os.remove(closed_path)
        if os.path.exists(binary_path):
            os.remove(binary_path)
        if str(is_save) == 'True':
            for i in range(len(box)):
                dif_img = os.path.join(filepath, "%s_%d.jpeg" % (name, i + 1))
                leftup_x, leftup_y, w, h = box[i]
                rightdown_x = leftup_x + w
                rightdown_y = leftup_y + h
                self._corp_image(model, (leftup_x, leftup_y,
                                         rightdown_x, rightdown_y), dif_img)
        return len(box)

    def get_different_area_number_of_images_ex(self, model, image, is_save=False, detectpix=20):
        '''
        比较图片,对比两次(model和image交换后在对比一次)
        参数:
            model: 图片模版(图片),图片路径(如：D:\\test\\Resolution_1080P.jpeg)
            image: 待比较图片,图片路径(如：D:\\test\\Resolution_1080P.jpeg)
            is_save: 是否保存不同点图片，默认不保存
                若置为True,保存为模版所在路径，图片名称在模版名称后加上不同点序号(如：D:\\test\\Resolution_1080P_1.jpeg)
            detectpix: 检测的最小区域，单位像素，默认20(作为本地菜单的检查，20pix足够检查出不同点)
        返回值：
            两次对比，返回不同区域的个数(两次对比最多的)
        说明：
            由于图片会经过灰度及二值化处理，导致某些情况下对比异常，所以进行交换对比后能够较为准确的判断两幅图片是否相同
        作者：sumuzhong
        '''
        dif_box_1 = self.get_different_area_number_of_images(
            model, image, is_save, detectpix)
        dif_box_2 = self.get_different_area_number_of_images(
            image, model, is_save, detectpix)
        return dif_box_1 if dif_box_1 > dif_box_2 else dif_box_2

    def corp_image(self, srcfile, dstfile, x, y, width, height):
        '''
        对图片进行裁剪操作，并保存裁剪后的图片。
        参数：
            srcfile: 被裁剪图片的路径
            dstfile：裁剪后的图片保存的路径
            x: 裁剪区域的左上角坐标x，默认值为0，即被裁剪图片的左上角
            y: 裁剪区域的左上角坐标y，默认值为0，即被裁剪图片的左上角
            width: 裁剪宽度，默认值为1920像素
            height: 裁剪高度，默认值为1080像素
        返回值：
            截屏操作成功，返回True;否则抛出异常
        作者：
            lianjiaxin
        '''
        self._check_file(srcfile)
        if not os.path.exists(os.path.dirname(dstfile)):
            os.makedirs(os.path.dirname(dstfile))
        img = Image.open(srcfile)
        new_img = img.crop((int(x), int(y), int(width) +
                            int(x), int(height) + int(y)))
        new_img.save(dstfile)
        return True

    def _detect_top_black_edge(self, im, threshold, detect_width):
        '''
        从图像顶部检测水平方向的黑边
        参数：
            im：读入的图像亮度矩阵
            threshold: 亮度阈值，亮度低于此值则认为是黑边
            detect_width: 检测范围(以像素为单位，只检测图片边缘detect_width宽度内是否存在黑边，可提高检测速度)
        返回：
            有黑边返回True，无黑边返回False
        '''
        # 从顶部检测
        for y in range(detect_width):
            row_check = 1
            for x in range(im.size[0]):
                if im.getpixel((x, y)) > threshold:  # 若某行出现了亮度大于阈值的像素，则该行不是黑边，row_check置为0
                    row_check = 0
                    break
            if row_check == 1:
                return True
        return False

    def detect_image_black_edge(self, filepath, threshold, detect_width):
        '''
        检测图像黑边
        参数：
            filepath: 图片路径
            threshold: 亮度阈值，亮度低于此值则认为是黑边，取值范围[0,255], 纯黑色亮度为0
            detect_width: 检测范围(以像素为单位，只检测图片边缘detect_width宽度内是否存在黑边，可提高检测速度)
        返回值：
            不存在黑边则返回True及空列表, 例如：(True, [])
            存在黑边则返回False及黑边位置的列表，例如：(False, ['bottom', 'left'])
        作者：
            yinting5
        '''
        im = Image.open(filepath)  # 载入
        im = im.convert("L")  # 灰度化
        threshold = int(threshold)
        detect_width = int(detect_width)
        if detect_width > im.size[0] or detect_width > im.size[1]:
            raise Exception("parameter 'detect_width' error，{} greater than image size {}*{}".format(
                detect_width, im.size[0], im.size[1]))

        top_detect = self._detect_top_black_edge(im, threshold, detect_width)

        im_bottom = im.transpose(Image.FLIP_TOP_BOTTOM)
        bottom_detect = self._detect_top_black_edge(
            im_bottom, threshold, detect_width)

        im_left = im.transpose(Image.ROTATE_270)
        left_detect = self._detect_top_black_edge(
            im_left, threshold, detect_width)

        im_right = im.transpose(Image.ROTATE_90)
        right_detect = self._detect_top_black_edge(
            im_right, threshold, detect_width)

        black_edge_list = []
        if top_detect:
            black_edge_list.append('top')
        if bottom_detect:
            black_edge_list.append('bottom')
        if left_detect:
            black_edge_list.append('left')
        if right_detect:
            black_edge_list.append('right')

        if len(black_edge_list) > 0:
            return False, black_edge_list
        else:
            return True, black_edge_list

    def _times_of_max_numpy_support_pixels(self, img):
        '''
        获取图像像素值是numpy支持的最大像素的倍数
        如果图像像素过大，矩阵运算时会出现内存溢出，Python 32位进程最大内存为2GB
        参数：
            图片数组
        返回：
            返回图像像素是最大Numpy支持像素的倍数
        '''
        return (img.shape[0] * img.shape[1]) / MAX_PIXELS_FOR_NUMPY

    def count_dead_pixels(self, filepath, comparison="lower", intensity=10):
        '''
        统计图像坏点个数。
        参数：
            comparison:像素灰度比较方式
                lower: 图片像素中低于灰度阈值intensity的视为坏点，用于检测白色图片黑色坏点
                larger: 图片像素中高于灰度阈值intensity的视为坏点，用于检测黑色图片中的黑色坏点
            intensity: 坏点图像的灰度阈值
        返回：
            返回图像坏点个数
        作者：
            lianjiaxin
        '''
        self._check_file(filepath)
        im = cv2_img_read(filepath, cv2.COLOR_BGR2GRAY)
        if str(comparison).lower() == "lower":
            return np.sum(im < int(intensity))
        elif str(comparison).lower() == "larger":
            return np.sum(im > int(intensity))
        else:
            raise Exception(
                "value of comparison is invalid: {}".format(comparison))


def get_all_slice_from_array(narray, height_num=4, width_num=4):
    '''
    将数组的第一维、第二维进行切片处理，得到(height_num * width_num)个子数组，并以列表形式返回分割后子数组
    参数：
        narray:传入的数组
        height_num:第一维分割个数
        width_num:第二维分割个数
    返回：将分割后的子数组以列表形式返回
    '''
    try:
        height_num = int(height_num)
        width_num = int(width_num)
    except Exception as e:
        raise e
    height = narray.shape[0]
    width = narray.shape[1]
    sub_h = height / height_num
    sub_w = width / width_num
    sub_array_list = []
    for x in range(0, height_num):
        for y in range(0, width_num):
            sub_array_list.append(
                narray[x * sub_h:(x + 1) * sub_h, y * sub_w:(y + 1) * sub_w])
    return sub_array_list
