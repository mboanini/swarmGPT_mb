"""
Copyright (c) 2024 WindyLab of Westlake University, China
All rights reserved.

This software is provided "as is" without warranty of any kind, either
express or implied, including but not limited to the warranties of
merchantability, fitness for a particular purpose, or non-infringement.
In no event shall the authors or copyright holders be liable for any
claim, damages, or other liability, whether in an action of contract,
tort, or otherwise, arising from, out of, or in connection with the
software or the use or other dealings in the software.
"""

# -*- coding: utf-8 -*-

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def create_char_image(char, font_path, font_size=200, image_size=(300, 300)):
    img = np.zeros((image_size[0], image_size[1], 3), dtype=np.uint8)
    pil_img = Image.fromarray(img)
    draw = ImageDraw.Draw(pil_img)
    font = ImageFont.truetype(font_path, font_size)
    text_size = draw.textbbox((0, 0), char, font=font)
    text_x = (image_size[1] - (text_size[2] - text_size[0])) // 2
    text_y = (image_size[0] - (text_size[3] - text_size[1])) // 2
    draw.text((text_x, text_y), char, font=font, fill=(255, 255, 255))
    img = np.array(pil_img)
    # cv2.imshow('img', img)
    # cv2.waitKey(0)
    return img


def create_star_image(
    image_size=(300, 300), num_points=5, radius_inner=50, radius_outer=100
):
    center = (image_size[1] // 2, image_size[0] // 2)
    angle_offset = np.pi / num_points  # 用于生成星形的角度偏移

    points = []
    for i in range(2 * num_points):
        angle = i * np.pi / num_points
        if i % 2 == 0:
            radius = radius_outer
        else:
            radius = radius_inner
        x = int(center[0] + radius * np.cos(angle + angle_offset))
        y = int(center[1] + radius * np.sin(angle + angle_offset))
        points.append((x, y))

    # 创建一个空白图像
    img = np.zeros((image_size[0], image_size[1], 3), dtype=np.uint8)

    # 绘制海星形状
    cv2.fillPoly(img, [np.array(points)], (255, 255, 255))
    cv2.imshow("Star Shape", img)
    cv2.waitKey(0)

    return img


def draw_contours(img):
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary_img = cv2.threshold(gray_img, 1, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    return contours


def sample_contour_points(contours, num_points):
    all_points = []
    total_length = 0

    for contour in contours:
        contour = contour[:, 0, :]
        distances = np.sqrt(np.sum(np.diff(contour, axis=0) ** 2, axis=1))
        total_length += np.sum(distances)
        all_points.append((contour, distances))

    if total_length == 0:
        return []

    sample_distance = total_length / num_points
    sampled_points = []
    accumulated_distance = 0

    for contour, distances in all_points:
        for i in range(1, len(contour)):
            segment_distance = distances[i - 1]
            accumulated_distance += segment_distance

            while accumulated_distance >= sample_distance:
                ratio = (accumulated_distance - sample_distance) / segment_distance
                new_point = contour[i] - ratio * (contour[i] - contour[i - 1])
                sampled_points.append(tuple(new_point))
                accumulated_distance -= sample_distance

    return sampled_points


def validate_contour_points(char, num_points=150, star=False):
    from modules.utils.root import get_project_root

    font_path = f"{get_project_root()}/assets/fonts/HanYiCuYuanJian-1.ttf"
    if not star:
        image = create_char_image(char, font_path)
    else:
        image = create_star_image()

    contours = draw_contours(image)
    sampled_points = sample_contour_points(contours, num_points)
    # print(len(sampled_points))
    # sampled_img = np.zeros_like(image)
    # for point in sampled_points:
    #     cv2.circle(sampled_img, (int(point[0]), int(point[1])), 1, (255, 255, 255), -1)
    #
    # cv2.imshow('Sampled Contour Points', sampled_img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    sampled_points = np.array(sampled_points)
    # remap to pygame coordinates
    sampled_points = sampled_points / 300 * 5 - 2.5
    return sampled_points


# # Example usage:
# font_path = "HanYiCuYuanJian-1.ttf"
# character = '国'
# num_points = 150  # Adjust the number of points as needed
# validate_contour_points(character, num_points)
