import cv2
import numpy as np

mask_image_path = "../asserts/result1.jpg"
rgb_image_path = "../asserts/10.jpg"


def post_process_image(image_path):
    # 读取二值化的mask图像
    mask = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    filled_mask = hole_filled(mask)
    repaired_mask = edge_repair(filled_mask)

    return repaired_mask


def hole_filled(mask):
    kernel = np.ones((5, 5), np.uint8)
    # 闭运算
    closed_mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # 查找轮廓
    contours, _ = cv2.findContours(closed_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 绘制并填充轮廓
    return cv2.drawContours(closed_mask, contours, -1, (255, 255, 255), thickness=cv2.FILLED)


def edge_repair(mask):
    # 查找边缘
    edges = cv2.Canny(mask, 100, 200)

    # 膨胀边缘以使其明显
    kernel = np.ones((9, 9), np.uint8)
    # dilated_edges = cv2.dilate(edges, kernel)
    closed_mask = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    # 使用Poisson融合修复边缘
    return cv2.inpaint(mask, closed_mask, 5, cv2.INPAINT_TELEA)


if __name__ == '__main__':
    mask_original = cv2.imread(mask_image_path)
    result = post_process_image(mask_image_path)
    cv2.imshow("mask original", mask_original)
    cv2.imshow("result", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
