import cv2
import numpy as np

rgb_image_path = "../asserts/10.jpg"


def edge_repair(mask):
    # 查找边缘
    edges = cv2.Canny(mask, 100, 200)

    # 膨胀边缘以使其明显
    kernel = np.ones((9, 9), np.uint8)
    dilated_edges = cv2.dilate(edges, kernel)

    # 使用Poisson融合修复边缘
    return cv2.inpaint(mask, dilated_edges, 3, cv2.INPAINT_TELEA)


def get_original_edge(image_path):
    # 读取灰度图
    gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # 高斯模糊
    gaussian = cv2.GaussianBlur(gray, (3, 3), 0)

    # 边缘检测
    canny = cv2.Canny(gaussian, 50, 150)
    return canny


def show_contours(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # 二值化
    ret, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    # 查找轮廓
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 绘制并填充轮廓
    res = cv2.drawContours(image.copy(), contours, -1, (0, 0, 255), thickness=2)
    cv2.imshow('res', res)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return res


def get_closed_edges(image_path):
    image = cv2.imread(image_path)
    # 读取灰度图
    gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # 高斯模糊
    gaussian = cv2.GaussianBlur(gray, (3, 3), 0)

    # 边缘检测
    canny = cv2.Canny(gaussian, 50, 150)

    # 查找轮廓
    contours, _ = cv2.findContours(canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    print("轮廓数量", len(contours))
    contours = sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True)
    for i in contours:
        print(cv2.contourArea(i), cv2.arcLength(i, True))
    # 过滤出面积>100的轮廓
    contours = [i for i in contours if cv2.contourArea(i) >= 800 and cv2.arcLength(i, True) >= 500]

    # 过滤未闭合的轮廓并绘制闭合的轮廓
    res = cv2.drawContours(image, contours, -1, (0, 0, 255), thickness=2)

    return res


if __name__ == '__main__':
    closed_edges = get_closed_edges(rgb_image_path)
    cv2.imshow("Closed Edges", closed_edges)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
