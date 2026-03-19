import cv2
import numpy as np

def image_matting(img, bg, mask, x=0, y=0, x_deta=100, y_deta=100, is_show=False, is_save=True):
    """
    人像抠图并换背景
    :param img: 原图 BGR
    :param bg: 背景图 BGR
    :param mask: 人像掩膜 (单通道，0或255)
    :param x, y: 背景图中放置人像的位置（简单版未使用）
    :param x_deta, y_deta: 调整大小（简单版未使用）
    :param is_show: 是否显示
    :param is_save: 是否保存
    """
    _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    mask_inv = cv2.bitwise_not(mask)

    fg = cv2.bitwise_and(img, img, mask=mask)
    bg_roi = cv2.bitwise_and(bg, bg, mask=mask_inv)
    combined = cv2.add(fg, bg_roi)

    if is_show:
        cv2.imshow('Matting Result', combined)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    if is_save:
        cv2.imwrite('matting_result.jpg', combined)
    return combined

def highlight_human(img, mask, color=[193, 182, 255], is_show=False, is_save=True):
    """
    人像高亮
    :param img: 原图 BGR
    :param mask: 人像掩膜 (0/255)
    :param color: 高亮颜色 BGR
    """
    color_mask = np.zeros_like(img)
    color_mask[:] = color
    fg = cv2.bitwise_and(img, img, mask=mask)
    color_fg = cv2.bitwise_and(color_mask, color_mask, mask=mask)
    highlighted = cv2.addWeighted(fg, 0.5, color_fg, 0.5, 0)
    bg = cv2.bitwise_and(img, img, mask=cv2.bitwise_not(mask))
    result = cv2.add(bg, highlighted)

    if is_show:
        cv2.imshow('Highlight Result', result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    if is_save:
        cv2.imwrite('highlight_result.jpg', result)
    return result

if __name__ == '__main__':
    # 示例用法，需准备图片
    img = cv2.imread('image.jpg')
    mask = cv2.imread('mask.png', cv2.IMREAD_GRAYSCALE)
    bg = cv2.imread('bg.jpg')
    if img is not None and mask is not None and bg is not None:
        image_matting(img, bg, mask, is_show=True)
        highlight_human(img, mask, is_show=True)
    else:
        print("请准备 image.jpg, mask.png, bg.jpg 放在当前目录")