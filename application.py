import cv2
import numpy as np

def image_matting(img, bg, mask, x=0, y=0, x_deta=100, y_deta=100, is_show=False, is_save=True):
    """
    人像抠图并换背景（适配无图形界面的服务器环境）
    :param img: 原图 BGR
    :param bg: 背景图 BGR
    :param mask: 人像掩膜 (单通道，0或255 / 0~1浮点数均可)
    :param x, y: 背景图中放置人像的位置（简单版未使用）
    :param x_deta, y_deta: 调整大小（简单版未使用）
    :param is_show: 是否显示（服务器环境设为False）
    :param is_save: 是否保存（设为True，保存图片到本地）
    """
    # 统一掩码与背景图尺寸
    mask = cv2.resize(mask, (bg.shape[1], bg.shape[0]), interpolation=cv2.INTER_NEAREST)
    
    # 归一化掩码到0-255，并转为uint8
    if mask.dtype == np.float32 or mask.dtype == np.float64:
        mask = (mask * 255).astype(np.uint8)
    elif mask.dtype != np.uint8:
        mask = mask.astype(np.uint8)
    
    # 阈值处理
    _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    mask_inv = cv2.bitwise_not(mask)

    # 原图缩放到和背景图一致
    img = cv2.resize(img, (bg.shape[1], bg.shape[0]), interpolation=cv2.INTER_CUBIC)
    
    # 抠图合成
    fg = cv2.bitwise_and(img, img, mask=mask)
    bg_roi = cv2.bitwise_and(bg, bg, mask=mask_inv)
    combined = cv2.add(fg, bg_roi)

    # ========== 核心修改：注释imshow，仅保留保存逻辑 ==========
    if is_show:
        # 服务器环境无法显示，打印提示
        print("警告：当前为无图形界面环境，无法显示窗口，已自动保存图片到本地")
        # cv2.imshow('Matting Result', combined)  # 注释掉
        # cv2.waitKey(0)                         # 注释掉
        # cv2.destroyAllWindows()                # 注释掉
    if is_save:
        cv2.imwrite('matting_result.png', combined)
        print(f"抠图结果已保存到：{Path.cwd()}/matting_result.png")  # 新增提示
    return combined

def highlight_human(img, mask, color=[193, 182, 255], is_show=False, is_save=True):
    """
    人像高亮（适配无图形界面的服务器环境）
    :param img: 原图 BGR
    :param mask: 人像掩膜 (0/255)
    :param color: 高亮颜色 BGR
    """
    # 同步修复掩码
    mask = cv2.resize(mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
    if mask.dtype == np.float32 or mask.dtype == np.float64:
        mask = (mask * 255).astype(np.uint8)
    elif mask.dtype != np.uint8:
        mask = mask.astype(np.uint8)
    _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

    color_mask = np.zeros_like(img)
    color_mask[:] = color
    fg = cv2.bitwise_and(img, img, mask=mask)
    color_fg = cv2.bitwise_and(color_mask, color_mask, mask=mask)
    highlighted = cv2.addWeighted(fg, 0.5, color_fg, 0.5, 0)
    bg = cv2.bitwise_and(img, img, mask=cv2.bitwise_not(mask))
    result = cv2.add(bg, highlighted)

    # ========== 核心修改：注释imshow ==========
    if is_show:
        print("警告：当前为无图形界面环境，无法显示窗口，已自动保存图片到本地")
        # cv2.imshow('Highlight Result', result)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
    if is_save:
        cv2.imwrite('highlight_result.png', result)
        print(f"高亮结果已保存到：{Path.cwd()}/highlight_result.png")  # 新增提示
    return result

if __name__ == '__main__':
    from pathlib import Path  # 新增导入，用于打印保存路径
    # 示例用法，需准备图片
    img = cv2.imread('image.png')
    mask = cv2.imread('mask.png', cv2.IMREAD_GRAYSCALE)
    bg = cv2.imread('bg.png')
    
    # 输入合法性检查
    if img is None:
        print("错误：未找到 image.png，请确认文件存在")
    if mask is None:
        print("错误：未找到 mask.png，请确认文件存在")
    if bg is None:
        print("错误：未找到 bg.png，请确认文件存在")
    
    if img is not None and mask is not None and bg is not None:
        # ========== 核心修改：is_show设为False（默认） ==========
        image_matting(img, bg, mask, is_show=False, is_save=True)
        highlight_human(img, mask, is_show=False, is_save=True)
    else:
        print("请确保 image.png, mask.png, bg.png 都放在当前目录")