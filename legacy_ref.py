"""旧 Image.pyw の計算式を、画面部分を除いてそのまま写した基準（比較テスト用）。"""
import cv2
import numpy as np
from PIL import Image, ImageDraw


def gamma(img, g):
    return np.clip(np.power(img / 255.0, 1.0 / g) * 255.0, 0, 255).astype(np.uint8)


def sepia(img):
    m = np.array([[0.393, 0.769, 0.189], [0.349, 0.686, 0.168], [0.272, 0.534, 0.131]])
    return np.clip(np.dot(img, m.T), 0, 255).astype(np.uint8)


def pencil(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)             # 旧版は RGB 配列に BGR2GRAY を使っていた
    inv = cv2.bitwise_not(gray)
    sm = cv2.GaussianBlur(inv, (21, 21), sigmaX=0, sigmaY=0)
    return cv2.divide(gray, 255 - sm, scale=256)


def contrast(img, alpha):
    return cv2.convertScaleAbs(img, alpha=alpha, beta=0.0)


def clahe(img):
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2Lab)
    l, a, b = cv2.split(lab)
    cl = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(l)
    return cv2.cvtColor(cv2.merge((cl, a, b)), cv2.COLOR_Lab2RGB)


def sharp(img):
    blurred = cv2.GaussianBlur(img, (0, 0), 3)
    return cv2.addWeighted(img, 1.5, blurred, -0.5, 0)


def mosaic_whole(img, block):
    image = img.copy()
    h, w, _ = image.shape
    for y in range(0, h, block):
        for x in range(0, w, block):
            roi = image[y:y + block, x:x + block]
            image[y:y + block, x:x + block] = cv2.mean(roi)[:3]
    return image


def dot(img, sz=10):
    h, w = img.shape[:2]
    out = Image.new("RGB", (w, h))
    d = ImageDraw.Draw(out)
    for y in range(0, h, sz):
        for x in range(0, w, sz):
            roi = img[y:y + sz, x:x + sz].astype(np.float32)
            r, g, b = (int(np.uint8(roi[..., c].mean())) for c in range(3))
            d.ellipse((x + 1, y + 1, x + sz - 1, y + sz - 1), fill=(r, g, b))
    return np.asarray(out)


_DA1 = np.array([[0, 48, 12, 60, 3, 51, 15, 63], [32, 16, 44, 28, 35, 19, 47, 31],
                 [8, 56, 4, 52, 11, 59, 7, 55], [40, 24, 36, 20, 43, 27, 39, 23],
                 [2, 50, 14, 62, 1, 49, 13, 61], [34, 18, 46, 30, 33, 17, 45, 29],
                 [10, 58, 6, 54, 9, 57, 5, 53], [42, 26, 38, 22, 41, 25, 37, 21]], dtype=np.float32) / 64.0
_DA2 = np.array([[24, 10, 12, 26, 35, 47, 49, 37], [8, 0, 2, 14, 45, 59, 61, 51],
                 [22, 6, 4, 16, 43, 57, 63, 53], [30, 20, 18, 28, 33, 41, 55, 39],
                 [34, 46, 48, 36, 25, 11, 13, 27], [44, 58, 60, 50, 9, 1, 3, 15],
                 [42, 56, 62, 52, 23, 7, 5, 17], [32, 40, 54, 38, 31, 21, 19, 29]], dtype=np.float32) / 64.0


def bayer(img, met):
    g = cv2.cvtColor(img.astype(np.float32) / 255, cv2.COLOR_RGB2GRAY)
    s1, s2 = g.shape
    s1, s2 = s1 // 8 * 8, s2 // 8 * 8
    g = g[:s1, :s2]
    mask = np.tile(_DA1 if met == 1 else _DA2, (s1 // 8, s2 // 8))
    ret = (g < mask).astype(np.uint8) * 255
    return cv2.bitwise_not(ret)


def ulichney(img, od, bs):
    g = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255
    s1, s2 = g.shape
    mask = np.tile(od, (s2 // bs, s1 // bs)).T
    return (g > mask).astype(np.uint8) * 255
