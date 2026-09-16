import cv2
import numpy as np
from PIL import Image

def enhance_image(pil_image: Image.Image) -> Image.Image:
    try:
        img = np.array(pil_image.convert("RGB"))
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)

        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 11, 2
        )

        edges = cv2.Canny(binary, 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100, minLineLength=100, maxLineGap=10)
        if lines is not None:
            angles = []
            for line in lines:
                x1, y1, x2, y2 = line[0]
                if x2 != x1:
                    angles.append(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
            if angles:
                median_angle = np.median(angles)
                if abs(median_angle) > 0.5:
                    h, w = binary.shape
                    M = cv2.getRotationMatrix2D((w/2, h/2), median_angle, 1.0)
                    binary = cv2.warpAffine(binary, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

        return Image.fromarray(binary)
    except Exception:
        return pil_image
