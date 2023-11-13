import cv2
import numpy as np

img = cv2.imread('test.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Apply Gaussian blur to reduce noise
gray_blur = cv2.GaussianBlur(gray, (9, 9), 2)

# Find them circles!
def find_circles(image, r1, r2):
    res = img.copy()
    circles = cv2.HoughCircles(
        gray_blur,
        cv2.HOUGH_GRADIENT, 1, 20,
        param1=50,
        param2=30,
        minRadius=r1,
        maxRadius=r2
    )
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        for (x, y, r) in circles:
            print(f'Circle at: x={x} | y={y} | r={r}')
            cv2.circle(res, (x, y), r, (0, 255, 0), 2)
    else:
        print('well shit..]. no circles')
    return res

result = find_circles(gray_blur, 90, 500)

print('Saving to out.jpg')
cv2.imwrite('out.jpg', result)
