import os

import cv2

img_dir = os.path.join('data', 'img')
img_path = os.path.join(img_dir, 'memorial.jpg')
outpath = 'output'

# load the image
img = cv2.imread(img_path)


# copy image to not overwrite original with the drawn rectangle
img_ROI = img.copy()

# coordinates for top left and bottom right corner of recangle
x1 = 1390
y1 = 870
x2 = 2870
y2 = 2790

# draw a rectangle around text
cv2.rectangle(img_ROI, (x1, y1), (x2, y2), (0,255,0), 2)

# save the file
ROI_name = 'image_with_ROI.jpg'
cv2.imwrite(os.path.join(outpath, ROI_name), img_ROI)

# crop with the rectangle coordinates
img_crop = img[y1:y2, x1:x2]

# save cropped file
crop_name = 'image_cropped.jpg'
cv2.imwrite(os.path.join(outpath, crop_name), img_crop)

# smooth out irregularites
img_blur = cv2.bilateralFilter(img_crop, 5, 75, 75)

# find edges
img_canny = cv2.Canny(img_blur, 50, 150)

# reduce to only the outer contours
contours, _ = cv2.findContours(img_canny.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# draw contours onto image
img_cont = cv2.drawContours(img_crop.copy(), contours, -1, (0, 255, 0), 1)

# save contoured image
letters_name = 'image_letters.jpg'
cv2.imwrite(os.path.join(outpath, letters_name), img_cont)