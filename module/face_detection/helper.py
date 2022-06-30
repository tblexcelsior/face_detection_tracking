import cv2 as cv
import numpy as np
import os
import random


def sliding_window(image, kernel_size=[12, 12], strides=2):
    rows, cols, channels = image.shape
    # Iterate through x, y with step size = 2 to get window
    for y in range(0, rows - kernel_size[1], strides):
        for x in range(0, cols - kernel_size[0], strides):
            yield (x, y, image[y: y + kernel_size[1], x: x + kernel_size[0]])


def create_scale(bbox):
    facesize = [12, 11, 10]
    height = bbox[2]
    width = bbox[3]
    larger = max(height, width)
    scale = [larger/i for i in facesize]
    return scale


def scale_image(image, scale):
    h, w, channel = image.shape
    scaled_image = cv.resize(
        image, (int(w // scale), int(h // scale)), cv.INTER_AREA)
    return scaled_image


def scale_box(bbox, scale):
    bbox = [int(i / scale) for i in bbox]
    return bbox


def extract_bbox(filepath):
    # name_split pattern [file_name, 't***', 'l***', 'h***', 'w***.jpg']
    name_split = filepath.split('_')
    top = int(name_split[1][1:])
    left = int(name_split[2][1:])
    height = int(name_split[3][1:])
    width = int(name_split[4][1:-4])
    return [top, left, height, width]


def train_neg_gen(file_path, bbox, count):
    neg_folder = "datasets/neg/"
    image = cv.imread(file_path)
    scales = create_scale(bbox)
    bbox = [bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]]
    for s in scales:
        img = scale_image(image, s)
        box = scale_box(bbox, s)
        height, width, _ = img.shape
        y1 = 0
        # If image is too small => Skip
        if width <= 13 or height <= 13:
            continue
        # Randomly pick 15 x, y coordinate to crop image
        for i in range(15):
            x = random.randint(0, width-13)
            y = random.randint(0, height - 13)
            # Check if cropped image overlap with our ground truth bounding box
            if [x, y] > [box[1]-9, box[0]-9] and [x, y] < [box[3], box[2]]:
                continue
            else:
                crop_img = img[y:y+12, x:x+12]
                fn = neg_folder + str(count) + '.jpg'
                cv.imwrite(fn, crop_img)
                count += 1
    return count


def train_pos_gen(file_path, bbox, count):
    pos_folder = "datasets/pos/"
    image = cv.imread(file_path)
    scales = create_scale(bbox)
    # Convert bounding box from [top, left, height, width] to [x1, y1, x2, y2]
    bbox = [bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]]
    for s in scales:
        img = scale_image(image, s)
        box = scale_box(bbox, s)

        height, width, _ = img.shape
        if width <= 13 or height <= 13:
            continue
        all_crops = sliding_window(img, strides=2)
        for x1, y1, crop in all_crops:
            if x1 >= box[1] - 2 and x1 < box[1] and y1 > box[0] - 1 and y1 < box[0] + 1:
                x2 = x1 + 12
                y2 = y1 + 12
                # Normalize coordinate to [0, 1]
                normalized_x1 = (box[1] - x1) / \
                    12 if (box[1] - x1) / 12 > 0 else 0
                normalized_y1 = (box[0] - y1) / \
                    12 if (box[0] - y1) / 12 > 0 else 0
                normalized_x2 = (box[3] - x1) / \
                    12 if (box[3] - x1) / 12 < 1 else 1
                normalized_y2 = (box[2] - y1) / \
                    12 if (box[2] - y1) / 12 < 1 else 1

                fn = pos_folder + '{}_x1_{:.2f}_y1_{:.2f}_x2_{:.2f}_y2_{:.2f}_'.format(
                    count, normalized_x1, normalized_y1, normalized_x2, normalized_y2) + '.jpg'
                cv.imwrite(fn, crop)
                count += 1
    return count


def non_max_supression(bbox, threshold=0.3):
    # Reference: https://viblo.asia/p/tim-hieu-va-trien-khai-thuat-toan-non-maximum-suppression-bJzKmr66Z9N
    # Get bounding box coordinate and confidence score
    x1 = bbox[:, 0]
    y1 = bbox[:, 1]
    x2 = bbox[:, 2]
    y2 = bbox[:, 3]
    scores = bbox[:, 4]
    # Calculate Area of each bounding box
    areas = (x2 - x1) * (y2 - y1)
    # Sort confidence scores list in ascending order and return their index
    order = scores.argsort()

    # To store appropriate bounding box
    keep = []

    while(len(order) > 0):
        # Get the largest score index
        idx = order[-1]
        # Because it has largest confidence score => We keep it
        keep.append(bbox[idx])
        # Remove its index from order list
        order = order[:-1]
        # Get its coordinate
        xx1 = np.take(x1, axis=0, indices=order)
        xx2 = np.take(x2, axis=0, indices=order)
        yy1 = np.take(y1, axis=0, indices=order)
        yy2 = np.take(y2, axis=0, indices=order)

        # Get the intersection height and width
        w = xx2 - xx1
        h = yy2 - yy1
        # If w or h < 0 -> set w = 0, h = 0
        # If w > 480 -> set w = 480
        # If h > 640 -> set h = 640
        w = np.clip(w, a_min=0.0, a_max=480.0)
        h = np.clip(h, a_min=0.0, a_max=640.0)

        # Intersection Area
        inter = w*h
        # Get the area of all left bounding box
        rem_areas = np.take(areas, axis=0, indices=order)
        # Calculate Union Region
        union = (rem_areas - inter) + areas[idx]
        # Calculate IoU
        IoU = inter/union
        # Create mask to remove any bounding box that has IoU larger than threshold
        mask = IoU < threshold
        order = order[mask]
    return keep


def pipeline(image, model):
    scales = [20, 21, 22, 23]
    # To contain all bounding box from each scale
    final_bbox = np.zeros((1, 5))
    for s in scales:
        # To contain all bounding box from each kernel
        out_bbox = []
        scaled_img = scale_image(image, s)
        for x, y, window in sliding_window(scaled_img):
            window = np.asarray(window)
            window = window.reshape((1, 12, 12, 3))
            prediction = model.predict(window)
            confidence = np.squeeze(prediction[0])
            bbox = np.squeeze(prediction[1])
            # Only get the bounding box that has class 0 confidence score larger than 0.8
            if np.argmax(confidence) == 0 and max(confidence) > 0.8:
                # Convert bounding box coordinate to [0, 1] range
                bbox = np.where(bbox > 0, bbox, 0)
                bbox = np.where(bbox < 1, bbox, 1)
                bbox = np.multiply(bbox, 12)
                # Convert coordinate to global coordinate
                bbox = np.add(bbox, [x, y, x, y])
                # Unscaled bounding box
                bbox = scale_box(bbox, 1/s)
                # Concatenate bounding box and confidence score
                out_prediction = np.hstack((bbox, max(confidence)))
                out_bbox.append(out_prediction)

        out_bbox = np.asarray(out_bbox)
        # Apply Non max suppresion for each kernel
        s_nms = np.asarray(non_max_supression(out_bbox, 0.1))
        final_bbox = np.concatenate((final_bbox, s_nms))
    final_bbox = np.asarray(final_bbox[1:])
    # Apply Non max suppresion for all kernel
    result = non_max_supression(final_bbox, 0.3)
    return result


# This code is used to create positive image

# if __name__ == '__main__':
#     image_path = 'datasets/images/'
#     lst_dir = os.listdir(image_path)
#     count = 0
#     for f in lst_dir:
#         f_path = image_path + f
#         bbox = extract_bbox(f)
#         count = train_pos_gen(f_path, bbox, count)


# This code is used to create negative image

# if __name__ == '__main__':
#     image_path = 'datasets/images/'
#     lst_dir = os.listdir(image_path)
#     count = 0
#     for f in lst_dir:
#         f_path = image_path + f
#         bbox = extract_bbox(f)
#         count = train_neg_gen(f_path, bbox, count)
