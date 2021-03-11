import os
from typing import List
import cv2
import numpy as np
from multiprocessing import Process

CATEGORIES = [
    "ballet",
    "break",
    "cha",
    "flamenco",
    "foxtrot",
    "jive",
    "latin",
    "pasodoble",
    "quickstep",
    "rumba",
    "samba",
    "square",
    "swing",
    "tango",
    "tap",
    "waltz",
]

CATEGORY_TO_ID = {c: i for i, c in enumerate(CATEGORIES)}

# read coco object names
coco_names_file = "yolo/coco.names"
LABELS = open(coco_names_file).read().strip().split("\n")
# print('The COCO dataset contains images of the following items: \n', LABELS)

# configure YOLOv3
yolov3_weight_file = "yolo/yolov3.weights"
yolov3_config_file = "yolo/yolov3.cfg"
net = cv2.dnn.readNetFromDarknet(yolov3_config_file, yolov3_weight_file)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

# get layers in the YOLO v3 model
ln = net.getLayerNames()
# print("--- YOLO v3 has {} layers: ---".format(len(ln)))
# print(ln)

# the output layers are those with unconnected output
ln_out = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]


# print("Names of YOLO v3 output layers: \n{}".format(ln_out))


# read image file
def load_img(path: str):
    image = cv2.imread(path)

    # determine image size
    (h, w) = image.shape[:2]

    # preprocess image data with rescaling and resizing
    # opencv assumes BGR images: we have to convert to RGB, with swapRB=True
    blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    return image, blob, w, h


def infer(img):
    net.setInput(img)
    return net.forward(ln_out)


def get_bounding_boxes(outputs, width: int, height: int):
    """
    - merges the three outputs
    - removes all empty bounding boxes (i.e. class_probability < threshold)
    - rescales the bounding boxes
    """

    # detected bounding boxes, obtained confidences and class's number
    boxes = []
    scores = []
    classes = []

    # this is our threshold for keeping the bounding box
    probability_minimum = 0.5

    # iterating through all three outputs
    for result in outputs:
        # going through all bounding boxes from current output layer
        for detection in result:
            # getting class for current object
            scores_current = detection[5:]
            class_current = np.argmax(scores_current)

            # getting probability for current object
            probability_current = scores_current[class_current]

            # getting object confidence for current object
            object_confidence = detection[4]

            # eliminating weak predictions by minimum probability
            if probability_current > probability_minimum:
                # if probability_current*object_confidence > probability_minimum:  # this is an alternative way

                # Scaling bounding box coordinates to the initial image size
                # by element-wise multiplying them with the width and height of the image
                box_current = np.array(detection[0:4]) * np.array([width, height, width, height])

                # YOLO data format keeps center of detected box and its width and height
                # here we reconstruct the top left and bottom right corner
                x_center, y_center, box_width, box_height = box_current.astype('int')
                x_min = int(x_center - (box_width / 2))
                y_min = int(y_center - (box_height / 2))
                x_max = int(x_center + (box_width / 2))
                y_max = int(y_center + (box_height / 2))

                # adding results into prepared lists
                boxes.append([x_min, y_min, x_max, y_max])
                scores.append(float(probability_current))
                classes.append(class_current)

    boxes = np.array(boxes)
    scores = np.array(scores)
    classes = np.array(classes)
    return boxes, scores, classes


def iou(box1, box2):
    """ Calculate IoU between box1 and box2
        box1/box2 : (x1, y1, x2, y2), where x1 and y1 are coordinates of upper left corner,
                    x2 and y2 are of lower right corner
        return: IoU
    """

    # get the area of intersection
    xi1 = max(box1[0], box2[0])
    yi1 = max(box1[1], box2[1])
    xi2 = min(box1[2], box2[2])
    yi2 = min(box1[3], box2[3])

    inter_area = max(xi2 - xi1, 0) * max(yi2 - yi1, 0)

    # get the area of union
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    union_area = box1_area + box2_area - inter_area

    # get iou
    iou = inter_area / union_area
    return iou


def yolo_non_max_suppression(boxes, scores, score_threshold=0.5, iou_threshold=0.5):
    """
    Apply Non-max suppression.
    boxes: Array of coordinates of boxes (x1, y1, x2, y2)
    scores: Array of confidence scores with respect to boxes
    score_threshold: Threshold, higher will be kept
    iou_threshold: Threshold, lower will be kept

    return: Indices of boxes and scores to be kept
    """
    sorted_idx = np.argsort(scores)[::-1]

    remove = []
    for i in np.arange(len(scores)):
        if i in remove:
            continue
        if scores[sorted_idx[i]] < score_threshold:
            remove.append(i)
            continue

        for j in np.arange(i + 1, len(scores)):
            if scores[sorted_idx[j]] < score_threshold:
                remove.append(j)
                continue

            overlap = iou(boxes[sorted_idx[i]], boxes[sorted_idx[j]])
            if overlap > iou_threshold:
                remove.append(j)

    sorted_idx = np.delete(sorted_idx, remove)
    return sorted(sorted_idx)


def get_final_bounding_box(boxes, nms_idx, width: int, height: int):
    """
    Get a final bounding box by finding a box that includes all boxes
    """
    x1 = np.inf
    y1 = np.inf
    x2 = -np.inf
    y2 = -np.inf

    bx = [boxes[i] for i in nms_idx]
    for box in bx:
        xmin = np.min(box[[0, 2]])
        xmax = np.max(box[[0, 2]])
        ymin = np.min(box[[1, 3]])
        ymax = np.max(box[[1, 3]])

        x1 = np.min([xmin, x1])
        y1 = np.min([ymin, y1])
        x2 = np.max([xmax, x2])
        y2 = np.max([ymax, y2])
    return x1, y1, x2, y2


def person_segmentation(path: str):
    img, blob, width, height = load_img(path)
    outputs = infer(blob)
    boxes, scores, classes = get_bounding_boxes(outputs, width, height)
    # apply non-max suppression
    nms_idx = yolo_non_max_suppression(boxes, scores, score_threshold=0.5, iou_threshold=0.5)
    nms_idx = filter_person(nms_idx, classes)

    x1, y1, x2, y2 = get_final_bounding_box(boxes, nms_idx, width, height)
    return img[int(y1):int(y2), int(x1):int(x2), :]


def filter_person(nms_idx, classes) -> List[int]:
    ret = []
    for i in nms_idx:
        if LABELS[classes[i]] == "person":
            ret.append(i)
    return ret


def number_of_people(path: str):
    img, blob, width, height = load_img(path)
    outputs = infer(blob)
    boxes, scores, classes = get_bounding_boxes(outputs, width, height)
    nms_idx = yolo_non_max_suppression(boxes, scores, score_threshold=0.5, iou_threshold=0.5)
    nms_idx = filter_person(nms_idx, classes)
    return len(nms_idx)


def preprocess_category(category: str, cat_dir: str, cat_out_dir: str):
    if category not in CATEGORIES:
        return

    # yolo inference
    for f in os.scandir(cat_dir):
        filename: str = f.name
        out_filename = f'{os.path.splitext(filename)[0]}.jpg'
        output_path = os.path.join(cat_out_dir, out_filename)

        if os.path.exists(output_path):
            print(f"Skipping {output_path}")
            continue

        try:
            img = person_segmentation(f.path)
            img = cv2.resize(img, (224, 224))
            cv2.imwrite(output_path, img)

            # from matplotlib import pyplot as plt
            # mm = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # plt.imshow(mm)
            # plt.show()
        except:
            print(f"{f.path} FAILED")

    print(f'Category {category} DONE')


def filter_by_num_people(category: str, cat_dir: str, cat_out_dir: str):
    from shutil import copyfile
    if category not in CATEGORIES:
        return

    # yolo inference
    for f in os.scandir(cat_dir):
        filename: str = f.name
        out_filename = f'{os.path.splitext(filename)[0]}.jpg'

        try:
            n = number_of_people(f.path)

            out_dir = os.path.join(cat_out_dir, f"{n}")
            os.makedirs(out_dir, exist_ok=True)

            output_path = os.path.join(out_dir, out_filename)
            copyfile(f.path, output_path)
        except:
            print(f"{f.path} FAILED")

    print(f'Category {category} DONE')


def main():
    out_dir = 'letsdance-filter-num-people'
    data_dir = 'letsdance/rgb'

    # procs = []
    for cat_dir in os.scandir(data_dir):  # categories
        if cat_dir.is_dir():
            category = cat_dir.name
            cat_out_dir = os.path.join(out_dir, category)
            os.makedirs(cat_out_dir, exist_ok=True)

            # procs.append(Process(target=preprocess_category, args=(category, cat_dir.path, cat_out_dir)))

            # preprocess_category(category, cat_dir.path, cat_out_dir)
            filter_by_num_people(category, cat_dir.path, cat_out_dir)

    # for p in procs:
    #     p.start()

    # for p in procs:
    #     p.join()


if __name__ == '__main__':
    main()
