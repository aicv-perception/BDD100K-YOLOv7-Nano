import numpy as np
import torch
import torch.nn as nn
from PIL import ImageDraw
from utils.utils import cvtColor, get_anchors, get_classes, preprocess_input, resize_image
from utils.utils_bbox import DecodeBox

class YOLO(object):
    def __init__(self, imgSize=None, model_path=None, cuda=False, classes_path=None, anchors_path=None):
        super(YOLO, self).__init__()
        self.confidence = 0.3
        self.nms_iou = 0.45
        self.letterbox_image = False
        self.anchors_mask = [[], [0, 1, 2, 3, 4, 5], []]

        self.classes_path = classes_path
        self.anchors_path = anchors_path
        self.input_shape = imgSize
        self.model_path = model_path
        self.cuda = cuda
        self.class_names, self.num_classes = get_classes(self.classes_path)
        self.anchors, self.num_anchors = get_anchors(self.anchors_path)
        self.bbox_util = DecodeBox(self.anchors, self.num_classes, (self.input_shape[0], self.input_shape[1]), self.anchors_mask)
        self.generate()

    def generate(self):
        device = torch.device('cuda' if self.cuda else 'cpu')
        self.net = torch.jit.load(self.model_path, map_location=device)
        if self.cuda:
            self.net = nn.DataParallel(self.net)
            self.net = self.net.to(device)
        self.net.eval()

        print('{} model, and classes loaded.'.format(self.model_path))

    def detect_image(self, image):
        image_shape = np.array(np.shape(image)[0:2])
        image = cvtColor(image)
        image_data = resize_image(image, (self.input_shape[1], self.input_shape[0]), self.letterbox_image)
        image_data = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, dtype='float32')), (2, 0, 1)), 0)

        with torch.no_grad():
            images = torch.from_numpy(image_data)
            if self.cuda:
                images = images.cuda()
            outputs = self.net(images)
            outputs = self.bbox_util.decode_box(outputs)
            results = self.bbox_util.non_max_suppression(torch.cat(outputs, 1), self.num_classes, self.input_shape,
                                                         image_shape, self.letterbox_image,
                                                         conf_thres=self.confidence, nms_thres=self.nms_iou)
            if results[0] is None:
                return image

            top_label = np.array(results[0][:, 6], dtype='int32')
            top_conf = results[0][:, 4] * results[0][:, 5]
            top_boxes = results[0][:, :4]

        thickness   = int(max((image.size[0] + image.size[1]) // np.mean(self.input_shape), 1))

        for i, c in list(enumerate(top_label)):
            predicted_class = self.class_names[int(c)]
            box = top_boxes[i]
            score = top_conf[i]

            top, left, bottom, right = box

            top = max(0, np.floor(top).astype('int32'))
            left = max(0, np.floor(left).astype('int32'))
            bottom = min(image.size[1], np.floor(bottom).astype('int32'))
            right = min(image.size[0], np.floor(right).astype('int32'))
            draw = ImageDraw.Draw(image)

            for i in range(thickness):
                draw.rectangle([left + i, top + i, right - i, bottom - i], outline=(0, 0, 255))
            del draw

        return image