import time
import cv2
import numpy as np
from PIL import Image
from yolo import YOLO
import argparse
import datetime
x = datetime.datetime.now()
date = "%s%s" % (x.strftime("%m"), x.strftime("%d"))

def main(opts):
    mode = opts.mode
    video_save_path = opts.video_save_path
    input_shape = opts.input_shape
    model_path = opts.model_path
    yolo = YOLO(imgSize=input_shape, model_path=model_path, cuda=opts.cuda,
                classes_path=opts.classes_path, anchors_path=opts.anchors_path)

    if mode == "image":
        while True:
            img = input('Input image filename:')
            try:
                image = Image.open(img)
            except:
                print('Open Error! Try again!')
                continue
            else:
                r_image = yolo.detect_image(image)
                r_image.show()

    elif mode == "video":
        capture = cv2.VideoCapture(opts.video_path)
        if video_save_path != "":
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            size = (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
            out = cv2.VideoWriter(video_save_path, fourcc, 30, size)

        ref, frame = capture.read()
        if not ref:
            raise ValueError("can not load camera or video ! ")

        fps = 0.0
        while(True):
            t1 = time.time()

            ref, frame = capture.read()
            if not ref:
                break

            frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
            frame = Image.fromarray(np.uint8(frame))
            frame = np.array(yolo.detect_image(frame))
            frame = cv2.cvtColor(frame,cv2.COLOR_RGB2BGR)
            fps = (fps + (1./(time.time()-t1)) ) / 2
            frame = cv2.putText(frame, "fps= %.2f" % (fps), (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            cv2.imshow("video", frame)
            c= cv2.waitKey(1) & 0xff
            if video_save_path!="":
                out.write(frame)

            if c==27:
                capture.release()
                break

        print("Video Detection Done!")
        capture.release()
        if video_save_path!="":
            print("Save processed video to the path :" + video_save_path)
            out.release()
        cv2.destroyAllWindows()
    else:
        raise AssertionError("Please specify the correct mode: 'image', 'video'.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(" inference ")
    parser.add_argument("--cuda", type=bool, default=False, help="use gpu.")

    parser.add_argument("--mode", type=str, default="video", help="Test Mode.",
                        choices=['image', 'video'])

    parser.add_argument("--input_shape", type=int, nargs='+', default=[640, 640],
                        help="Save Path to Result Video.")

    parser.add_argument("--video_path", type=str, default="./MOVI0453.avi",
                        help="Path to Testing Video.")
    parser.add_argument("--video_save_path", type=str, default="",
                        help="Save Path to Result Video. video_save_path="" means don't save the video.")

    parser.add_argument("--model_path", type=str, default='./Nano640_bdd100k_original_annotation.pt',
                        help="Path to Your Model.")

    parser.add_argument("--classes_path", type=str, default='./utils/bdd100k_class.txt',
                        help="Path to Your Model.")

    parser.add_argument("--anchors_path", type=str, default="./utils/bdd_nano_640_6_anchors.txt",
                        help="Path to Your Model.")

    args = parser.parse_args()
    main(args)
