import cv2
import time
import torch
import math
import argparse
import numpy as np
from utils.datasets import letterbox
from utils.torch_utils import select_device
from models.experimental import attempt_load
from utils.plots import output_to_keypoint, plot_skeleton_kpts, plot_one_box_kpt, colors
from utils.general import non_max_suppression_kpt, strip_optimizer
from torchvision import transforms
import tensorflow
import json
from PIL import ImageFont, ImageDraw, Image

def getAngles(replica_img, list_to_use):
    left_shoulder = (list_to_use[15], list_to_use[16])
    right_shoulder = (list_to_use[18], list_to_use[19])
    left_elbow = (list_to_use[21], list_to_use[22])
    right_elbow = (list_to_use[24], list_to_use[25])
    left_wrist = (list_to_use[27], list_to_use[28])
    right_wrist = (list_to_use[30], list_to_use[31])
    left_hip = (list_to_use[33], list_to_use[34])
    right_hip = (list_to_use[36], list_to_use[37])
    left_knee = (list_to_use[39], list_to_use[40])
    right_knee = (list_to_use[42], list_to_use[43])
    left_ankle = (list_to_use[45], list_to_use[46])
    right_ankle = (list_to_use[48], list_to_use[49])
    value = []
    value.append(findAngle(replica_img, left_hip, left_shoulder, left_elbow, True))
    value.append(findAngle(replica_img, right_elbow, right_shoulder, right_hip, True))
    value.append(findAngle(replica_img, left_shoulder, left_elbow, left_wrist, True))
    value.append(findAngle(replica_img, right_wrist, right_elbow, right_shoulder, True))
    value.append(findAngle(replica_img, left_ankle, left_knee, left_hip, True))
    value.append(findAngle(replica_img, right_ankle, right_knee, right_hip, True))
    value.append(findAngle(replica_img, right_knee, right_hip, right_shoulder, True))
    value.append(findAngle(replica_img, left_shoulder, left_hip, left_knee, True))

    return value

#Finding angle between 3-keypoints
def findAngle(image,p1,p2,p3,draw=True):
    x1 = p1[0]
    y1 = p1[1]
    x2 = p2[0]
    y2 = p2[1]
    x3 = p3[0]
    y3 = p3[1]
    
    #claculate the angle
    angle = math.degrees(math.atan2(y3-y2,x3-x2) - math.atan2(y1-y2, x1-x2))
    if angle < 0:
        angle += 360
    return angle

def load_classes(path):
    # Loads *.names file at 'path'
    with open(path, 'r') as f:
        names = f.read().split('\n')
    return list(filter(None, names))  # filter removes empty strings (such as last line)


@torch.no_grad()
def run(poseweights='yolov7-w6-pose.pt', source='pose.mp4', device='cpu', names='utils/coco.names', line_thickness=2):
    path = source
    ext = path.split('/')[-1].split('.')[-1].strip().lower()
    if ext in ["mp4", "webm", "avi"] or ext not in ["mp4", "webm", "avi"] and ext.isnumeric():
        input_path = int(path) if path.isnumeric() else path
        device = select_device(opt.device)
        names = load_classes(names)
        half = device.type != 'cpu'
        model = attempt_load(poseweights, map_location=device)
        _ = model.eval()

        cap = cv2.VideoCapture(input_path)
        webcam = False

        if (cap.isOpened() == False):
            print('Error while trying to read video. Please check path again')

        fw, fh = int(cap.get(3)), int(cap.get(4))
        if ext.isnumeric():
            webcam = True
            fw, fh = 1280, 768
        vid_write_image = letterbox(
            cap.read()[1], (fw), stride=64, auto=True)[0]

        resize_height, resize_width = vid_write_image.shape[:2]
        out_video_name = "output" if path.isnumeric(
        ) else f"{input_path.split('/')[-1].split('.')[0]}"
        out = cv2.VideoWriter(f"{out_video_name}_wed.mp4", cv2.VideoWriter_fourcc(
            *'mp4v'), 30, (resize_width, resize_height))
        if webcam:
            out = cv2.VideoWriter(f"{out_video_name}_mon.mp4", cv2.VideoWriter_fourcc(
                *'mp4v'), 30, (fw, fh))

        frame_count, total_fps = 0, 0

        # =3.0===Load custom font ===========
        fontpath = "sfpro.ttf"
        font = ImageFont.truetype(fontpath, 32)
        sequence = []
        keypoints = []
        sequence_angle = []
        keypoints_angle = []
        final_keypoints=[]
        j = 1
        seq = 2
        while cap.isOpened:

            print(f"Frame {frame_count} Processing")
            ret, frame = cap.read()
            if ret:
                origimage = frame
                mask = np.zeros(frame.shape[:2], dtype="uint8")
                roi = cv2.rectangle(mask, (293, 336), (984, 707), (255, 255, 255), -1)
                masked = cv2.bitwise_and(frame, frame, mask=mask)
                masked[np.where((masked == [0, 0, 0]).all(axis=2))] = [255, 0, 0]
                orig_image = masked

                # preprocess image
                image_ = cv2.cvtColor(origimage, cv2.COLOR_BGR2RGB)
                image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)
                image_ = letterbox(image_, (resize_width), stride=64, auto=True)[0]

                image = letterbox(image, (resize_width), stride=64, auto=True)[0]
                image_ = transforms.ToTensor()(image_)
                image = transforms.ToTensor()(image)
                image_ = torch.tensor(np.array([image_.numpy()]))
                image = torch.tensor(np.array([image.numpy()]))
                image_ = image_.to(device)
                image_ = image_.float()
                image = image.to(device)
                image = image.float()
                start_time = time.time()

                with torch.no_grad():
                    output, _ = model(image)

                output_data = non_max_suppression_kpt(output, 0.25, 0.65, nc=model.yaml['nc'], nkpt=model.yaml['nkpt'],
                                                      kpt_label=True)
                output = output_to_keypoint(output_data)

                img_ = image_[0].permute(1, 2, 0) * 255
                img_ = img_.cpu().numpy().astype(np.uint8)

                img_img = cv2.cvtColor(img_, cv2.COLOR_RGB2BGR)

                img = image[0].permute(1, 2, 0) * 255
                img = img.cpu().numpy().astype(np.uint8)

                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

                gn = torch.tensor(img.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                for i, pose in enumerate(output_data):  # detections per image
                    # flag = 1
                    if len(output_data):  # check if no pose
                        for c in pose[:, 5].unique():  # Print results
                            n = (pose[:, 5] == c).sum()  # detections per class
                            print("No of Objects in Current Frame : {}".format(n))
                        flag = 0
                        for det_index, (*xyxy, conf, cls) in enumerate(
                                reversed(pose[:, :6])):  # loop over poses for drawing on frame
                            c = int(cls)  # integer class
                            # print(c)
                            kpts = pose[det_index, 6:]
                            label = names[c]
                            # print(label)
                            plot_one_box_kpt(xyxy, img, label=label, color=colors(c, True),
                                             line_thickness=opt.line_thickness, kpts=kpts, steps=3,
                                             orig_shape=img.shape[:2])
                            if j <= seq:
                                kkk=0
                                for idx in range(output.shape[0]):
                                    kpts = output[idx, 7:].T
                                    # print(kpts.tolist())
                                    kkk+=1
                                    list_to_use = kpts.tolist()
                                    plot_skeleton_kpts(img_img, kpts, 3)
                                    temp=getAngles(img_img, list_to_use)
                                    sequence_angle.append(temp)
                                    kpts = kpts[15:]
                                    seqq=[]
                                    for i in range(len(kpts)):
                                        if (i + 1) % 3 != 0:  # Check if the index is not a multiple of 3
                                            seqq.append(kpts[i])
                                    sequence.append(seqq)

                                    
                                    break

                            if len(sequence) == 2:
                                keypoints_angle.append(sequence_angle)
                                keypoints.append(sequence)
                             
                                # print(sequence_angle)
                                # print(sequence)
                                combined_arrays = np.concatenate((sequence_angle, sequence), axis=1)
                                final_keypoints.append(combined_arrays.tolist())
                                # print(final_keypoints)
                                sequence = []
                                sequence_angle = []

                            break


                # display image
                if webcam:
                    cv2.imshow("Detection", img_img)
                    key = cv2.waitKey(1)
                    if key == ord('c'):
                        break
                else:
                    img_ = img.copy()
                    img_ = cv2.resize(
                        img_, (960, 540), interpolation=cv2.INTER_LINEAR)
                    cv2.imshow("Detection", img_img)
                    cv2.waitKey(1)

                end_time = time.time()
                fps = 1 / (end_time - start_time)
                total_fps += fps
                frame_count += 1
                out.write(img_img)
            else:
                break

        cap.release()
        avg_fps = total_fps / frame_count
        print(f"Average FPS: {avg_fps:.3f}")
        with open("keypoints.json", "w") as f:
            json.dump(final_keypoints, f)


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--poseweights', nargs='+', type=str,
                        default='yolov7-w6-pose.pt', help='model path(s)')
    parser.add_argument('--source', type=str,
                        help='path to video or 0 for webcam')
    parser.add_argument('--device', type=str, default='cpu',
                        help='cpu/0,1,2,3(gpu)')
    parser.add_argument('--line_thickness', default=3, help='Please Input the Value of Line Thickness')

    opt = parser.parse_args()
    return opt


def main(opt):
    run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    strip_optimizer(opt.device, opt.poseweights)
    main(opt)
