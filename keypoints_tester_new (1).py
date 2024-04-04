#Importing All the Required Libraries*
import cv2
import time
import torch
import argparse
import numpy as np
from utils.datasets import letterbox
from utils.torch_utils import select_device
from models.experimental import attempt_load
from utils.plots import output_to_keypoint, plot_skeleton_kpts, plot_one_box_kpt, colors
from utils.general import non_max_suppression_kpt, strip_optimizer
from torchvision import transforms
import tensorflow
from PIL import ImageFont, ImageDraw, Image
from google.colab.patches import cv2_imshow
#Creating an Empty Dictionary, to save the to shot count of each type of stroke
# The first element in the dictionary is the  key, which contains the shot type
# The second element in the dictionary is the value, which contains the shot count of each of the stroke
# for example ForeHandGroundStroke is played by the player 10 times
# while the Backhand Ground Stroke is played by the player 4 times.

import math
import mediapipe as mp
import matplotlib.pyplot as plt

import plotly.express as px
# Initializing mediapipe pose class.
mp_pose = mp.solutions.pose

# Setting up the Pose function.
posee = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.3, model_complexity=2)

# Initializing mediapipe drawing class, useful for annotation.
mp_drawing = mp.solutions.drawing_utils

output_image = np.zeros((100, 100, 3), dtype=np.uint8)
output_image[:, :] = [0, 0, 255]  # Set color to red
object_counter = {}

import pandas as pd
import plotly.graph_objects as go

_PRESENCE_THRESHOLD = 0.5
_VISIBILITY_THRESHOLD = 0.5


def plot_landmarks(
    landmark_list,
    connections=None,
):
    if not landmark_list:
        return
    plotted_landmarks = {}
    for idx, landmark in enumerate(landmark_list.landmark):
        if (
            landmark.HasField("visibility")
            and landmark.visibility < _VISIBILITY_THRESHOLD
        ) or (
            landmark.HasField("presence") and landmark.presence < _PRESENCE_THRESHOLD
        ):
            continue
        plotted_landmarks[idx] = (-landmark.z, landmark.x, -landmark.y)
    if connections:
        out_cn = []
        num_landmarks = len(landmark_list.landmark)
        # Draws the connections if the start and end landmarks are both visible.
        for connection in connections:
            start_idx = connection[0]
            end_idx = connection[1]
            if not (0 <= start_idx < num_landmarks and 0 <= end_idx < num_landmarks):
                raise ValueError(
                    f"Landmark index is out of range. Invalid connection "
                    f"from landmark #{start_idx} to landmark #{end_idx}."
                )
            if start_idx in plotted_landmarks and end_idx in plotted_landmarks:
                landmark_pair = [
                    plotted_landmarks[start_idx],
                    plotted_landmarks[end_idx],
                ]
                out_cn.append(
                    dict(
                        xs=[landmark_pair[0][0], landmark_pair[1][0]],
                        ys=[landmark_pair[0][1], landmark_pair[1][1]],
                        zs=[landmark_pair[0][2], landmark_pair[1][2]],
                    )
                )
        cn2 = {"xs": [], "ys": [], "zs": []}
        for pair in out_cn:
            for k in pair.keys():
                cn2[k].append(pair[k][0])
                cn2[k].append(pair[k][1])
                cn2[k].append(None)

    df = pd.DataFrame(plotted_landmarks).T.rename(columns={0: "z", 1: "x", 2: "y"})
    df["lm"] = df.index.map(lambda s: mp_pose.PoseLandmark(s).name).values
    fig = (
        px.scatter_3d(df, x="z", y="x", z="y", hover_name="lm")
        .update_traces(marker={"color": "red"})
        .update_layout(
            margin={"l": 0, "r": 0, "t": 0, "b": 0},
            scene={"camera": {"eye": {"x": 2.1, "y": 0, "z": 0}}},
        )
    )
    fig.add_traces(
        [
            go.Scatter3d(
                x=cn2["xs"],
                y=cn2["ys"],
                z=cn2["zs"],
                mode="lines",
                line={"color": "black", "width": 5},
                name="connections",
            )
        ]
    )

    return fig

def detectPose(image, posee, display=True):
    '''
    This function performs pose detection on an image.
    Args:
        image: The input image with a prominent person whose pose landmarks needs to be detected.
        pose: The pose setup function required to perform the pose detection.
        display: A boolean value that is if set to true the function displays the original input image, the resultant image,
                 and the pose landmarks in 3D plot and returns nothing.
    Returns:
        output_image: The input image with the detected pose landmarks drawn.
        landmarks: A list of detected landmarks converted into their original scale.
    '''

    # Create a copy of the input image.
    output_image = image.copy()

    # Convert the image from BGR into RGB format.
    imageRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Perform the Pose Detection.
    results = posee.process(imageRGB)

    # Retrieve the height and width of the input image.
    height, width, _ = image.shape

    # Initialize a list to store the detected landmarks.
    landmarks = []
    # Check if any landmarks are detected.
    if results.pose_landmarks:

        # Draw Pose landmarks on the output image.
        mp_drawing.draw_landmarks(image=output_image, landmark_list=results.pose_landmarks,
                                  connections=mp_pose.POSE_CONNECTIONS)

        # Iterate over the detected landmarks.
        for landmark in results.pose_landmarks.landmark:

            # Append the landmark into the list.
            landmarks.append((int(landmark.x * width), int(landmark.y * height),
                                  (landmark.z * width)))

    # Check if the original input image and the resultant image are specified to be displayed.
    if display:

        # Display the original input image and the resultant image.
        plt.figure(figsize=[22,22])
        plt.subplot(121);plt.imshow(image[:,:,::-1]);plt.title("Original Image");plt.axis('off');
        plt.subplot(122);plt.imshow(output_image[:,:,::-1]);plt.title("Output Image");plt.axis('off');

        # Also Plot the Pose landmarks in 3D.
        # mp_drawing.plot_landmarks(results.pose_world_landmarks, mp_pose.POSE_CONNECTIONS)
        # mp_drawing = mp.solutions.drawing_utils
        fig=plot_landmarks(
        results.pose_world_landmarks,  mp_pose.POSE_CONNECTIONS)
        # cv2_imshow(image)
        fig.write_image('ajaAns.png', engine='kaleido', width=800, height=600)

        # fig.savefig('graph_image.png')
        # Save the plot as an image file (e.g., PNG)
        plt.savefig('graph_image.png')
        # cv2_imshow(plt)
    # Otherwise
    else:

        # Return the output image and the found landmarks.
        return output_image, landmarks


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

#Creating a Function  by the name load_classes,using this function we will load the coco.names file and read each
#of the object name, in the coco.names file and this function will return all the object names in the coco.names
#file in the form of a list
def load_classes(path):
    # Loads *.names file at 'path'
    with open(path, 'r') as f:
        names = f.read().split('\n')
    return list(filter(None, names))  # filter removes empty strings (such as last line)

#This is the main function, here we are passing the yolov7 pose weights, by default we are setting the deivce as CPU
#setting the line thick ness of the skeleton lines as well

@torch.no_grad()
def run(poseweights='yolov7-w6-pose.pt', source='pose.mp4', device='cpu', names = 'utils/coco.names', line_thickness = 2):

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
        # ===================================

        # ==4.0=== Load trained pose-indentification model======
        tf_model = tensorflow.keras.models.load_model('mmodelfinal.h5')
        # ==================================================

        # == 5.0 == variable declaration===========
        sequence = []
        keypoints = []
        pose_name = ''
        posename_list = []
        actions = np.array(['Drop_Shot', 'Smash_Shot'])
        label_map = {label: num for num, label in enumerate(actions)}
        sequence_angle = []
        keypoints_angle = []
        j = 1
        seq = 2
        isImage=0

        # =============================================
        while cap.isOpened:

            print(f"Frame {frame_count} Processing")
            ret, frame = cap.read()
            temp_frame =  frame
            if ret:
                #First, we will do for the Player 2
                origimage = frame
                #Creating a Black Mask
                mask=np.zeros(frame.shape[:2] , dtype="uint8")
                #Setting the ROI for the Player 2
                roi = cv2.rectangle(mask, (297, 378), (997, 702),(255,255,255), -1)
                #Overlapping the Mask on the Original Image
                masked=cv2.bitwise_and(frame,frame,mask=mask)
                masked[np.where((masked==[0,0,0]).all(axis=2))]=[255,0,0]
                orig_image = masked

                # preprocess image
                image_ = cv2.cvtColor(origimage, cv2.COLOR_BGR2RGB)
                image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)
                image_ = letterbox(image_, (resize_width), stride=64, auto=True)[0]

                image = letterbox(image, (resize_width), stride=64, auto=True)[0]
                image__ = image_.copy()
                image___ = image.copy()
                image_ = transforms.ToTensor()(image_)
                image = transforms.ToTensor()(image)
                image_ = torch.tensor(np.array([image_.numpy()]))
                image = torch.tensor(np.array([image.numpy()]))
                image_ = image_.to(device)
                image_ = image_.float()
                image = image.to(device)
                image = image.float()
                start_time = time.time()


                ###Player1 Starts From Here
                origimage2 = frame
                #Creating a Black Mask
                mask2=np.zeros(frame.shape[:2] , dtype="uint8")
                #Setting the Region of Interest for the Player 1
                roi = cv2.rectangle(mask2, (350, 260), (926, 383),(255,255,255), -1)
                #Overlapping the Mask on the Original Image
                masked2=cv2.bitwise_and(frame,frame,mask=mask2)
                masked2[np.where((masked2==[0,0,0]).all(axis=2))]=[255,0,0]
                orig_image2 = masked2

                # preprocess image
                image_2 = cv2.cvtColor(origimage2, cv2.COLOR_BGR2RGB)
                image2 = cv2.cvtColor(orig_image2, cv2.COLOR_BGR2RGB)

                image_2 = letterbox(image_2, (resize_width), stride=64, auto=True)[0]

                image2 = letterbox(image2, (resize_width), stride=64, auto=True)[0]
                image__2 = image_2.copy()
                image___2 = image2.copy()
                image_2 = transforms.ToTensor()(image_2)
                image2 = transforms.ToTensor()(image2)
                image_2 = torch.tensor(np.array([image_2.numpy()]))
                image2 = torch.tensor(np.array([image2.numpy()]))
                image_2 = image_2.to(device)
                image_2 = image_2.float()
                image2 = image2.to(device)
                image2 = image2.float()
                start_time = time.time()

                with torch.no_grad():
                    output, _ = model(image)
                    output2, _ = model(image2)
                output_data = non_max_suppression_kpt(output, 0.25, 0.65, nc=model.yaml['nc'], nkpt=model.yaml['nkpt'], kpt_label=True)
                output = output_to_keypoint(output_data)
                output_data2 = non_max_suppression_kpt(output2, 0.25, 0.65, nc=model.yaml['nc'], nkpt=model.yaml['nkpt'], kpt_label=True)
                output2 = output_to_keypoint(output_data2)

                img_ = image_[0].permute(1, 2, 0) * 255
                img_ = img_.cpu().numpy().astype(np.uint8)

                img_img = cv2.cvtColor(img_, cv2.COLOR_RGB2BGR)

                img = image[0].permute(1, 2, 0) * 255
                img = img.cpu().numpy().astype(np.uint8)

                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                img_2 = image_2[0].permute(1, 2, 0) * 255
                img_2 = img_2.cpu().numpy().astype(np.uint8)

                img_img2 = cv2.cvtColor(img_2, cv2.COLOR_RGB2BGR)
                height, width,_ = img_img2.shape
                img2 = image2[0].permute(1, 2, 0) * 255
                img2 = img2.cpu().numpy().astype(np.uint8)

                img2 = cv2.cvtColor(img2, cv2.COLOR_RGB2BGR)
                #Plotting the Skeleton Keypoints on the Player 1
                for idx in range(output2.shape[0]):
                    plot_skeleton_kpts(img_img2, output2[idx, 7:].T, 3)
                    break

                gn = torch.tensor(img.shape)[[1, 0, 1, 0]]  # normalization gain whwh

                temp_frame = cv2.flip(temp_frame, 1)

                # Get the width and height of the frame
                frame_height, frame_width, _ = temp_frame.shape

                # Resize the frame while keeping the aspect ratio.
                temp_frame = cv2.resize(temp_frame, (int(frame_width * (640 / frame_height)), 640))

                # Convert the frame to a suitable depth format (e.g., CV_8U)
                temp_frame = cv2.convertScaleAbs(temp_frame)

                top_left = (200, 300)
                bottom_right = (880, 800)

                # cv2_imshow(frame)
                  # Extracting the region of interest from the frame
                temp_frame = temp_frame[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]

                for i, pose in enumerate(output_data):  # detections per image

                    if len(output_data):  #check if no pose
                        for c in pose[:, 5].unique(): # Print results
                            n = (pose[:, 5] == c).sum()  # detections per class
                            print("No of Objects in Current Frame : {}".format(n))

                        for det_index, (*xyxy, conf, cls) in enumerate(reversed(pose[:,:6])): #loop over poses for drawing on frame
                            c = int(cls)  # integer class
                            kpts = pose[det_index, 6:]
                            label = names[c]
                            #This function will create a bounding box and assign a label to the player 2
                            plot_one_box_kpt(xyxy, img_img2, label=label, color=colors(c, True),
                                        line_thickness=opt.line_thickness, kpts=kpts, steps=3,
                                        orig_shape=img.shape[:2])
                            # == 6.0 === preprocess model input data and pose prediction =======
                            #1.So now, each video frame was fed into the YOLOv7 Pose Estimation Model
                            #and predicted Key points Landmarks (X-coordinate, Y-coordinate and confidence) were extracted
                            # 2. and stacked together as a sequence of 30 frames.
                            #Here we will plot the skeleton keypoints on the Player 2
                            detectPose(temp_frame, posee, display=True)
                            isImage+=1
                            if j <= seq:
                                for idx in range(output.shape[0]):
                                    kpts = output[idx, 7:].T
                                    plot_skeleton_kpts(img_img2, kpts, 3)
                                    list_to_use = kpts.tolist()
                                    sequence_angle.append(getAngles(img_img, list_to_use))
                                    kpts = kpts[15:]
                                    seqq=[]
                                    for i in range(len(kpts)):
                                        if (i + 1) % 3 != 0:  # Check if the index is not a multiple of 3
                                            seqq.append(kpts[i])
                                    sequence.append(seqq)
                                    
                                    break
                            
                            # So if the length of the sequence is 30, then we will do the shot prediction,
                            # that whether it is a forehand ground stroke or the backhand ground stroke
                            if len(sequence) == 2:
                                # Doing the Shot Prediction over here
                                combined_arrays = np.concatenate((sequence_angle, sequence), axis=1)
                                final_keypoints=(combined_arrays.tolist())
                                result = tf_model.predict(np.expand_dims(final_keypoints, axis=0))
                                pose_name = actions[np.argmax(result)]
                                #So we will save all the sequence values which we have got
                                keypoints.append(sequence)
                                #We are also saving all the Shot Names in a list as well
                                posename_list.append(pose_name)
                                # print(sequence)
                                # print(keypoints)
                                # print(pose_name)
                                print(posename_list)
                                # So here we are saying if the shot name is not in the object counter dictionary
                                #then add the name
                                #If the shot name is already there in the object counter dictionary then just
                                #increment the counter
                                if pose_name not in object_counter:
                                  object_counter[pose_name] = 1
                                else:
                                  object_counter[pose_name] += 1
                                sequence = []
                                sequence_angle = []
                            #And when the value of j becomes equal to the value of the sequence
                            #which we have the defined as 30, then remove all the previous values of sequence
                            #and set it as an empty list and start processing on the next frames
                            break
                # =============================================================
                xstart = (fw//2)
                ystart = (fh-100)
                yend = (fh-50)
                # So After we have the shot name and the total shot count of each shot, we want to diplay them
                # in the output video so here we are just setting the UI where we want to display the Shot names and
                #the total shot count in the Output Video
                # = 7.0 == Draw prediction ==================================
                # print("super hi")


                if pose_name == "BackHand-GroundStroke":
                    cv2.line(img_img2, ((width - (width-50)),25), ((width - (width-200)),25), [85,45,255], 40)
                    cv2.putText(img_img2, "Shot Type", ((width - (width-50)),35), 0, 1, [225, 255, 255], thickness=2, lineType=cv2.LINE_AA)
                    cv2.line(img_img2, ((width - (width-50)),75), ((width - (width-500)),75), [85,45,255], 40)
                    cv2.putText(img_img2, pose_name, ((width - (width-50)),85), 0, 1, [225, 255, 255], thickness=2, lineType=cv2.LINE_AA)

                elif pose_name == "Forehand-GroundStroke":
                    cv2.line(img_img2, ((width - (width-50)),25), ((width - (width-200)),25), [85,45,255], 40)
                    cv2.putText(img_img2, "Shot Type", ((width - (width-50)),35), 0, 1, [225, 255, 255], thickness=2, lineType=cv2.LINE_AA)
                    cv2.line(img_img2, ((width - (width-50)),75), ((width - (width-500)),75), [85,45,255], 40)
                    cv2.putText(img_img2, pose_name, ((width - (width-50)),85), 0, 1, [225, 255, 255], thickness=2, lineType=cv2.LINE_AA)
                if isImage>=1:
                    # Assuming img_img2 is your video frame and pose_img is the output pose image
# Resize the pose image to a smaller size (e.g., 100x100) for better visibility
                  pose_img_resized = cv2.resize(output_image, (100, 100))

                  # Get the height and width of the resized pose image
                  pose_height, pose_width, _ = pose_img_resized.shape

                  # Define the position where you want to place the pose image (top-left corner in this example)
                  pos_x = 10
                  pos_y = 10

                  # Extract the region of interest from the video frame where you want to overlay the pose image
                  roi = img_img2[pos_y:pos_y + pose_height, pos_x:pos_x + pose_width]

                  # Create a binary mask for the pose image (convert to grayscale and threshold)
                  pose_mask = cv2.cvtColor(pose_img_resized, cv2.COLOR_BGR2GRAY)
                  _, pose_mask = cv2.threshold(pose_mask, 1, 255, cv2.THRESH_BINARY_INV)

                  # Invert the binary mask for the region where the pose image will be placed
                  roi_inv = cv2.bitwise_not(pose_mask)

                  # Blend the video frame and the resized pose image using the binary mask
                  img_img2_bg = cv2.bitwise_and(roi, roi, mask=roi_inv)
                  pose_img_fg = cv2.bitwise_and(pose_img_resized, pose_img_resized, mask=pose_mask)
                  result = cv2.addWeighted(img_img2_bg, 1, pose_img_fg, 1, 0)

                  # Replace the region of interest in the video frame with the blended result
                  img_img2[pos_y:pos_y + pose_height, pos_x:pos_x + pose_width] = result



                # ====================================================================
                for idx, (key, value) in enumerate(object_counter.items()):
                    cnt_str = str(key) + ":" +str(value)
                    cv2.line(img_img2, ((width-500),45), (width,45), [85,45,255], 40)

                    cv2.putText(img_img2, f'Total Shot Count', ((width-500),55), 0, 1, [225, 255, 255], thickness=2, lineType=cv2.LINE_AA)
                    cv2.line(img_img2, ((width-500), 85 + (idx*40)), (width, 85 + (idx*40)), [85, 45, 255], 30)
                    cv2.putText(img_img2, cnt_str, ((width-500), 95 + (idx*40)), 0, 1, [255, 255, 255], thickness = 2, lineType = cv2.LINE_AA)

                #Changes Made from Here

                small_image = cv2.imread('ajaAns.png')
                small_image=cv2.flip(small_image, 1)
                # Resize the small image to the desired size
                try:
                  small_image = cv2.resize(small_image, (200, 200))  # Adjust the size as needed
                except:
                  print('Error')

                # Define the position to append the small image
                x_offset_image = 400  # Adjust the offset as needed
                y_offset_image = 20  # Adjust the offset as needed

                # Append the small image to the frame
                img_img2[y_offset_image:y_offset_image + small_image.shape[0], x_offset_image:x_offset_image + small_image.shape[1]] = small_image
                out.write(img_img2)
                # Display the updated video frame
                # cv2_imshow(img_img2)
                # cv2_imshow(output_image)
                # display image
                if webcam:
                    # cv2.imshow("Detection", img)
                    key = cv2.waitKey(1)
                    if key == ord('c'):
                        break
                else:
                    img_ = img.copy()
                    img_ = cv2.resize(
                        img_, (960, 540), interpolation=cv2.INTER_LINEAR)
                    # cv2.imshow("Detection", img_img2)
                    cv2.waitKey(1)

                end_time = time.time()
                fps = 1 / (end_time - start_time)
                total_fps += fps
                frame_count += 1
                out.write(img_img2)
            else:
                break

        cap.release()
        avg_fps = total_fps / frame_count
        print(f"Average FPS: {avg_fps:.3f}")


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--poseweights', nargs='+', type=str,
                        default='yolov7-w6-pose.pt', help='model path(s)')
    parser.add_argument('--source', type=str,
                        help='path to video or 0 for webcam')
    parser.add_argument('--device', type=str, default='cpu',
                        help='cpu/0,1,2,3(gpu)')
    parser.add_argument('--line_thickness', default = 3, help = 'Please Input the Value of Line Thickness')

    opt = parser.parse_args()
    return opt

def main(opt):
    run(**vars(opt))

if __name__ == "__main__":
    opt = parse_opt()
    strip_optimizer(opt.device, opt.poseweights)
    main(opt)
