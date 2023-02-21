from pain_detector import PainDetector
import cv2, os
import time
import argparse
import re
import random

 
parser = argparse.ArgumentParser(description='Testing')
parser.add_argument('-path', help='path to frames data')
parser.add_argument('-test_framerate', action='store_true', default=False, help='Runs frame rate test as well')
args = parser.parse_args()
# print('Device: ', pain_detector.device)

data_dir=args.path
cls1_dir=os.path.join(data_dir,'BL1/')
cls2_dir=os.path.join(data_dir,'PA1/')
cls3_dir=os.path.join(data_dir,'PA2/')
cls4_dir=os.path.join(data_dir,'PA3/')
cls5_dir=os.path.join(data_dir,'PA4/')

subjects=[]
for subs in os.listdir(os.path.join(data_dir,'BL1')):
    subjects.append(subs[:11])

for sub in subjects:
    sub='071309_w_21'
    pain_detector = PainDetector(image_size=160, checkpoint_path='checkpoints/50342566/50343918_3/model_epoch4.pt', num_outputs=40)
    for sample in os.listdir(cls1_dir):
        m = re.search(sub, sample)
        if m is not None:
            # pain_detector = PainDetector(image_size=160, checkpoint_path='checkpoints/50342566/50343918_3/model_epoch4.pt', num_outputs=40)
            ref_frame1 = cv2.imread(os.path.join(os.path.join(cls1_dir,sample),'00000.jpg'))
            ref_frame2 = cv2.imread(os.path.join(os.path.join(cls1_dir,sample),'00002.jpg'))
            ref_frame3 = cv2.imread(os.path.join(os.path.join(cls1_dir,sample),'00004.jpg'))
            ref_frame4 = cv2.imread(os.path.join(os.path.join(cls1_dir,sample),'00006.jpg'))
            pain_detector.add_references([ref_frame1, ref_frame2, ref_frame3, ref_frame4])
            pain_estimate=0
            for img_n in range(15,34):
                target_frame = cv2.imread(os.path.join(os.path.join(cls1_dir,sample),'0000'+str(img_n)+'.jpg'))
                tmp=pain_detector.predict_pain(target_frame)
                pain_estimate += tmp
                # pain_estimate.append(tmp)
                # print('BL1-img'+str(img_n)+': ', tmp)
            print('BL1: ', pain_estimate, sample)
    for sample in os.listdir(cls2_dir):
        m = re.search(sub, sample)
        if m is not None:
            # pain_detector = PainDetector(image_size=160, checkpoint_path='checkpoints/50342566/50343918_3/model_epoch4.pt', num_outputs=40)
            ref_frame1 = cv2.imread(os.path.join(os.path.join(cls2_dir,sample),'00000.jpg'))
            ref_frame2 = cv2.imread(os.path.join(os.path.join(cls2_dir,sample),'00002.jpg'))
            ref_frame3 = cv2.imread(os.path.join(os.path.join(cls2_dir,sample),'00004.jpg'))
            ref_frame4 = cv2.imread(os.path.join(os.path.join(cls2_dir,sample),'00006.jpg'))
            pain_detector.add_references([ref_frame1, ref_frame2, ref_frame3, ref_frame4])
            pain_estimate=0
            for img_n in range(15,34):
                target_frame = cv2.imread(os.path.join(os.path.join(cls2_dir,sample),'0000'+str(img_n)+'.jpg'))
                tmp=pain_detector.predict_pain(target_frame)
                pain_estimate += tmp
                # print('PA1-img'+str(img_n)+': ', tmp)
            print('PA1: ', pain_estimate,sample)
    for sample in os.listdir(cls3_dir):
        m = re.search(sub, sample)
        if m is not None:
            # pain_detector = PainDetector(image_size=160, checkpoint_path='checkpoints/50342566/50343918_3/model_epoch4.pt', num_outputs=40)
            ref_frame1 = cv2.imread(os.path.join(os.path.join(cls3_dir,sample),'00000.jpg'))
            ref_frame2 = cv2.imread(os.path.join(os.path.join(cls3_dir,sample),'00003.jpg'))
            ref_frame3 = cv2.imread(os.path.join(os.path.join(cls3_dir,sample),'00006.jpg'))
            pain_detector.add_references([ref_frame1, ref_frame2, ref_frame3])
            pain_estimate=0
            for img_n in range(15,34):
                target_frame = cv2.imread(os.path.join(os.path.join(cls3_dir,sample),'0000'+str(img_n)+'.jpg'))
                tmp=pain_detector.predict_pain(target_frame)
                pain_estimate += tmp
                # print('PA2-img'+str(img_n)+': ', tmp)
            print('PA2: ', pain_estimate,sample)
    for sample in os.listdir(cls4_dir):
        m = re.search(sub, sample)
        if m is not None:
            # pain_detector = PainDetector(image_size=160, checkpoint_path='checkpoints/50342566/50343918_3/model_epoch4.pt', num_outputs=40)
            ref_frame1 = cv2.imread(os.path.join(os.path.join(cls4_dir,sample),'00000.jpg'))
            ref_frame2 = cv2.imread(os.path.join(os.path.join(cls4_dir,sample),'00003.jpg'))
            ref_frame3 = cv2.imread(os.path.join(os.path.join(cls4_dir,sample),'00006.jpg'))
            pain_detector.add_references([ref_frame1, ref_frame2, ref_frame3])
            pain_estimate=0
            for img_n in range(15,34):
                target_frame = cv2.imread(os.path.join(os.path.join(cls4_dir,sample),'0000'+str(img_n)+'.jpg'))
                tmp=pain_detector.predict_pain(target_frame)
                pain_estimate += tmp
                # print('PA3-img'+str(img_n)+': ', tmp)
            print('PA3: ', pain_estimate,sample)
            
    for sample in os.listdir(cls5_dir):
        m = re.search(sub, sample)
        if m is not None:
            ref_frame1 = cv2.imread(os.path.join(os.path.join(cls5_dir,sample),'00000.jpg'))
            ref_frame2 = cv2.imread(os.path.join(os.path.join(cls5_dir,sample),'00003.jpg'))
            ref_frame3 = cv2.imread(os.path.join(os.path.join(cls5_dir,sample),'00006.jpg'))
            pain_estimate=0
            for img_n in range(15,34):
                target_frame = cv2.imread(os.path.join(os.path.join(cls5_dir,sample),'0000'+str(img_n)+'.jpg'))
                pain_detector.add_references([ref_frame1, ref_frame2, ref_frame3])
                tmp=pain_detector.predict_pain(target_frame)
                pain_estimate += tmp
                # print('PA4-img'+str(img_n)+': ', tmp)
            print('PA4: ', pain_estimate,sample)
# In this example the reference frames are identical, but in a real scenario, the idea is to use different
# reference frames from the same person. Ideally, the reference frames should have a neutral expression and should
# exhibit slight lighting and camera angle variations.
    

# if args.test_framerate:
#     num_of_frames = 30
#     print('Testing frame rate with {} frames'.format(num_of_frames))
#     start_time = time.time()
#     for _ in range(num_of_frames):
#         pain_detector.predict_pain(target_frame)
#     print('FPS: {}'.format(num_of_frames / (time.time() - start_time)))