##裏と表の判別##マイナンバーカードに反応するようにする##
# -*- coding: UTF-8 -*-
#http://qiita.com/Algebra_nobu/items/a488fdf8c41277432ff3
from skimage import io
import cv2
import os
import sys
import dlib
import numpy as np
#import cam
sys.path.append(os.getcwd())
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches

#学習(train)

options = dlib.simple_object_detector_training_options()
# 学習時に画像の左右の反転を行うか
options.add_left_right_image_flips = False
# コストパラメータ
options.C = 3    
options.detection_window_size = 150 * 150    
options.epsilon = 0.01
# スレッド数
options.num_threads = 4
# 学習中に詳細を表示するか
options.be_verbose = True

    #表面の検出
#def front():
train_xml_1 = "image_front/front.xml"
svm_file_1 = "image_front/front_detector.svm"
dlib.train_simple_object_detector(train_xml_1, svm_file_1, options)
detector1 = dlib.simple_object_detector("image_front/front_detector.svm")
    
    #裏面の検出
#def back():
train_xml_2 = "image_back/back.xml"
svm_file_2 = "image_back/back_detector.svm"
dlib.train_simple_object_detector(train_xml_2, svm_file_2, options)
detector2 = dlib.simple_object_detector("image_back/back_detector.svm")
    
    #斜めの検出
#def naname():
train_xml_3 = "image_naname/naname.xml"
svm_file_3 = "image_naname/naname_detector.svm"
dlib.train_simple_object_detector(train_xml_3, svm_file_3, options)
detector3 = dlib.simple_object_detector("image_naname/naname_detector.svm")


#動画をキャプチャ-.
cap = cv2.VideoCapture(0)

while(cap.isOpened()):
    #キャプチャーで動画のフレームを読み込む.
    ret, frame = cap.read()

    #フレームが出来なかった場合（動画が終了した場合など）はループを抜ける.
    if ret == False:
        break
            
    # 読み込んだフレーム(カードの表裏)を矩形認識処理に通す.
    
    dets1 = detector1(frame)#表   
    dets2 = detector2(frame)#裏
    dets3 = detector3(frame)#斜め
    

    # 返された矩形の数分、フレームに矩形を書き込む
    for det1 in dets1:
        cv2.rectangle(frame, (det1.left(), det1.top()), (det1.right(), det1.bottom()),(128, 255, 0), 8)
        #cv2.drawMarker(frame, (det1.left(), det1.top()),(128, 255, 0), markerType=cv2.MARKER_STAR,markerSize=10)
        text='omote'
        font = cv2.FONT_HERSHEY_PLAIN
        cv2.putText(frame,text,(det1.left(),det1.top()-10),font, 1, (128, 255, 0), 2, cv2.LINE_AA)
        
    
    
    for det2 in dets2:
        cv2.rectangle(frame, (det2.left(), det2.top()), (det2.right(), det2.bottom()),(255, 0, 0), 8)
        #cv2.drawMarker(frame, (det2.left(), det2.top()),(255, 0, 0), markerType=cv2.MARKER_STAR,markerSize=10)
        text='ura'
        font = cv2.FONT_HERSHEY_PLAIN
        cv2.putText(frame,text,(det2.left(),det2.top()-10),font, 1, (255, 0, 0), 2, cv2.LINE_AA)
    
    for det3 in dets3:
        cv2.rectangle(frame, (det3.left(), det3.top()), (det3.right(), det3.bottom()),(255, 255, 0), 8)
        #cv2.drawMarker(frame, (det3.left(), det3.top()),(255,255, 0), markerType=cv2.MARKER_STAR,markerSize=10)
        text='naname'
        font = cv2.FONT_HERSHEY_PLAIN
        cv2.putText(frame,text,(det3.left(),det3.top()-10),font, 1, (255, 255, 0), 2, cv2.LINE_AA)
         
    #def card_face():     
    #顔の認識
    f_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt2.xml')
    
    #顔認識の実行
    facerect = f_cascade.detectMultiScale(frame, scaleFactor=1.2, minNeighbors=10, minSize=(10, 10))
    
    #検出した顔を囲む矩形の作成
    for rect in facerect:
        cv2.rectangle(frame, tuple(rect[0:2]),tuple(rect[0:2] + rect[2:4]), (255, 255, 255), thickness=2) 
        #cv2.drawMarker(frame, tuple(rect[0:2]), tuple(rect[0:2] + rect[2:4]),(255,255, 255), markerType=cv2.MARKER_STAR,markerSize=10)
        text = 'face'
        font = cv2.FONT_HERSHEY_PLAIN
        cv2.putText(frame,'face',(rect[0],rect[1]-10),font, 2, (255, 255, 255), 2, cv2.LINE_AA)
    

    # 出力ウィンドウにフレームを表示   
    cv2.imshow("frame", frame)

    #終了
    k = cv2.waitKey(0)
    if k == 27:
        break

#キャプチャーの解放        
cap.release()
#出力ウィンドウの破棄
cv2.destroyAllWindows()
