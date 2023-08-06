# -Program Name : Intelligen Teaching Web Based Application - virtualMouseControl.py
# -Description : Use for generate implement virtual mouse control functions. The main usage is using Mediapipe and self-trained model to recognise and gesture, then use PyautoGUI for control the mouse
# -First Written on: 25 Feb 2023
# -Editted on: 1 May 2023

import cv2
import mediapipe as mp
import numpy as np
import sys
import pyautogui as pg
import os
import datetime
from keras.models import load_model

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_drawing = mp.solutions.drawing_utils
screenWidth, screenHeight = pg.size()   #get the size of screen
frameR=110  #limit the area of drag
pg.FAILSAFE =False
tidId=[4,8,12,16,20] #finger tip id
clk=1
clk2=1
dragLeft = False
SStime = 1
Snumber = 0
startAction=False
currentTime=datetime.datetime.now().strftime("%Y%m%d_%H%M")
label=['screenshot','none',"open"]
model = load_model('gesture_model_virtual_mouse.h5')


def landmark_to_vector(landmarks):
    vector = []
    for landmark in landmarks:
        vector.append(landmark.x)
        vector.append(landmark.y)
        vector.append(landmark.z)
    return vector


def gen():   
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_BUFFERSIZE,1)
    while cap.isOpened():
        status, frame = cap.read()
        if status:
             
            #mirror the view
            frame = cv2.flip(frame,1)
            h,w,c = frame.shape
            
            outcomes = hands.process(frame)
            cv2.rectangle(frame,(frameR,frameR),(w-frameR,h-frameR),(0,0,255),2)#draw box to limit the area of action
            
            if outcomes.multi_hand_landmarks:
                landmarkList =[]
                hand= outcomes.multi_handedness[0].classification[0].label
                      
                for hand_landmarks in outcomes.multi_hand_landmarks:
                    vector = landmark_to_vector(hand_landmarks.landmark)
                    #print(vector)
                    #sys.stdout.flush()
                    new_vector = np.array(vector).reshape(1,63)
                    prediction = model.predict(new_vector)
                    index = np.argmax(prediction)
                    
                    for id, landmarks in enumerate(hand_landmarks.landmark):
                        #get the location of hand
                        cx,cy = int(landmarks.x*w),int(landmarks.y*h)
                    
                        landmarkList.append([id,cx,cy])
                    
                    #----------count finger raise---------------
                    fingers=[]
                    if hand == "Right":
                        if landmarkList[tidId[0]][1] < landmarkList[tidId[0]-1][1]: #thumb
                            fingers.append(1)
                        else:
                            fingers.append(0)
                    else:
                        if landmarkList[tidId[0]][1] > landmarkList[tidId[0]-1][1]: #thumb
                            fingers.append(1)
                        else:
                            fingers.append(0)
                        
                    for id in range(1,5): #finger 2-5
                        if landmarkList[tidId[id]][2] < landmarkList[tidId[id]-3][2]:
                            fingers.append(1)
                        else:
                            fingers.append(0)
                        #print(fingers)
                        #sys.stdout.flush()
                        
                    #allow access to all functions of virtual mouse by opem palm
                    if label[index] == "open" and fingers==[1,1,1,1,1]:
                        global startAction
                        startAction = True
                    
                    if startAction == True :
                        #draw landmarks on video
                        mp_drawing.draw_landmarks(frame,hand_landmarks,mp_hands.HAND_CONNECTIONS)    
                        global dragLeft, drawMode
                        
                        #calculate the size of hand
                        size = ((landmarkList[8][1]- landmarkList[0][1])**2 + (landmarkList[8][2] - landmarkList[0][2])**2)**0.5
                        
                        #print("Hand Size: "+ str(size))
                        #sys.stdout.flush()
                        
                        #----------do actions---------------
                        #action 1 +2 move + drag
                        if fingers[1] == 1 and fingers[2] == 0 and fingers[3] == 0 and fingers[4] == 0:
                            #draw circle on point 12(finger tip of index finger)
                            cv2.circle(frame,(landmarkList[8][1],landmarkList[8][2]),10,(0,0,255),cv2.FILLED)
                            
                            # Define a list to store the last 5 mouse positions
                            mouse_positions = []

                            # In the loop where you generate the mouse position
                            x = np.interp(landmarkList[8][1], (frameR, w-frameR), (0, screenWidth))
                            y = np.interp(landmarkList[8][2], (frameR, w-frameR), (0, screenWidth))

                            # Add the new position to the list
                            mouse_positions.append((x, y))

                            # If the list is longer than 5, remove the oldest position
                            if len(mouse_positions) > 7:
                                mouse_positions.pop(0)

                            # Take the average of the last 5 positions
                            x_avg = sum(pos[0] for pos in mouse_positions) / len(mouse_positions)
                            y_avg = sum(pos[1] for pos in mouse_positions) / len(mouse_positions)

                            # Move the mouse to the averaged position
                            pg.moveTo(x_avg, y_avg)
                            if fingers[0] == 1:
                                if dragLeft == False:
                                    pg.mouseDown(button='left')
                                    dragLeft = True
                            else:
                                if dragLeft == True:
                                    pg.mouseUp(button='left')
                                    dragLeft = False
                                    
                            
                        
                        #action 3 left click
                        elif fingers == [0,1,1,0,0]:
                            global clk
                            #calculate the distance of index finger and middle finger
                            
                            #draw circle on point 12(finger tip of index finger)
                            cv2.circle(frame,(landmarkList[8][1],landmarkList[8][2]),10,(255,0,0),cv2.FILLED)
                            cv2.line(frame,(landmarkList[8][1],landmarkList[8][2]),(landmarkList[12][1],landmarkList[12][2]),(255,0,0),8)
                            cv2.circle(frame,(landmarkList[12][1],landmarkList[12][2]),10,(255,0,0),cv2.FILLED)
                            
                            length = abs(landmarkList[8][1] - landmarkList[12][1])

                            #print(length/size)
                            #sys.stdout.flush()
                            if(length/size) <= 0.15 and clk>0:
                                # print("Close!")
                                # sys.stdout.flush()
                                pg.click()
                                clk=-1
                            else:
                                clk=1
                            
                                
                        #action 4 right click
                        elif fingers == [1,1,1,0,0]:
                            #calculate the distance of index finger and middle finger
                            #draw circles and line
                            global clk2
                            cv2.circle(frame,(landmarkList[8][1],landmarkList[8][2]),10,(255,0,0),cv2.FILLED)
                            cv2.line(frame,(landmarkList[8][1],landmarkList[8][2]),(landmarkList[12][1],landmarkList[12][2]),(255,0,0),8)
                            cv2.circle(frame,(landmarkList[12][1],landmarkList[12][2]),10,(255,0,0),cv2.FILLED)
                            
                            length = abs(landmarkList[8][1] - landmarkList[12][1])

                            #print(length/size)
                            #sys.stdout.flush()
                            if(length/size) <= 0.15 and clk2>0:
                                # print("Close!")
                                # sys.stdout.flush()
                                pg.rightClick()
                                clk2=-1
                            else:
                                clk2=1
                        
                        #action 5 and 6 scroll up and down
                        elif fingers == [0,1,1,1,1] and (landmarkList[4][1] > landmarkList[5][1]):
                            folded=0
                            fingers_folded=0
                            # print("in scoroll fnction!")
                            # sys.stdout.flush()
                            for id in range(1,5): #finger 2-5
                                if landmarkList[tidId[id]][2] > landmarkList[tidId[id]-1][2]: #4 finger folded
                                    fingers_folded += 1
                            
                            if fingers_folded == 4:
                                folded = 1
                            elif fingers_folded == 0:
                                folded = -1
                            else:
                                folded = 0
                            
                            if folded== 1:
                                pg.scroll(-20)
                            elif folded ==-1:
                                pg.scroll(20)
                        
                        
                        #action 7: screenshot
                        elif label[index] == "screenshot" and fingers[2] ==1 and fingers[3] ==1 and fingers[4] ==1:
                            folded2=False
                            global SStime, Snumber
                            
                            for id in range(2,4): #finger 2-5
                                if landmarkList[tidId[id]][2] > landmarkList[tidId[id]-1][2]: #3 finger folded
                                    
                                    folded2 = True
                                    
                                else:
                                    folded2 = False
                                    SStime=0
                            # print(folded2)
                            # sys.stdout.flush()        
                            
                            if SStime == 0 and folded2==True:
                                screenshot=pg.screenshot()
                                filepath =os.path.dirname(os.path.abspath(__file__))
                                #print(filepath+"\screenshots\screenshot" + currentTime +".jpg")
                                screenshot.save(filepath+"\screenshots\screenshot" + currentTime +" (" +str(Snumber) +")"+".jpg")
                                #sys.stdout.flush()
                                cv2.putText(frame, 'captured',(20,40),cv2.FONT_HERSHEY_DUPLEX,1,(0,0,255),2)
                                SStime = 1
                                Snumber+=1
                            
                            if SStime == 1:
                                cv2.putText(frame, 'captured',(20,40),cv2.FONT_HERSHEY_DUPLEX,1,(0,0,255),2)
                                  
                
            else:
                startAction = False
                
            
            frame = cv2.resize(frame, (640, 480))               
            status, jpeg = cv2.imencode('.jpg', frame)
            frame = jpeg.tobytes()
            yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n') 
    cap.release()  
    
 