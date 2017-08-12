from Tkinter import *
from _tkinter import *
import cv2
import numpy as np
import sqlite3
import os
from PIL import Image
import tkMessageBox


def detector():
        faceDetect = cv2.CascadeClassifier('face.xml')
        cam = cv2.VideoCapture(0)
        recognizer = cv2.createLBPHFaceRecognizer()

        recognizer.load('recognizer/tranningData.yml')
        # id =0
        font = cv2.cv.InitFont(cv2.cv.CV_FONT_HERSHEY_COMPLEX_SMALL, 1, 1, 0, 2)

        def getProfile(id):
            conn = sqlite3.connect('FaceBase.db')
            cmd = 'SELECT * FROM people WHERE ID=' + str(id)
            cursor = conn.execute(cmd)
            profile = None
            for row in cursor:
                profile = row
            conn.close()
            return profile

        while (cam.isOpened()):
            ret, img = cam.read()
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = faceDetect.detectMultiScale(gray, 1.3, 5)
            for (x, y, w, h) in faces:

                # cv2.rectangle(img, (x, y), (x + w, y + h), (0,0, 255), 2)
                id, conf = recognizer.predict(gray[y:y + h, x:x + w])
                cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
                profile = getProfile(id)
                if (profile != None):
                    cv2.cv.PutText(cv2.cv.fromarray(img), "Name: " + str(profile[1]), (x, y + h + 30), font, 255)
                    cv2.cv.PutText(cv2.cv.fromarray(img), "Age: " + str(profile[2]), (x, y + h + 60), font, 255)
                    cv2.cv.PutText(cv2.cv.fromarray(img), "Gender: " + str(profile[3]), (x, y + h + 90), font, 255)
                    cv2.cv.PutText(cv2.cv.fromarray(img), "Criminal Records: " + str(profile[4]), (x, y + h + 120),
                                   font, 255)

            cv2.imshow('face', img)
            if (cv2.waitKey(1) == ord('q')):
                break

        cam.release()
        cv2.destroyAllWindows()


def trainer():
        recognizer = cv2.createLBPHFaceRecognizer()
        path = 'dataSet'

        def getImagesWithid(path):
            imagepaths = [os.path.join(path, f) for f in os.listdir(path)]
            faces = []
            IDs = []
            for imagePath in imagepaths:
                faceImg = Image.open(imagePath).convert('L')
                faceNp = np.array(faceImg, 'uint8')
                ID = int(os.path.split(imagePath)[-1].split('.')[1])
                faces.append(faceNp)
                IDs.append(ID)
                cv2.imshow('trainer', faceNp)
                cv2.waitKey(10)
            return IDs, faces

        Ids, faces = getImagesWithid(path)
        recognizer.train(faces, np.array(Ids))
        recognizer.save('recognizer/tranningData.yml')
        cv2.destroyAllWindows()

def create_dataset():
        faceDetect = cv2.CascadeClassifier('face.xml')
        cam = cv2.VideoCapture(0)

        def insertORupdate(Id, Name):
            conn = sqlite3.connect('FaceBase.db')
            cmd = 'SELECT * FROM people WHERE ID =' + str(Id)
            cursor = conn.execute(cmd)
            isRecordExist = 0
            for row in cursor:
                isRecordExist = 1
            if (isRecordExist == 1):
                cmd = 'UPDATE people SET Name = ' + str(Name) + ' WHERE ID=' + str(Id)
            else:
                cmd = 'INSERT INTO people(ID,Name) values (' + str(Id) + ',' + str(Name) + ')'
            conn.execute(cmd)
            conn.commit()
            conn.close()


        #===============================data entry=================================
        id = raw_input('enter user id')
        name = raw_input('enter name')

        #================================================================================
        insertORupdate(id,name)

        sampleNum = 0

        while (cam.isOpened()):
            ret, img = cam.read()
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = faceDetect.detectMultiScale(gray, 1.3, 5)
            for (x, y, w, h) in faces:
                sampleNum = sampleNum + 1
                cv2.imwrite('dataSet/user.' + str(id) + '.' + str(sampleNum) + '.jpg', gray[y:y + h, x:x + w])
                cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
                cv2.waitKey(100)

            cv2.imshow('face', img)
            cv2.waitKey(1)
            if sampleNum > 20:
                break

        cam.release()
        cv2.destroyAllWindows()

def qExit():
    qExit = tkMessageBox.askyesno("Exit",'Do you want to exit?')
    if qExit > 0:
        root.destroy()
        return

#===================================object detection=====================================
def object_detection():
    MIN_MATCH_COUNT = 30

    detector = cv2.SIFT()
    FLANN_INDEX_KDTREE = 0
    flannParam = dict(algorithm=FLANN_INDEX_KDTREE, tree=5)
    flann = cv2.FlannBasedMatcher(flannParam, {})

    trainImg = cv2.imread('book.jpg', 0)
    trainKP, trainDecs = detector.detectAndCompute(trainImg, None)

    cam = cv2.VideoCapture(0)
    while (cam.isOpened()):
        ret, QueryImgBGR = cam.read()
        QueryImg = cv2.cvtColor(QueryImgBGR, cv2.COLOR_BGR2GRAY)
        queryKP, queryDesc = detector.detectAndCompute(QueryImg, None)
        matches = flann.knnMatch(queryDesc, trainDecs, k=2)

        goodMatch = []
        for m, n in matches:
            if (m.distance < 0.75 * n.distance):
                goodMatch.append(m)

        if (len(goodMatch) > MIN_MATCH_COUNT):
            tp = []
            qp = []
            for m in goodMatch:
                tp.append(trainKP[m.trainIdx].pt)
                qp.append(queryKP[m.queryIdx].pt)
            tp, qp = np.float32((tp, qp))
            H, status = cv2.findHomography(tp, qp, cv2.RANSAC, 3.0)
            h, w = trainImg.shape
            traingBorder = np.float32([[[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]])
            queryBorder = cv2.perspectiveTransform(traingBorder, H)
            cv2.polylines(QueryImgBGR, [np.int32(queryBorder)], True, (0, 255, 0), 5)
        else:
            print ('not enough matches')
        cv2.imshow('object recg', QueryImgBGR)
        cv2.waitKey(10)

#======================================================================================================

root = Tk()
root.geometry("1350x750+0+0")
root.title("Face recognization system")
root.configure(bg = 'black')
#==================================Top heading name======================================
Tops = Frame(root,width = 1350, height =80, bd = 12,relief = 'raise')
Tops.pack(side = TOP )
lblInfo = Label(Tops,font=('arial',60,'bold'),text = ' Face recognization system ',bd =10)
lblInfo.grid(row =0,column = 0)
#===============================frame===========================================
f1 = Frame(root,width = 900, height =650, bd = 8,relief = 'raise')     #main left side
f1.pack(side = LEFT )

f2 = Frame(root,width = 440, height =650, bd = 8,relief = 'raise')     #help right side
f2.pack(side = RIGHT )

f2a = Frame(f2,width = 640, height =450, bd = 12,relief = 'raise')    # right upper  help guide
f2a.pack(side = TOP )
f2b = Frame(f2,width = 440, height =550, bd = 16,relief = 'raise')    #right lower exit button
f2b.pack(side = BOTTOM )

f1a = Frame(f1,width = 400,height = 650,bd = 8,relief = 'raise')
f1a.pack(side = LEFT)
f1b = Frame(f1,width = 400,height = 650,bd = 8,relief = 'raise')
f1b.pack(side = LEFT)

#===================================Help desk lable=====================================================
def b_help():
    tkMessageBox.showinfo('Help', 'hello everyone,\n Thanks for downloading Face-Recognization application.\n'
                                  'To use it you have to go through 3 easy steps.\n'
                                  '1)Create Dataset \n'
                                  "enter unique ID each time and name in quotes. Ex.->'Prit' \n"
                                  '2)Train the dataSet \n'
                                  'Only click the option ,it will automatically load the dataSet\n'
                                  '3)Recognization \n'
                                  'just click option & it will start.\n'
                                  '\n \n'
                                  'For any queries visit to -> bhatepritesh@gmail.com')


lblReceipt = Label(f2a,font=('arial',12,'bold'),text = '     Object Detection System    ' ,bd =2,anchor='w').grid(row =0,column=0,sticky = W)
btnHelp = Button(f2a,padx=16,pady=1,bd=4,fg='black',font=('arial',15,'bold'),width=8,text='Guidance ',command=b_help).grid(row=1,column=1)
#==========================================================================================================

dataset = Checkbutton(f1a,text = 'Step 1 ',onvalue = 1,offvalue = 0,       #checkboxes..................
                font=('arial',18,'bold')).grid(row =0,sticky = W)
train_set = Checkbutton(f1a,text = 'Step 2 ',onvalue = 1,offvalue = 0,
                font=('arial',18,'bold')).grid(row =1,sticky = W)
Detector = Checkbutton(f1a,text = 'Step 3 ',onvalue = 1,offvalue = 0,
                font=('arial',18,'bold')).grid(row =2,sticky = W)



btnExit = Button(f2b,padx=16,pady=1,bd=4,fg='black',font=('arial',16,'bold'),width=4,text='Exit ',command=qExit).grid(row=0,column=0)
btnDataset = Button(f1b,padx=16,pady=1,bd=4,fg='black',font=('arial',15,'bold'),width=9,text='Create DataSet ',command=create_dataset).grid(row=0,column=0)
btnTrain = Button(f1b,padx=16,pady=1,bd=4,fg='black',font=('arial',15,'bold'),width=9,text='Train Data ',command=trainer).grid(row=1,column=0)
btnDetect = Button(f1b,padx=16,pady=1,bd=4,fg='black',font=('arial',15,'bold'),width=9,text='Recognization ',command=detector).grid(row=2,column=0)
btnObj_Detect = Button(f2a,padx=16,pady=1,bd=4,fg='black',font=('arial',14,'bold'),width=10,text='Object Detector ',command=object_detection).grid(row=2,column=1)



root.mainloop()