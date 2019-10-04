from flask import Flask, request, abort, jsonify, send_from_directory

import face_recognition
import numpy as np
import os
import os.path as path
from skimage import io
import json
from flask_cors import CORS

import cv2
import operator
from scipy import spatial
from skimage.feature import hog
from operator import itemgetter
from math import sqrt
import requests
from urllib.request import urlopen
import pyimgur

app = Flask(__name__)
CORS(app)

xrange = range

rs={}
#<Face recognition>################################################################################
def compareFace(idCardLink, selfieLink):
    idCardImg = face_recognition.load_image_file(idCardLink)
    face_locations = face_recognition.face_locations(idCardImg)

    rs["faceIdLocations"] = face_locations

    if(len(face_locations))>0:
        for face_location in face_locations:
            top, right, bottom, left = face_location
            encodings = face_recognition.face_encodings(idCardImg, face_locations)

    selfieImg = face_recognition.load_image_file(selfieLink)
    face_locations=face_recognition.face_locations(selfieImg)

    rs["faceSelfieLocations"] = face_locations

    if(len(face_locations))>0:
        for face_location in face_locations:
            top, right, bottom, left = face_location
            verify_encodings = face_recognition.face_encodings(selfieImg, face_locations)
    
    results = face_recognition.compare_faces(np.array(encodings), np.array(verify_encodings))

    rs["faceVerified"] = str(results[0])

#</Face recognition>###############################################################################

#<Document Template Check>#########################################################################
#   <Scanner>
#       Scan to get square
def angle_cos(p0, p1, p2):
    d1, d2 = (p0-p1).astype('float'), (p2-p1).astype('float')
    return abs( np.dot(d1, d2) / np.sqrt( np.dot(d1, d1)*np.dot(d2, d2) ) )

def find_squares(img):
    cos_thresh=0.07
    img = cv2.GaussianBlur(img, (1, 1), cv2.BORDER_DEFAULT)
    for gray in cv2.split(img):
        for thrs in xrange(0, 255, 26):
            if thrs == 0:
                bin = cv2.Canny(gray, 0, 50, apertureSize=5)
                bin = cv2.dilate(bin, None)
            else:
                 _retval, bin = cv2.threshold(gray, thrs, 255, cv2.THRESH_BINARY)
            contours, _hierarchy = cv2.findContours(bin, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[-2:]
            for cnt in contours:
                cnt_len = cv2.arcLength(cnt, True)
                cnt = cv2.approxPolyDP(cnt, 0.02*cnt_len, True)
                    # boundingRect=cv2.boundingRect(cnt)
                [intX, intY, intWidth, intHeight] = cv2.boundingRect(cnt)
                if len(cnt) == 4 and cv2.contourArea(cnt) > 1000 and cv2.isContourConvex(cnt):
                    cnt = cnt.reshape(-1, 2)
                    max_cos = np.max([angle_cos( cnt[i], cnt[(i+1) % 4], cnt[(i+2) % 4] ) for i in xrange(4)])
                    if max_cos < cos_thresh:
                            # cos_thresh=max_cos
                        if (np.float(intHeight/intWidth)>=np.float(5400/8600-0.06))&(np.float(intHeight/intWidth)<=np.float(5400/8600+0.06)):
                            squares = []
                            squares.append(cnt)
                            cos_thresh=max_cos
    sq=[]
    list_pos=[]
    pos=[]
    for i in range (0,4):
        sq.append(sqrt(squares[0][i][0]^2+squares[0][i][1]^2))

    values=squares[0]
    dictionary=dict(zip(sq, values))
    keys=sorted(dictionary)

    for key in keys:
        list_pos.append(dictionary[key])
    pos.append(list_pos[0])

    list1=sorted(list_pos[1:], key=itemgetter(0))
    pos.append(list1[0])

    list2=sorted(list1[1:], key=itemgetter(1),reverse=True)
    pos=pos+list2[0:2]

    return squares, pos

def Scanner(idCardLink):
    #pts2 = np.float32([[0,0],[0,8600*2],[5400*2,8600*2],[5400*2,0]])
    pts2 = np.float32([[0,0],[0,860],[540,860],[540,0]])

    img = cv2.imread(idCardLink)
    image= img.copy()
    squares,list_pos = find_squares(img)
    pts1=np.float32(list_pos)

    M = cv2.getPerspectiveTransform(pts1,pts2)
    #dst = cv2.warpPerspective(image,M,(5400*2,8600*2))
    dst = cv2.warpPerspective(image,M,(540,860))
    cv2.drawContours( img, squares, -1, (0, 255, 0), 3 )

    return(dst)
#   </Scanner>

#   <Scan Emblem>
# module level variables ##########################################################################
MIN_CONTOUR_AREA = 200

RESIZED_IMAGE_WIDTH = 20
RESIZED_IMAGE_HEIGHT = 30

###################################################################################################
class ContourWithData():
    # member variables ############################################################################
    npaContour = None           # contour
    boundingRect = None         # bounding rect for contour
    intRectX = 0                # bounding rect top left corner x location
    intRectY = 0                # bounding rect top left corner y location
    intRectWidth = 0            # bounding rect width
    intRectHeight = 0           # bounding rect height
    fltArea = 0.0               # area of contour
    ###############################################################################################

    def calculateRectTopLeftPointAndWidthAndHeight(self):            # calculate bounding rect info
        [intX, intY, intWidth, intHeight] = self.boundingRect
        self.intRectX = intX
        self.intRectY = intY
        self.intRectWidth = intWidth
        self.intRectHeight = intHeight
        

    def EmblemCheckIfContourIsValid(self):
        if self.intRectHeight>=80:
            # print(self.intRectHeight,self.intRectWidth)                          
            # this is oversimplified, for a production grade program
            if (self.intRectHeight/self.intRectWidth<=99/97+0.05) &(self.intRectHeight/self.intRectWidth>=99/97-0.05):
                if self.fltArea > MIN_CONTOUR_AREA:       
                    # much better validity checking would be necessary
                    return True
        return False
    ###############################################################################################
def emblemScan(imgScanned):
    allContoursWithData = []                  # declare empty lists,
    validContoursWithData_1 = []              # we will fill these shortly
    validContoursWithData = []

    imgTestingNumbers = imgScanned                          # read in testing numbers image
    if imgTestingNumbers is None:                           # if image was not read successfully
        print( "error: image not read from file \n\n")      # print error message to std out
        os.system("pause")                                  # pause so user can see error message
        return                                              # and exit function (which exits program)
    # end if
    imgTestingNumbers=cv2.resize(imgTestingNumbers,(856,540))
    imgTestingNumbersCopy=imgTestingNumbers.copy()

    imgGray = cv2.cvtColor(imgTestingNumbers, cv2.COLOR_BGR2GRAY)       # get grayscale image
    imgBlurred = cv2.GaussianBlur(imgGray, (5,5), 0)                    # blur
    
    #Image threshoding binary

    ret,imgThresh = cv2.threshold(imgBlurred,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    # make a copy of the thresh image, this in necessary b/c findContours modifies the image    
    imgThreshCopy = imgThresh.copy()

    npaContours, npaHierarchy = cv2.findContours(imgThresh,                      # input image, make sure to use a copy since the function will modify this image in the course of finding contours
                                                 cv2.RETR_EXTERNAL,              # retrieve the outermost contours only
                                                 cv2.CHAIN_APPROX_SIMPLE)[-2:]   # compress horizontal, vertical, and diagonal segments and leave only their end points

    for npaContour in npaContours:                             # for each contour
        contourWithData = ContourWithData()                                             # instantiate a contour with data object
        contourWithData.npaContour = npaContour                                         # assign contour to contour with data
        contourWithData.boundingRect = cv2.boundingRect(contourWithData.npaContour)     # get the bounding rect
        contourWithData.calculateRectTopLeftPointAndWidthAndHeight()                    # get bounding rect info
        contourWithData.fltArea = cv2.contourArea(contourWithData.npaContour)           # calculate the contour area
        allContoursWithData.append(contourWithData)                                     # add contour with data object to list of all contours with data
    # end for

    for contourWithData in allContoursWithData:                    # for all contours
        if contourWithData.EmblemCheckIfContourIsValid():             # check if valid
            validContoursWithData.append(contourWithData)          # if so, append to valid contour list    


    validContoursWithData.sort(key = operator.attrgetter("intRectX"))         # sort contours from left to right

    for contourWithData in validContoursWithData:            # for each contour
                                                             # draw a green rect around the current char
        imgROI = imgTestingNumbersCopy[contourWithData.intRectY : contourWithData.intRectY + contourWithData.intRectHeight,     # crop char out of threshold image
                           contourWithData.intRectX : contourWithData.intRectX + contourWithData.intRectWidth]

    rs['emblem'] = []
    rs['emblem'].append ({
                            'X': contourWithData.intRectX ,
                            'Y': contourWithData.intRectY,
                            'Width':contourWithData.intRectWidth,
                            'Height':contourWithData.intRectHeight
                        })

    return imgROI, imgTestingNumbers, contourWithData

def emblemVerify(test_image):
    emblemStandardUrl = "img/standard/emblemStandard.jpg"

    image = cv2.imread(emblemStandardUrl)
    dim=(130,130)
    test=cv2.resize(test_image, dim)
    image=cv2.resize(image,dim)

    fd= hog(image, orientations=16, pixels_per_cell=(8, 8),
                    cells_per_block=(1, 1), visualize=False, multichannel=True)

    fd_test = hog(test, orientations=16, pixels_per_cell=(8, 8),
                    cells_per_block=(1, 1), visualize=False, multichannel=True)

    similarity = 1 - spatial.distance.cosine(fd_test, fd)

    return (similarity>=0.8)
#   </Scanner Emblem>
#</Document Template Check>########################################################################

# <Upload Image to Imgur>
def UploadToImgur(PATH):
    CLIENT_ID = "6d939fc69c5d794"
    im = pyimgur.Imgur(CLIENT_ID)
    return im.upload_image(PATH)
# </Upload Image to Imgur>

@app.route("/")
def main():
    return "Home"

'''
@app.route("/idchecker", methods=["POST"])
def idchecker():
    upload_dir = "img/data"
    if request.method == 'POST':
        idCard = request.files['idCard']
        selfie = request.files['selfie']
        idCardLink = path.join(upload_dir, idCard.filename)
        selfieLink = path.join(upload_dir, selfie.filename)
        idCard.save(idCardLink)
        selfie.save(selfieLink)

    compareFace(idCardLink, selfieLink)

    imgScanned = Scanner(idCardLink)
    emblem = emblemScan(imgScanned)
    emblemVerified = emblemVerify(emblem)
    
    rs["emblem_verified"] = str(emblemVerified)
    
    return json.dumps(rs)
'''

@app.route("/idchecker", methods=["GET"])
def idchecker():
    idCardUrl = request.args.get('idCardPath')
    selfieUrl = request.args.get('selfiePath')

    rs['idCardUrl'] = idCardUrl
    rs['selfieUrl'] = selfieUrl

    idCardName = idCardUrl.split('/')[-1]
    selfieName = selfieUrl.split('/')[-1]

    with open(idCardName, 'wb') as handle:
        response = requests.get(idCardUrl, stream=True).content
        handle.write(response)

    with open(selfieName, 'wb') as handle:
        response = requests.get(selfieUrl, stream=True).content
        handle.write(response)

    imgScanned = Scanner(idCardName)
    emblem, idImgScanned, contourWithData = emblemScan(imgScanned)

    cv2.imwrite(idCardName, idImgScanned)
    rsScanIdImgUrl = UploadToImgur(idCardName)
    
    compareFace(idCardName, selfieName)

    emblemVerified = emblemVerify(emblem)

    if(emblemVerified):
        color = (0, 255, 0)
    else:
        color = (35, 35, 254)

    idImgRsEmblem = idImgScanned.copy()
    idImgRsFace = idImgScanned.copy()
    
    #Draw rectangle to result emblem
    cv2.rectangle(idImgRsEmblem,        
                      (contourWithData.intRectX, contourWithData.intRectY), 
                      (contourWithData.intRectX + contourWithData.intRectWidth, contourWithData.intRectY + contourWithData.intRectHeight),
                      color,             
                      3)
    cv2.imwrite('emblemVerify.jpg', idImgRsEmblem)
    rsDrawIdEmblemUrl = UploadToImgur('emblemVerify.jpg')

    if(rs["faceVerified"] == "True"):
        color = (0, 255, 0)
    else:
        color = (35, 35, 254)
    
    #Draw rectangle to result face
    cv2.rectangle(idImgRsFace,
                        (rs["faceIdLocations"][0][3], rs["faceIdLocations"][0][2]),
                        (rs["faceIdLocations"][0][1], rs["faceIdLocations"][0][0]),
                        color,
                        3)
    cv2.imwrite('faceIdVerify.jpg', idImgRsFace)
    rsDrawIdFaceUrl = UploadToImgur('faceIdVerify.jpg')
    
    #Draw rectangle to selfie img
    selfieImg = cv2.imread(selfieName)
    cv2.rectangle(selfieImg,
                        (rs["faceSelfieLocations"][0][3], rs["faceSelfieLocations"][0][2]),
                        (rs["faceSelfieLocations"][0][1], rs["faceSelfieLocations"][0][0]),
                        color,
                        3)
    cv2.imwrite('faceVerify.jpg', selfieImg)
    rsDrawSelfieUrl = UploadToImgur('faceVerify.jpg')
    
    rs["emblem_verified"] = str(emblemVerified)

    rs['rsScanIdImgUrl'] = rsScanIdImgUrl.link
    rs['rsDrawIdFaceUrl']= rsDrawIdFaceUrl.link
    rs['rsDrawIdEmblemUrl'] =rsDrawIdEmblemUrl.link
    rs['rsDrawSelfieUrl'] = rsDrawSelfieUrl.link

    rs_json = json.dumps(rs)
    r = requests.post("http://henryangminh.pythonanywhere.com/record", json=rs_json)
    
    return json.dumps(rs)

if __name__ == "__main__":
    app.run()
