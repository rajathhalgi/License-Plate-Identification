import cv2 as cv
import pytesseract
import numpy as np

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe'

cascade = cv.CascadeClassifier("haarcascade_russian.xml")

states = {"AN":"Andaman and Nicobar","AP":"Andhra Pradesh", "AR":"Arunachal Pradesh","AS":"Assam", "BR":"Bihar",
          "CH":"Chandigarh", "CG":"Chhattisgarh","DD":"Dadra and Nagar Haveli and Daman and Diu", "DL":"Delhi",
          "GA":"Goa", "GJ":"Gujrat", "HR":"Haryana", "HP":"Himachal Pradesh","JK":"Jammu and Kashmir", "JH":"Jharkhand",
          "KA":"Karnataka", "KL":"Kerla","LA":"Ladakh","LD":"Lakshadweep", "MP":"Madhya Pradesh","MH":"Maharashtra",
          "MN":"Manipur", "ML":"Meghalaya","MZ":"Mizoram", "NL":"Nagaland", "OD":"Odisha", "PY":"Puducherry", "PB":"Punjab",
          "RJ":"Rajasthan", "SK":"Sikkim", "TN":"Tamil Nadu", "TS":"Telangana", "TR":"Tripura", "UP":"Uttar Pradesh",
          "UK":"Uttarakhand", "WB":"West Bengal"}

def extract_num(img_name):
    global read
    img = cv.imread(img_name)
    #img = cv.resize(img, (620,480))
    gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    n_plate = cascade.detectMultiScale(gray,1.1,3)

    for (x,y,w,h) in n_plate:
      # cropping of the number plate
      a,b = (int(0.02*img.shape[0]), int(0.02*img.shape[1]))
      plate = img[y+a:y+h-a, x+b:x+w-b, :]

    # image processing
      kernel = np.ones((1,1), np.uint8)
      plate = cv.dilate(plate, kernel, iterations=1)
      plate = cv.erode(plate, kernel, iterations=1)
      plate_gray = cv.cvtColor(plate, cv.COLOR_BGR2GRAY)
      (thresh, plate) = cv.threshold(plate_gray, 127, 255, cv.THRESH_BINARY)

      read = pytesseract.image_to_string(plate)
      print(read)
      read = ''.join(e for e in read if e.isalnum())
      stat = read[0:2]
      try:
          print('Car belongs to ',states[stat])
      except:
          print('Car does not belong to India!')
      print(read)
      cv.rectangle(img,(x,y),(x+w, y+h), (50,50,255),4)
      cv.rectangle(img, (x, y-40),(x+w, y),(50,50,255),-1)
      cv.putText(img,read,(x,y-10),cv.FONT_HERSHEY_SIMPLEX,0.7,(255,255,255),2)
      cv.putText(img,f"The car belongs to {states[stat]} ", (150,150), cv.FONT_ITALIC, 1.0, (0,0,255), 2)
      cv.imshow('Plate', plate)

    cv.imshow('Result', img)
    #cv.imwrite('Result.jpg', img)
    cv.waitKey(0)
    cv.destroyAllWindows()

extract_num('photos/test2.jpg')
