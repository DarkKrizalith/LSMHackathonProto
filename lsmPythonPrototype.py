# -*- coding: utf-8 -*-
"""
Created on Sun Jun 21 12:24:41 2020

@author: erick
"""

#-Último intento de unificación...-
# Sample: How to capture video from the camera (WebCam) and display it, with OpenCV-python.

import numpy as np
import cv2

import json
import requests

def callCustomVisionAPI(url, data, headers):
    print('intentando mandar llamada a la API de Custom Vision.')
    response = requests.post(url, data=data, headers=headers) 
    
    if response.status_code == 200:
        return json.loads(response.content.decode('utf-8'))
    else:
        return None

prediction_Key = '830c876d9e8a413f8892ec963bddb118'
api_url_base = 'https://westus2.api.cognitive.microsoft.com/customvision/v3.0/Prediction/a9db5e27-cb27-493c-8a6b-b31daf42ebb3/classify/iterations/IterationLSM/image'

headers = {'Content-Type':'application/octet-stream',
           'Prediction-Key':prediction_Key}
 
cap = cv2.VideoCapture(0)
 
while(True):
    ret, frame = cap.read()
    
    # Determine the type of <frame>
    # <frame> es de TIPO: <class 'numpy.ndarray'> ('numpy.ndarray')
    print(type(frame))
    print("np.ndarray: ", frame.shape)
    # RGB pixels 480 x 640 px (3 colores por pixel)
    print("np.darray.dtype: ", frame.dtype)
    #print("Pixel(0,0),  rgb[0]: ", frame[0][0])
    #print("Pixel(0,0), rgb[1]: ", frame[0][0][1])
    #print("Pixel(0,0), rgb[2]: ", frame[0][0][2])
    
    print("--Experimento bytearray...--")
    retval, buff = cv2.imencode(".jpg", frame)
    
    #print("retval: ", retval)
    #print("buff: ", buff)
    
    
    '''
    type(repBytes1024)
    Out[26]: bytes
    '''
    
    
    
    
    # --TODO: Schedule llamadas a la API de Custom Vision.--
    # llamadas en el mismo thread() - 'Main' Thread ::== lag en el Video.
    imgBytes = buff.tobytes()
    
    #-- Intentar llamada a la API de Custom Vision.
    print("Intentar conexión a Custom Vision service...")
    respuestaAPI = callCustomVisionAPI(api_url_base, imgBytes, headers)

    if respuestaAPI is not None:
        print("Here's your info: ")
        print(type(respuestaAPI))
        print(respuestaAPI.keys())
        print("Prediccion con PROBABILIDAD MÁS ALTA:")
        #print(type(respuestaAPI['predictions'][0]))
        print(respuestaAPI['predictions'][0]['probability'])
        print(respuestaAPI['predictions'][0]['tagName'])
    else:
        print('[!] Request Failed')
        
    
    # font = cv2.FONT_HERSHEY_SIMPLEX
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    # cv2.putText(image, 'texto a mostrar', (10,460), font, 4, (255, 255, 255), 2, cv2.LINE_AA)
    # --++
    stringToDisplay = str(respuestaAPI['predictions'][0]['probability']) + ', ' + str(respuestaAPI['predictions'][0]['tagName'])
    
    cv2.putText(frame, stringToDisplay, (10, 460), font, 0.5, (255,255,255), 2, cv2.LINE_AA)
    
    cv2.namedWindow('Capturando video...')
    
    
    cv2.imshow('Capturando video...',frame)
    # Presione la tecla 'q' para salir del loop while() de Grabación
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
 
cap.release()
cv2.destroyAllWindows()