import time
import serial
import requests
import numpy
from io import BytesIO
from pprint import pprint

import cv2
import math
import requests
from requests.auth import HTTPBasicAuth
import time


total = 0
ser = serial.Serial("/dev/ttyACM0", 9600)

# API endpoint
api_url = "###"
def PaintBbox(img):
    
    start = time.time()

    class_name = ['RASPBERRY PICO','HOLE','BOOTSEL','OSCILLATOR','USB','CHIPSET']
    class_color = [[0,0,255],[0,127,255],[255,0,255],[0,255,0],[255,0,0],[173,119,137]]

    class_score = [1,4,1,1,1,1]
    class_score_img = [0,0,0,0,0,0]
    count_all = 0
    count_correct = 0

    #NEW
    th = 3
    defect = False
    inpico = False
    class_center = {}
    HOLE_x = []
    HOLE_y = []
    HOLEs = []
    pico = []


    _, img_encoded = cv2.imencode(".jpg", img)

    # Prepare the image for sending
    img_bytes = BytesIO(img_encoded.tobytes())

    response = {}

    try:
        response = requests.post(
            url=api_url,
            auth=HTTPBasicAuth("kdt2024_1-4", "###"),
            headers={"Content-Type": "image/jpeg"},
            data=img_bytes,
        )
        # response = requests.post(api_url, files=files)
        if response.status_code == 200:
            pprint(response.json())
            print("Image sent successfully")
            # return response.json()
        else:
            print(f"Failed to send image. Status code: {response.status_code}")
    except requests.exceptions.RequestException as e:
        print(f"Error sending request: {e}")

    #print(response.json())

    for info in response.json()['objects']:

        #print(info['box'])

        #print('for')
        
        text = info['class']
        idx = class_name.index(text)
        co = class_color[idx]
        class_score_img[idx] += 1

        start_point = (info['box'][0],info['box'][1]) # 박스 시작 좌표 (x, y)
        end_point = (info['box'][2],info['box'][3]) # 박스 끝 좌표 (x, y)
        color = (co[0],co[1],co[2]) # BGR 색상 (초록색)
        thickness = 2 # 박스 선의 두께
        
        # 박스 그리기
        cv2.rectangle(img, start_point, end_point, color, thickness)

        position = (info['box'][0]+3 , info['box'][1]-3 ) # 텍스트 시작 위치 (x, y)
        #position = (50,50)
        
        font = cv2.FONT_HERSHEY_SIMPLEX # 글꼴 설정
        font_scale = 0.5 # 글자 크기
        color = (co[0],co[1],co[2]) # BGR 색상 (초록색)
        thickness = 1 # 글자 두께

        cv2.putText(img, text, position, font, font_scale, color, thickness, cv2.LINE_AA)

        #NEW
        if text == 'HOLE':
            x = round((info['box'][0]+info['box'][2])/2,3)
            y = round((info['box'][1]+info['box'][3])/2,3)
            HOLE_x.append(x)
            HOLE_y.append(y)
            HOLEs.append([x,y])
        else:
            if text=='RASPBERRY PICO':
                pico = info['box']
                inpico = True
                #print(pico)
            else:
                class_center[text] = [(info['box'][0]+info['box'][2])/2,(info['box'][1]+info['box'][3])/2]
            
        
    #NEW
    if len(HOLEs) == 4:
        min_idx = HOLE_x.index(min(HOLE_x))
        HOLE1 = HOLEs[min_idx]

        dist = [0,0,0,0]
        for i in range(4):
            if min_idx == i:
                continue
            dist[i] = (HOLE1[0]-HOLEs[i][0])**2 + (HOLE1[1]-HOLEs[i][1])**2
        
        max_idx = dist.index(max(dist))
        HOLE4 = HOLEs[max_idx]
        dist[max_idx] = 0
        max_idx = dist.index(max(dist))
        HOLE3 = HOLEs[max_idx]
        dist[max_idx] = 0
        max_idx = dist.index(max(dist))
        HOLE2 = HOLEs[max_idx]

        if abs(HOLE1[0]-HOLE3[0]) < 0.01 or abs(HOLE2[0]-HOLE4[0]) < 0.01:
            grad_1to3 = round((HOLE1[1]-HOLE3[1])/(HOLE1[0]-HOLE3[0]+0.5),3)
            grad_2to4 = round((HOLE2[1]-HOLE4[1])/(HOLE2[0]-HOLE4[0]+0.5),3)
        else:
            grad_1to3 = round((HOLE1[1]-HOLE3[1])/(HOLE1[0]-HOLE3[0]),3)
            grad_2to4 = round((HOLE2[1]-HOLE4[1])/(HOLE2[0]-HOLE4[0]),3)
        
        grad_1to3_loss = abs(grad_1to3-grad_2to4)

        if abs(HOLE1[0]-HOLE2[0]) < 0.01 or abs(HOLE3[0]-HOLE4[0]) < 0.01:
            grad_1to2 = round((HOLE1[1]-HOLE2[1])/(HOLE1[0]-HOLE2[0] + 50),3)
            grad_3to4 = round((HOLE3[1]-HOLE4[1])/(HOLE3[0]-HOLE4[0] + 50),3)
        else:
            grad_1to2 = round((HOLE1[1]-HOLE2[1])/(HOLE1[0]-HOLE2[0]),3)
            grad_3to4 = round((HOLE3[1]-HOLE4[1])/(HOLE3[0]-HOLE4[0]),3)
        


        grad_1to2_loss = abs(grad_1to2-grad_3to4)


        print(f"grad_1to3_loss: {grad_1to3_loss}, grad_1to2_loss: {grad_1to2_loss}")
        print(grad_1to3,grad_2to4)
        print(grad_1to2,grad_3to4)
        #print(HOLE1)
        #print(HOLE2)
        #print(HOLE3)
        #print(HOLE4)

        if abs(HOLE1[0] - HOLE3[0]) > 125.0:
            th = 50

        if grad_1to3_loss > 3 or grad_1to2_loss > th:
            defect = True

    if inpico == True:
        for key in class_center:
            x = class_center[key][0]
            y = class_center[key][1]
            if (x>=pico[0] and y>= pico[1]) and (x <= pico[2] and y<= pico[3]):
                continue
            else:
                defect = True


    cv2.imshow("",img)
    cv2.waitKey(1)
 

    if class_score != class_score_img:
        defect = True
    
    global total
    total += 1

    print(f"throughput: {time.time()-start} sec, total: {total}")

    if defect == True:
        input("불량품을 제거하고 엔터를 입력하세요")
    else:
        print("정상품입니다.")

    ser.write(b"1")



def get_img():
    """Get Image From USB Camera

    Returns:
        numpy.array: Image numpy array
    """

    cam = cv2.VideoCapture(0)

    if not cam.isOpened():
        print("Camera Error")
        exit(-1)

    ret, img = cam.read()
    cam.release()

    return img

def crop_img(img, size_dict):
    x = size_dict["x"]
    y = size_dict["y"]
    w = size_dict["width"]
    h = size_dict["height"]
    img = img[y : y + h, x : x + w]
    return img


def inference_reqeust(img: numpy.array, api_rul: str, i):
    """_summary_

    Args:
        img (numpy.array): Image numpy array
        api_rul (str): API URL. Inference Endpoint
    """
    _, img_encoded = cv2.imencode(".jpg", img)

    # Prepare the image for sending
    img_bytes = BytesIO(img_encoded.tobytes())

    # Send the image to the API
    files = {"file": ("image.jpg", img_bytes, "image/jpeg")}

    print(files)

    # try:
    #     response = requests.post(
    #         url="https://suite-endpoint-api-apne2.superb-ai.com/endpoints/34fae42c-2e7e-4b29-911f-e9611b1d1618/inference",
    #         auth=HTTPBasicAuth("kdt2024_1-4", "oW5FXZy2Th2VANqPQgJs14jYDhQPQRPe7tqIMlA8"),
    #         headers={"Content-Type": "image/jpeg"},
    #         data=img_bytes,
    #     )
    #     # response = requests.post(api_url, files=files)
    #     if response.status_code == 200:
    #         pprint(response.json())
    #         print("Image sent successfully")
    #         return response.json()
    #     else:
    #         print(f"Failed to send image. Status code: {response.status_code}")
    # except requests.exceptions.RequestException as e:
    #     print(f"Error sending request: {e}")


# i = 300 # with cutton
i = 500 # without cutton

while 1:

    data = ser.read()
    print(data)
    if data == b"0":
        img = get_img()
        # crop_info = None
        crop_info = {"x": 150, "y": 150, "width": 300, "height": 300}

        if crop_info is not None:
            img = crop_img(img, crop_info)

        # cv2.imshow("", img)
        #cv2.imwrite(f"train/train_{i}.jpg", img)
        # cv2.waitKey(1)
        # ser.write(b"1")

        PaintBbox(img)
    else:
        pass
    
    i += 1

