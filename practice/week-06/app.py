import streamlit as st

import matplotlib.pyplot as plt
import cv2
import numpy as np
# 캐스케이드 파일 지정해서 검출기 생성하기 --- (*1)
cascade_file = "haarcascade_frontalface_alt.xml"
cascade = cv2.CascadeClassifier(cascade_file)



st.title("Streamlit Face App")
img_file_buffer = st.camera_input("Take a picture")
if img_file_buffer is not None:

    bytes_data = img_file_buffer.getvalue()

    # 이미지를 읽어 들이고 그레이스케일로 변환하기 --- (*2)
    img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)

    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # 얼굴 인식하기 --- (*3)
    face_list = cascade.detectMultiScale(img_gray)


    # 인식한 부분 표시하기 --- (*4)
    x,y,w,h = face_list[0]

    red = (0, 0, 255)
    cv2.rectangle(img, (x, y), (x+w, y+h), red, thickness=10)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    st.image(img, caption="와우!")