import os, sys
sys.path.append(os.path.join(os.getcwd(), "GroundingDINO"))    
sys.path.append(os.path.join(os.getcwd(), "segmentation_anything"))

from utils.vision_util import GroundingDINO_Vision
# from utils.vision_util import YOLOWorld_Vision
from planners.planners import VLMPlanner

import cv2
import numpy as np
import matplotlib.cm as cm

import streamlit as st
from PIL import Image
from datetime import datetime

# --------- Utils ---------
def initialize_session_state():
    defaults = {
        "chat_history": [],
        "image_input": None,
        "processed_image": None,
        "image_processing": False,
        "env_desc": None,
        "prev_image_path": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

def get_color_for_object(index):
    """
    Matlab 스타일 색상 팔레트를 사용하여 객체의 색상을 가져옵니다.
    :param index: 객체의 인덱스 (0부터 시작)
    :return: RGB 색상 (tuple)
    """
    # Matlab 색상 팔레트(tab10)을 사용하여 색상 추출
    cmap = cm.get_cmap('tab10')
    color = cmap(index % 10)  # 색상 팔레트의 크기에 맞게 인덱스 순환
    return tuple(int(c * 255) for c in color[:3])  # 0~1 범위의 값을 0~255 범위로 변환

def draw_oriented_bbox(image, objects):
    """
    Draw oriented bounding boxes and object names on an image.
    
    :param image_path: Path to the input image.
    :param objects: List of objects, where each object is a tuple:
                    (name, logit, bbox, oriented_bbox)
                    - name: Object name (str)
                    - logit: Confidence score (float)
                    - bbox: (x_min, y_min, x_max, y_max)
                    - oriented_bbox: (cx, cy, w, h, angle)
    :return: Processed image with annotations (numpy array format).
    """
    # Load the image
    # image = Image.open(image_path)
    image = np.array(image)    

    # 이미지 크기 가져오기
    height, width = image.shape[:2]

    # 폰트 크기와 두께를 이미지 크기에 비례
    base_font_scale = max(width, height) / 1000  # 기본 비율 조정 (이미지 크기 대비)
    base_thickness = max(1, int(base_font_scale * 2))  # 최소 두께는 1로 설정

    for i, obj in enumerate(objects):
        name, logit, bbox, oriented_bbox = obj
        
        # Unpack bbox and oriented_bbox
        x_min, y_min, x_max, y_max = bbox
        x_min, y_min, x_max, y_max = int(x_min), int(y_min), int(x_max), int(y_max)
        (cx, cy), (w, h), angle = oriented_bbox[0]

        # Get color for the current object from the predefined palette
        color = get_color_for_object(i)

        # Draw regular bounding box
        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color=color, thickness=5*base_thickness)        
        
        # Add object name and logit
        text = f"{name}: {logit:.2f}"

        # 텍스트 크기 계산
        text_size = cv2.getTextSize(text, fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=base_font_scale, thickness=base_thickness)[0]
        text_x, text_y = x_min - 5, y_min - 30

        # 텍스트 배경 크기 설정 (조금 더 여유를 줘서 키움)
        text_w, text_h = text_size[0] + 10, text_size[1] + 10

        # 텍스트 배경 상자 그리기 (같은 색상 사용)
        cv2.rectangle(
            image, 
            (text_x, text_y - text_h), 
            (text_x + text_w, text_y + 10), 
            color,  # 박스 색상을 각 물체의 색상으로 설정
            -1  # -1로 채움
        )

        # 텍스트 그리기 (검은색)
        cv2.putText(
            image,
            text,
            org=(text_x + 5, text_y - 5),  # 텍스트가 박스 안에 잘 들어가게 설정
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=base_font_scale,
            color=(0, 0, 0),
            thickness=base_thickness,
            lineType=cv2.LINE_AA,
        )

        # Draw oriented bounding box
        rect = cv2.boxPoints(((cx, cy), (w, h), angle))
        rect = np.intp(rect)
        cv2.polylines(image, [rect], isClosed=True, color=(255, 255, 255), thickness=4*base_thickness)

    return image

# 이미지 처리 함수
def process_image(image_input, image):
    if image is None:
        return None    
    image_np = np.array(image)
    
    vision_base = GroundingDINO_Vision(device = 'cuda')
    obb_results = vision_base.obb_predict(image_input, text_prompt = "yellow.green.red.")    
    # vision_base = YOLOWorld_Vision(device = 'cuda')
    # obb_results = vision_base.obb_predict(image_input, text_prompt = ["yellow", "green", "red"])

    # obb_results = [['yellow', 0.8888310790061951, [2024.3189697265625, 800.58837890625, 2518.831298828125, 1355.8414306640625], [((2271.23876953125, 1085.838134765625), (522.2755126953125, 307.27642822265625), 62.840206146240234)]], ['red', 0.7884427905082703, [1533.3343505859375, 196.00796508789062, 2076.036376953125, 690.5823364257812], [((1795.741943359375, 439.44207763671875), (263.4681701660156, 536.7791137695312), 55.35905075073242)]], ['green', 0.7970165610313416, [718.8109741210938, 1089.64990234375, 1253.7325439453125, 1385.6641845703125], [((987.5362548828125, 1235.5908203125), (520.7658081054688, 258.0560607910156), 4.28381872177124)]]]    
    
    # Draw the bounding boxes on the image
    processed_image = draw_oriented_bbox(image_np, obb_results)    
    processed_image = Image.fromarray(processed_image)

    return processed_image, obb_results

# 대화 처리 함수 (예시: VLMPlanner 응답)
def chat_with_llm(history, message, env_desc):
    # env_desc가 None인 경우 친절한 메시지를 반환
    if env_desc is None:
        response_message = "The robot hasn't detected any objects yet. Please allow the robot to view objects before proceeding."
        reasoning_message = None  # Reasoning isn't available if there's no detection
    else:
        planner = VLMPlanner(
            planner_prompt_file="template",
            env_desc=env_desc
        )
        
        # Plan and reasoning generation
        plan, reasoning = planner.plan(message)
        
        # If no plan is available, inform the user
        if not plan:
            response_message = "No plan available. Please ensure the robot has detected objects and try again."
            reasoning_message = "No reasoning available."
        else:
            response_message = plan  # The actual plan is used as the response
            reasoning_message = reasoning  # Reasoning for the plan
    
    # Add the user message, bot response (plan), and reasoning to the chat history
    history.append({
        "user_message": message,
        "bot_message": response_message,
        "reasoning": reasoning_message
    })

    return history

def message(text, is_user=False):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    bot_icon = "🤖"    
    if is_user:        
        st.write(f'<div style="text-align: right; margin-bottom: 10px;">'
                 f'<div style="display: inline-block; background-color: #DCF8C6; padding: 10px; border-radius: 15px; max-width: 60%; word-wrap: break-word;">{text}</div><br>'
                 f'<span style="font-size: 0.7em; color: #888;">{timestamp}</span></div>', unsafe_allow_html=True)
    else:        
        st.write(f'<div style="text-align: left; margin-bottom: 10px;">'
                 f'<div style="font-weight: bold; color: #00796B; font-size: 1.1em;">{bot_icon} Bot</div>'
                 f'<div style="display: inline-block; background-color: #E0F7FA; padding: 10px; border-radius: 15px; max-width: 60%; word-wrap: break-word;">'
                 f'{text}</div><br>'
                 f'<span style="font-size: 0.7em; color: #888;">{timestamp}</span></div>', unsafe_allow_html=True)

# ----- Start Streamlit App -----
st.title("🤖📦 VLM-Based Pick & Place 🛠️")

initialize_session_state()

with st.sidebar:
    st.subheader("Upload an Image")
    image_input = st.file_uploader("Upload an Image", type=["png", "jpg", "jpeg"])
    
    if image_input:
        if image_input != st.session_state.image_input:
            st.session_state.image_input = image_input
            image = Image.open(image_input)
            st.image(image, caption="Uploaded Image", use_container_width=True)

            processed_image, obb_results = process_image(image_input, image)
            if processed_image:
                st.session_state.processed_image = processed_image
                st.session_state.env_desc = obb_results
                st.image(processed_image, caption="Processed Image", use_container_width=True)
        else:
            image = Image.open(st.session_state.image_input)
            st.image(image, caption="Uploaded Image", use_container_width=True)
            if st.session_state.processed_image:
                st.image(st.session_state.processed_image, caption="Processed Image", use_container_width=True)
    else:
        st.text("No image uploaded yet.")
        st.session_state.env_desc = None

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

for entry in st.session_state.chat_history:
    if entry["user_message"]:                
        message(entry["user_message"], is_user=True)
    if entry["bot_message"]:                    
        message(entry["bot_message"])
        if entry.get("reasoning"):            
            st.markdown(f'<div style="background-color: #F1F1F1; padding: 10px; border-radius: 10px;">'
                f'<em>Reasoning:</em> {entry["reasoning"]}</div>', unsafe_allow_html=True)

# 채팅 입력 받기
user_input = st.chat_input("Type your message...")

if user_input:
    st.session_state.chat_history = chat_with_llm(
        st.session_state.chat_history,
        user_input,
        st.session_state.env_desc
    )
    st.rerun()