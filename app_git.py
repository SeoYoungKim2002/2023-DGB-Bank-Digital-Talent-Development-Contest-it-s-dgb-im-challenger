import streamlit as st
import pandas as pd

# Google Play Scraper 및 App Store Scraper를 사용한 앱 정보 스크래핑
from google_play_scraper import search as gps_search, app as gps_app
from app_store_scraper import AppStore 

# 스트림릿 파일 연결 유틸리티 (사용자 정의 모듈일 수 있음)
from st_files_connection import FilesConnection
import os
#from google.cloud import storage  # Google Cloud Storage 작업을 위한 라이브러리
from io import StringIO
import asyncio

# Google OAuth 인증을 위한 라이브러리
from streamlit_oauth import OAuth2Component


#firebase 로그인(login)
from firebase_admin import firestore


# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

from oauthlib.oauth2 import WebApplicationClient  # OAuth2 클라이언트를 위한 라이브러리
import requests


import numpy as np

import importlib.util
import sys

#GCP 연결
#from google.cloud import storage



#사용자 로그인 DB연결
import firebase_admin
from firebase_admin import credentials, firestore, storage


#준지도
import torch
from transformers import GPT2Model, GPT2LMHeadModel, PreTrainedTokenizerFast
import faiss
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, SimpleRNN, GRU, Dense

#클러스터
from gensim.models import Doc2Vec
from gensim.models.doc2vec import TaggedDocument
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from konlpy.tag import Okt
from sentence_transformers import SentenceTransformer, util
from datetime import datetime
from numpy import triu
from scipy.linalg import get_blas_funcs


#게시판 DB연결 (커뮤니티)
import sqlite3
import tempfile
from firebase_admin import storage #사용자가 올린 파일 저장하기 위한 Firebase storage 연결 라이브러리
import firebase_admin
from firebase_admin import credentials, firestore, storage
from firebase_admin import credentials, storage  # storage 모듈 추가


#SNA
import networkx as nx
from konlpy.tag import Okt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.font_manager as fm
from collections import defaultdict

#LDA부분
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from konlpy.tag import Okt
from gensim import corpora, models
from gensim.models import CoherenceModel
import pyLDAvis.gensim_models as gensimvis
import pyLDAvis
from PIL import Image
import numpy as np
import streamlit.components.v1 as components



#DGB Chatbot

import openai


# 페이지 설정
st.set_page_config(page_title="분석 시스템", layout="wide")

# 세션 상태 관리를 위한 초기화 함수
#  st.session_state는 Streamlit 서버의 메모리 내에서 관리
if 'posts' not in st.session_state:
    st.session_state['posts'] = []

 
 ########################################################3
 #메뉴 사이드바 부분
 
    
# 각 메뉴에 대한 내용은 화면에 표시되지 않으므로 원하는 메뉴를 선택하여 내용을 표시할 수 있습니다.
# dgb_image=st.sidebar.image('https://brassy-dolphin-f22.notion.site/image/https%3A%2F%2Fprod-files-secure.s3.us-west-2.amazonaws.com%2Fa60d437f-ec9f-4f91-a164-6f05ff0d8852%2F1f4539ac-bfe6-4b36-bf9c-6b6ad66e3300%2Fimage-removebg-preview_(14).png?table=block&id=ec2e0844-33f1-42d4-afdd-e42b67ab2ca9&spaceId=a60d437f-ec9f-4f91-a164-6f05ff0d8852&width=710&userId=&cache=v2', width='400')   
# 이미지 파일 불러오기
image_path = 'https://brassy-dolphin-f22.notion.site/image/https%3A%2F%2Fprod-files-secure.s3.us-west-2.amazonaws.com%2Fa60d437f-ec9f-4f91-a164-6f05ff0d8852%2F1f4539ac-bfe6-4b36-bf9c-6b6ad66e3300%2Fimage-removebg-preview_(14).png?table=block&id=ec2e0844-33f1-42d4-afdd-e42b67ab2ca9&spaceId=a60d437f-ec9f-4f91-a164-6f05ff0d8852&width=710&userId=&cache=v2'

# 사이드바에 이미지 삽입
st.sidebar.image(image_path, caption='DGB CAS로 고객분석을 쉽고 빠르게!', use_column_width=True)
st.sidebar.markdown("---")
menu = st.sidebar.selectbox("MENU SELECTION", ["🏠홈", "🔐로그인", "📝사용방법", "📁데이터 수집&전처리", "📊클러스터링", "📈소셜 네트워크 분석", "📉토픽 모델링", "💻기회점수", "🙎🏻DGB CAS 커뮤니티","⌨️DGB Chatbot"])
st.sidebar.markdown("---")





# 각 메뉴에 대한 설명 텍스트
menu_text="""
## 💙DGB CAS MENU💙
"""

home_text = """
## 🏠 홈

"""

login_text = """
## 🔐 로그인

"""

usage_text = """
## 📝 사용방법

"""

data_collection_text = """
## 📁 데이터 수집&전처리

"""

clustering_text = """
## 📊 클러스터링

"""

social_network_analysis_text = """
## 📈 소셜 네트워크 분석

"""

topic_modeling_text = """
## 📉 토픽 모델링

"""

opportunity_score_text = """
## 💻 기회점수

"""

bulletin_board_text = """
## 🙎🏻‍♀️ DGB CAS 커뮤니티

"""

DGB_Chatbot_text = """
## ⌨️ DGB Chatbot

"""

# 각 메뉴 설명을 사이드바에 표시
st.sidebar.markdown(menu_text, unsafe_allow_html=True)
st.sidebar.markdown(home_text, unsafe_allow_html=True)
st.sidebar.markdown(login_text, unsafe_allow_html=True)
st.sidebar.markdown(usage_text, unsafe_allow_html=True)
st.sidebar.markdown(data_collection_text, unsafe_allow_html=True)
st.sidebar.markdown(clustering_text, unsafe_allow_html=True)
st.sidebar.markdown(social_network_analysis_text, unsafe_allow_html=True)
st.sidebar.markdown(topic_modeling_text, unsafe_allow_html=True)
st.sidebar.markdown(opportunity_score_text, unsafe_allow_html=True)
st.sidebar.markdown(bulletin_board_text, unsafe_allow_html=True)
st.sidebar.markdown(DGB_Chatbot_text, unsafe_allow_html=True)


###############################################


# 전역 변수로 데이터프레임 초기화
df = None

if menu == '🏠홈':
    
    
    
    col1, col2 = st.columns([1, 3])  # 첫 번째 열은 1, 두 번째 열은 3의 비율로 설정
    
    html="""
        <div style='
        background-color: #f5fffa;
        color: white;
        padding: 20px;
        text-align: center;
        '>
            <img src='https://brassy-dolphin-f22.notion.site/image/https%3A%2F%2Fprod-files-secure.s3.us-west-2.amazonaws.com%2Fa60d437f-ec9f-4f91-a164-6f05ff0d8852%2F1f4539ac-bfe6-4b36-bf9c-6b6ad66e3300%2Fimage-removebg-preview_(14).png?table=block&id=ec2e0844-33f1-42d4-afdd-e42b67ab2ca9&spaceId=a60d437f-ec9f-4f91-a164-6f05ff0d8852&width=710&userId=&cache=v2' width='400'>
        </div>
    """

    st.markdown(html, unsafe_allow_html=True)
        
    st.image('https://brassy-dolphin-f22.notion.site/image/https%3A%2F%2Fprod-files-secure.s3.us-west-2.amazonaws.com%2Fa60d437f-ec9f-4f91-a164-6f05ff0d8852%2F86784a8f-09d7-4c59-9e9a-b03473fb5333%2F75efd937-105b-4dc3-bd81-2b37465caf7c.png?table=block&id=13127fed-6039-47fa-8d35-9808d0c0678b&spaceId=a60d437f-ec9f-4f91-a164-6f05ff0d8852&width=880&userId=&cache=v2',use_column_width=True)
    st.image('https://brassy-dolphin-f22.notion.site/image/https%3A%2F%2Fprod-files-secure.s3.us-west-2.amazonaws.com%2Fa60d437f-ec9f-4f91-a164-6f05ff0d8852%2F245d7569-a3fa-4a48-89dd-4af7bc0dd39e%2FUntitled.png?table=block&id=206d5f21-03a1-4a35-b3e1-864266b898e4&spaceId=a60d437f-ec9f-4f91-a164-6f05ff0d8852&width=1280&userId=&cache=v2', use_column_width=True)
    st.image('https://brassy-dolphin-f22.notion.site/image/https%3A%2F%2Fprod-files-secure.s3.us-west-2.amazonaws.com%2Fa60d437f-ec9f-4f91-a164-6f05ff0d8852%2Fc0e6e1b2-feeb-4e9a-aed0-6bc182b1bcaf%2FUntitled.png?table=block&id=1b9f7cc1-035d-4b7d-b262-eaa687877a08&spaceId=a60d437f-ec9f-4f91-a164-6f05ff0d8852&width=1520&userId=&cache=v2', use_column_width=True)
    st.image('https://brassy-dolphin-f22.notion.site/image/https%3A%2F%2Fprod-files-secure.s3.us-west-2.amazonaws.com%2Fa60d437f-ec9f-4f91-a164-6f05ff0d8852%2F7597632e-242d-4d9b-a6c2-66cdecc2353b%2FUntitled.png?table=block&id=3526b4a7-954d-4720-8860-9c8706511cb6&spaceId=a60d437f-ec9f-4f91-a164-6f05ff0d8852&width=1280&userId=&cache=v2', use_column_width=True)
    st.image('https://brassy-dolphin-f22.notion.site/image/https%3A%2F%2Fprod-files-secure.s3.us-west-2.amazonaws.com%2Fa60d437f-ec9f-4f91-a164-6f05ff0d8852%2F5e17fc70-2d5c-4459-bd86-73469eec471a%2FUntitled.png?table=block&id=8fa66bd6-87d5-47f0-bfb8-bc2d2921e6ba&spaceId=a60d437f-ec9f-4f91-a164-6f05ff0d8852&width=1280&userId=&cache=v2',use_column_width=True)
 
 
elif menu == '🔐로그인':
    st.image('https://brassy-dolphin-f22.notion.site/image/https%3A%2F%2Fprod-files-secure.s3.us-west-2.amazonaws.com%2Fa60d437f-ec9f-4f91-a164-6f05ff0d8852%2F72036241-44cc-4888-b714-f241a98fb706%2FUntitled.png?table=block&id=eb7729ef-4e30-400b-aafc-ad3e0ecba85b&spaceId=a60d437f-ec9f-4f91-a164-6f05ff0d8852&width=870&userId=&cache=v2', use_column_width=True)

    # st.title('Google 로그인')
    # st.markdown("<h1 ='color: blue;'>Google 로그인</h1>", unsafe_allow_html=True)

    
    # Set environment variables
    AUTHORIZE_URL = "https://accounts.google.com/o/oauth2/auth"
    TOKEN_URL = "https://oauth2.googleapis.com/token"
    REFRESH_TOKEN_URL = "https://oauth2.googleapis.com/token"
    REVOKE_TOKEN_URL = "https://oauth2.googleapis.com/revoke"
    USERINFO_URL = "https://www.googleapis.com/oauth2/v3/userinfo"   # 사용자 정보를 가져오기 위한 업데이트된 URL
    CLIENT_ID = "client_id"
    CLIENT_SECRET = "client_secret"
    REDIRECT_URI = "http://localhost:8501"
    SCOPE = "openid https://www.googleapis.com/auth/userinfo.email https://www.googleapis.com/auth/userinfo.profile"


    # Create OAuth2Component instance
    oauth2 = OAuth2Component(CLIENT_ID, CLIENT_SECRET, AUTHORIZE_URL, TOKEN_URL, REFRESH_TOKEN_URL, REVOKE_TOKEN_URL)


    # Firebase Admin SDK 초기화
    if not firebase_admin._apps:
        cred = credentials.Certificate("C:\My\Competition\DGB\Final\streamlit\Google_oauth_Fire_key\dgb-user-login-database-firebase-adminsdk-lhbhr-146d46b243.json") # Firebase Admin SDK json 파일 경로
        firebase_admin.initialize_app(cred, {
        'projectId': 'dgb-user-login-database',
        })
        firebase_admin.initialize_app(cred)

    # Firestore DB 인스턴스 생성
    db = firebase_admin.firestore.client()

    # 사용자 정보 Firestore에 저장 또는 업데이트
    def save_user_info(user_info):
        # Firestore의 users 컬렉션에 사용자 정보 저장
        user_ref = db.collection(u'users').document(user_info['email'])
        user_ref.set({
            u'email': user_info['email'],
            u'name': user_info['name'],
            u'picture': user_info['picture']
        })

    # Google 로그인 성공 후 사용자 정보 가져오기 및 저장
    if 'token' in st.session_state:
        # 사용자 정보 가져오기
        access_token = st.session_state['token']['access_token']
        headers = {'Authorization': f'Bearer {access_token}'}
        response = requests.get(USERINFO_URL, headers=headers)
        user_info = response.json()
        
        # Firestore에 사용자 정보 저장 또는 업데이트
        save_user_info(user_info)
        
 
    
    # 세션 상태에서 토큰이 있는지 확인
    if 'token' not in st.session_state:
        left, center, right = st.columns([1,2,1])
        with center:
            with st.container():
                
                    st.markdown("#####")
                    col1, col2, col3 = st.columns([1,2,1])
                    with col2:  # 중앙 컬럼에서 요소들을 추가
                        
                    # 없으면 Google 아이콘과 함께 인증 버튼 표시
                        st.image('https://brassy-dolphin-f22.notion.site/image/https%3A%2F%2Fprod-files-secure.s3.us-west-2.amazonaws.com%2Fa60d437f-ec9f-4f91-a164-6f05ff0d8852%2Fd4dbd66d-c2db-43bc-89f0-6c1a06b028f2%2FUntitled.png?table=block&id=813ed66c-d30c-47f9-a970-09825eefb4e9&spaceId=a60d437f-ec9f-4f91-a164-6f05ff0d8852&width=850&userId=&cache=v2', width=400) # 이미지 URL과 너비를 적절히 조정해주세요.
                        
                        result = oauth2.authorize_button("Continue with Google", REDIRECT_URI, SCOPE, icon="data:image/svg+xml;charset=utf-8,%3Csvg xmlns='http://www.w3.org/2000/svg' xmlns:xlink='http://www.w3.org/1999/xlink' viewBox='0 0 48 48'%3E%3Cdefs%3E%3Cpath id='a' d='M44.5 20H24v8.5h11.8C34.7 33.9 30.1 37 24 37c-7.2 0-13-5.8-13-13s5.8-13 13-13c3.1 0 5.9 1.1 8.1 2.9l6.4-6.4C34.6 4.1 29.6 2 24 2 11.8 2 2 11.8 2 24s9.8 22 22 22c11 0 21-8 21-22 0-1.3-.2-2.7-.5-4z'/%3E%3C/defs%3E%3CclipPath id='b'%3E%3Cuse xlink:href='%23a' overflow='visible'/%3E%3C/clipPath%3E%3Cpath clip-path='url(%23b)' fill='%23FBBC05' d='M0 37V11l17 13z'/%3E%3Cpath clip-path='url(%23b)' fill='%23EA4335' d='M0 11l17 13 7-6.1L48 14V0H0z'/%3E%3Cpath clip-path='url(%23b)' fill='%2334A853' d='M0 37l30-23 7.9 1L48 0v48H0z'/%3E%3Cpath clip-path='url(%23b)' fill='%234285F4' d='M48 48L17 24l-4-3 35-10z'/%3E%3C/svg%3E", use_container_width=True)
                        
                        if result and 'token' in result:
                            # 인증이 성공하면 세션 상태에 토큰 저장
                            st.session_state.token = result.get('token')
                            st.rerun()
    #
    else:
        left, center, right = st.columns([1,2,1])
        with center:
            with st.container():
                # 로그아웃 폼 시작
                with st.form("logout_form"):  # 폼 시작
                    col1, col2, col3 = st.columns([1,2,1])
                    with col2:  # 중앙 컬럼에서 요소들을 추가
                        # 중앙 이미지 추가
                        st.image('https://brassy-dolphin-f22.notion.site/image/https%3A%2F%2Fprod-files-secure.s3.us-west-2.amazonaws.com%2Fa60d437f-ec9f-4f91-a164-6f05ff0d8852%2F8c634fb7-8ee8-492b-a181-3345b5b7375a%2FUntitled.png?table=block&id=28c51d75-42ab-4ccc-8d49-b4d015751e77&spaceId=a60d437f-ec9f-4f91-a164-6f05ff0d8852&width=850&userId=&cache=v2', width=400)
                        # 사용자 프로필 사진과 환영 메시지를 Markdown과 HTML을 사용하여 같은 줄에 표시
                        st.markdown(f"""
                        <div style="display:flex;align-items:center;">
                            <img src="{user_info['picture']}" width="80" style="margin-right: 10px;"/>
                            <span>환영합니다, {user_info['name']}님</span>
                        </div>
                        """, unsafe_allow_html=True)

                    # 로그아웃 버튼
                    logout_button = st.form_submit_button("로그아웃")
                if logout_button:
                    # 로그아웃 버튼을 클릭하면, 토큰을 폐기하고 세션 상태에서 토큰 삭제
                    oauth2.revoke_token(st.session_state['token'])
                    del st.session_state['token']
                    st.success("로그아웃 되었습니다.")
                    st.experimental_rerun()
                    
        st.markdown("---")
        # my page부분
        st.image('https://brassy-dolphin-f22.notion.site/image/https%3A%2F%2Fprod-files-secure.s3.us-west-2.amazonaws.com%2Fa60d437f-ec9f-4f91-a164-6f05ff0d8852%2F17d488a8-cfb4-4c13-9efe-1148e273185d%2FUntitled.png?table=block&id=a3da8546-f591-4d9f-b8f0-cc50a1b8f3fc&spaceId=a60d437f-ec9f-4f91-a164-6f05ff0d8852&width=870&userId=&cache=v2',use_column_width=True)
        # 파일 업로드 인터페이스 추가
        uploaded_file = st.file_uploader("파일 업로드", type=None)
        if uploaded_file is not None:
            file_info = {
                'name': uploaded_file.name,
                'type': uploaded_file.type,
                'size': uploaded_file.size,
            }
            file_ref = db.collection(u'files').document(uploaded_file.name)
            file_ref.set(file_info)
            st.success(f"'{uploaded_file.name}' 파일이 성공적으로 업로드되었습니다.")

        # Firestore에서 'files' 컬렉션의 모든 파일 정보 조회
        docs = db.collection(u'files').stream()
        file_list = []
        for doc in docs:
            doc_dict = doc.to_dict()
            doc_dict['id'] = doc.id  # 문서 ID 추가
            file_list.append(doc_dict)

        if file_list:
            df_files = pd.DataFrame(file_list)
            st.write("업로드된 파일 목록:")
            st.dataframe(df_files)  # 업로드된 파일 정보를 표로 표시

            # 사용자가 파일을 선택하여 삭제할 수 있게 하는 부분
            selected_file = st.selectbox("삭제할 파일을 선택하세요.", df_files['name'])
            if st.button("파일 삭제"):
                db.collection(u'files').document(selected_file).delete()
                st.success(f"'{selected_file}' 파일이 삭제되었습니다.")

            # 파일 열람 기능 (여기서는 단순히 파일을 임시 저장소에 저장하고 사용자에게 제공하는 방식을 사용)
            open_file = st.selectbox("열람할 파일을 선택하세요.", df_files['name'])
            if st.button("파일 열람"):
                doc = db.collection(u'files').document(open_file).get()
                if doc.exists:
                    file_info = doc.to_dict()
                    file_path = os.path.join(tempfile.gettempdir(), file_info['name'])
                    # 여기서는 파일이 서버에 이미 저장되어 있다고 가정하고, 해당 경로에서 파일을 찾아 사용자에게 제공합니다.
                    # 실제로는 파일을 저장하고 관리하는 로직이 필요합니다.
                    st.write(f"파일 경로: {file_path}")
                    # 파일을 열람하거나 다운로드할 수 있는 링크 제공 등의 추가 작업이 필요할 수 있습니다.


                                
                    
                    
elif menu == '📝사용방법':
    st.image('https://brassy-dolphin-f22.notion.site/image/https%3A%2F%2Fprod-files-secure.s3.us-west-2.amazonaws.com%2Fa60d437f-ec9f-4f91-a164-6f05ff0d8852%2F339fa6c2-c453-4c3f-a063-9ce8dc533c64%2FUntitled.png?table=block&id=088cf27e-a04c-41c2-a884-d4f4442f6f06&spaceId=a60d437f-ec9f-4f91-a164-6f05ff0d8852&width=1390&userId=&cache=v2', use_column_width=True)
    
    # st.title('사용방법')
    st.title('DGB 고객 분석 시스템 사용방법 입니다. 다음과 같은 흐름으로 분석이 진행됩니다.')
    st.image('https://brassy-dolphin-f22.notion.site/image/https%3A%2F%2Fprod-files-secure.s3.us-west-2.amazonaws.com%2Fa60d437f-ec9f-4f91-a164-6f05ff0d8852%2Ffedded3b-f946-4a77-b700-fd5bcf480d75%2FUntitled.png?table=block&id=37126797-6dff-4319-9e04-4f511ba68b9c&spaceId=a60d437f-ec9f-4f91-a164-6f05ff0d8852&width=1280&userId=&cache=v2',use_column_width=True)
    
   
  
    #데이터 수집&전처리&준지도
    st.image('https://brassy-dolphin-f22.notion.site/image/https%3A%2F%2Fprod-files-secure.s3.us-west-2.amazonaws.com%2Fa60d437f-ec9f-4f91-a164-6f05ff0d8852%2F689c5e7d-5275-46dc-bd17-ad2ffffe66cf%2FUntitled.png?table=block&id=b984c3b9-6461-40c5-b497-35f161b400dc&spaceId=a60d437f-ec9f-4f91-a164-6f05ff0d8852&width=1360&userId=&cache=v2', use_column_width=True)
    st.image('https://brassy-dolphin-f22.notion.site/image/https%3A%2F%2Fprod-files-secure.s3.us-west-2.amazonaws.com%2Fa60d437f-ec9f-4f91-a164-6f05ff0d8852%2F3ceea4b0-b1f7-47e7-89c2-1b6cc5517251%2FUntitled.png?table=block&id=585de0d0-1332-4b19-8ba1-154ac8afc77e&spaceId=a60d437f-ec9f-4f91-a164-6f05ff0d8852&width=850&userId=&cache=v2', use_column_width=True)
    st.image('https://brassy-dolphin-f22.notion.site/image/https%3A%2F%2Fprod-files-secure.s3.us-west-2.amazonaws.com%2Fa60d437f-ec9f-4f91-a164-6f05ff0d8852%2F0a3854ca-2279-4352-94d9-2e8e004b627a%2FUntitled.png?table=block&id=d905cb50-b5e7-45fa-8a40-4ba06b427c85&spaceId=a60d437f-ec9f-4f91-a164-6f05ff0d8852&width=850&userId=&cache=v2', use_column_width=True)
    
    
    
    #클러스터링
    st.image('https://brassy-dolphin-f22.notion.site/image/https%3A%2F%2Fprod-files-secure.s3.us-west-2.amazonaws.com%2Fa60d437f-ec9f-4f91-a164-6f05ff0d8852%2F1456364b-eefa-4158-b9d1-106f8356e69e%2FUntitled.png?table=block&id=7e20f50a-85d5-47af-b5cd-3971e0a2db96&spaceId=a60d437f-ec9f-4f91-a164-6f05ff0d8852&width=850&userId=&cache=v2', use_column_width=True)
    #소셜 네트워크 분석
    st.image('https://brassy-dolphin-f22.notion.site/image/https%3A%2F%2Fprod-files-secure.s3.us-west-2.amazonaws.com%2Fa60d437f-ec9f-4f91-a164-6f05ff0d8852%2Fea0902e2-df38-46f9-8b3c-80fdd01e2abc%2FUntitled.png?table=block&id=6e413406-97a7-4ee2-b9fc-6af336683b96&spaceId=a60d437f-ec9f-4f91-a164-6f05ff0d8852&width=1360&userId=&cache=v2', use_column_width=True)
    #토픽 모델링
    st.image('https://brassy-dolphin-f22.notion.site/image/https%3A%2F%2Fprod-files-secure.s3.us-west-2.amazonaws.com%2Fa60d437f-ec9f-4f91-a164-6f05ff0d8852%2F114f8151-802b-4f9b-a95e-98072a74a92a%2FUntitled.png?table=block&id=c8cf6409-bde9-458a-999a-6c09213d3fc0&spaceId=a60d437f-ec9f-4f91-a164-6f05ff0d8852&width=1290&userId=&cache=v2', use_column_width=True)
    #기회점수 도출
    #st.image(' ', use_column_width=True)
    
elif menu == '📁데이터 수집&전처리':
    st.image('https://brassy-dolphin-f22.notion.site/image/https%3A%2F%2Fprod-files-secure.s3.us-west-2.amazonaws.com%2Fa60d437f-ec9f-4f91-a164-6f05ff0d8852%2Fd07544e2-b5d7-4900-8336-667702df8494%2FUntitled.png?table=block&id=c6a18559-169b-4873-854a-b4e6a68af24f&spaceId=a60d437f-ec9f-4f91-a164-6f05ff0d8852&width=870&userId=&cache=v2', use_column_width=True)
   
    # st.title('데이터 수집 및 전처리')

    # 사용자 선택 옵션
    st.image('https://brassy-dolphin-f22.notion.site/image/https%3A%2F%2Fprod-files-secure.s3.us-west-2.amazonaws.com%2Fa60d437f-ec9f-4f91-a164-6f05ff0d8852%2F1601552c-387c-43c9-9ce5-ac3fb961c99d%2FUntitled.png?table=block&id=4af6f754-9488-4c00-ac24-753ed86c8f76&spaceId=a60d437f-ec9f-4f91-a164-6f05ff0d8852&width=1390&userId=&cache=v2', width=500)
   
    
    st.subheader("3가지의 데이터 수집 방법 선택 후 'Add filters' 기능으로 원하는 데이터만 필터링 할 수 있습니다.")


    def filter_dataframe(df: pd.DataFrame) -> pd.DataFrame:
        """
        Adds a UI on top of a dataframe to let viewers filter columns

        Args:
            df (pd.DataFrame): Original dataframe

        Returns:
            pd.DataFrame: Filtered dataframe
        """
        modify = st.checkbox("Add filters")

        if not modify:
            return df

        df = df.copy()

        # Filter columns based on user input
        modification_container = st.container()

        with modification_container:
            to_filter_columns = st.multiselect("Filter dataframe on", df.columns)
            for column in to_filter_columns:
                left, right = st.columns((1, 20))
                left.write("↳")
                # Check column data type and apply appropriate filtering
                user_input = right.text_input(f"Filter by {column}")
                if user_input:
                    df = df[df[column].str.contains(user_input)]

        return df
    
    
    
    
    
    
    
    
    
    
    # st.markdown("<h1 style='color: blue;'>데이터 수집 방법 선택</h1>", unsafe_allow_html=True)

    option = st.radio('원하는 데이터 수집 방법을 선택하세요.',
                    ('1) 직접 파일 업로드', '2) 앱 스토어 고객 리뷰 크롤링', '3) GCP연결-->미리 수집된 데이터베이스'))

    # st.markdown("---")  # 섹션 구분을 위한 수평선
    
    

    if option == '1) 직접 파일 업로드':
        st.subheader('1) 직접 파일 업로드')
        uploaded_file = st.file_uploader("파일을 업로드하세요.", type=['csv', 'xlsx'], key="unique_key_1")
        if uploaded_file is not None:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            elif uploaded_file.name.endswith('.xlsx'):
                df = pd.read_excel(uploaded_file)
            st.write("업로드된 파일의 데이터프레임:")
            st.dataframe(df)
            # Filtered dataframe
            st.dataframe(filter_dataframe(df))
            
            
            

    elif option == '2) 앱 스토어 고객 리뷰 크롤링':
        st.subheader('2) 앱 스토어 고객 리뷰 크롤링')
        # 여기에 앱 스토어 리뷰 크롤링 관련 코드를 추가하세요.
        # 예제 코드에서 제공된 앱 검색 및 리뷰 검색 코드를 여기에 삽입하세요.
        st.text('앱 ID를 검색 후 앱 리뷰를 볼 수 있습니다.')
            
        # 스트림릿 타이틀 설정
        # st.text('앱 ID 검색')

        # 앱 이름 입력 받기
        app_name = st.text_input('앱 ID 검색: 앱 이름을 입력하세요', '')

        # 앱 검색 및 결과 표시
        if app_name:
                    st.subheader('검색 결과')
                    results = gps_search(app_name, lang='ko', country='kr')
                    for result in results[:5]:  # 최대 5개의 검색 결과 반환
                        st.write(f"앱 이름: {result['title']}, 앱 ID: {result['appId']}")

                    # 두번째  폼
                    st.subheader('앱 리뷰 검색')

                    # 사용자 입력 받기
                    app_input = st.text_input('앱 이름 또는 앱 ID를 입력하세요:', '')

                    # 데이터프레임 초기화
                    df_reviews = pd.DataFrame()

                    # 검색된 앱 리뷰 표시
                    if app_input:
                        try:
                            # 앱 리뷰 가져오기
                            app = AppStore(country="kr", app_name=app_input)
                            app.review(how_many=3)  # 최근 3개의 리뷰만 가져옴
                            
                            # 리뷰가 있는 경우 데이터프레임으로 변환
                            if app.reviews:
                                df_reviews = pd.DataFrame(app.reviews)
                                st.dataframe(df_reviews)
                                st.dataframe(filter_dataframe(df_reviews))
                        except Exception as e:
                            st.error(f'앱 리뷰를 가져오는 도중 오류가 발생했습니다: {e}')
    
    
    
 
    elif option == '3) GCP연결-->미리 수집된 데이터베이스':
        from google.cloud import storage
        st.subheader('3) GCP연결-->미리 수집된 데이터베이스')
        
        def connect_to_gcs():
            # JSON 키 파일의 경로를 명시적으로 지정하여 GCS 클라이언트 객체 생성
            client_gcs = storage.Client.from_service_account_json(r'C:\My\Competition\DGB\Final\streamlit\dgb-api-test-fb40a3e53580.json')
            return client_gcs

        def load_dataframe_from_gcs(client_gcs, bucket_name, file_name):
            # GCS 버킷에서 파일 읽어오기
            bucket = client_gcs.get_bucket(bucket_name)
            blob = bucket.blob(file_name)
            data = blob.download_as_string().decode("utf-8")  # 파일을 바이트로 읽어와 문자열로 디코딩
            data_io = StringIO(data)
            return pd.read_csv(data_io)

        def main():
            st.title('')
            # GCS 연결
            client_gcs = connect_to_gcs()

            if client_gcs:  # client_gcs가 None이 아니면 실행
                # GCS 버킷 목록 가져오기
                buckets = client_gcs.list_buckets()
                bucket_options = [bucket.name for bucket in buckets]

                # 사용자가 선택한 버킷 이름
                selected_bucket_name = st.selectbox("버킷 선택", bucket_options)

                if selected_bucket_name:
                    # 선택한 버킷에서 파일 목록 가져오기
                    bucket = client_gcs.get_bucket(selected_bucket_name)
                    blobs = bucket.list_blobs()
                    file_options = [blob.name for blob in blobs]

                    # 사용자가 선택한 파일명
                    selected_file_name = st.selectbox("파일 선택", file_options)

                    if selected_file_name and st.button('수집'):
                        # 선택한 파일에서 데이터프레임으로 로드
                        df = load_dataframe_from_gcs(client_gcs, selected_bucket_name, selected_file_name)
                        # 데이터프레임 출력
                        st.write(df)
                        st.dataframe(filter_dataframe(df))

        if __name__ == '__main__':
            main()

            
            
    st.markdown("---")  # 세로 구분선
    
    #전처리 부분 ----------------------------------------------------------
    # st.title('전처리')
    st.image('https://brassy-dolphin-f22.notion.site/image/https%3A%2F%2Fprod-files-secure.s3.us-west-2.amazonaws.com%2Fa60d437f-ec9f-4f91-a164-6f05ff0d8852%2F890fc8e1-45bb-4718-a9bb-b7e2ec8e36de%2FUntitled.png?table=block&id=64a95fe1-944a-4d83-b10b-2fd436d9c263&spaceId=a60d437f-ec9f-4f91-a164-6f05ff0d8852&width=1390&userId=&cache=v2', width=500)
    st.image('https://brassy-dolphin-f22.notion.site/image/https%3A%2F%2Fprod-files-secure.s3.us-west-2.amazonaws.com%2Fa60d437f-ec9f-4f91-a164-6f05ff0d8852%2F616ab9b7-d048-4616-9f74-384ead265592%2FUntitled.png?table=block&id=aef18537-2845-4012-b892-116bbdfa71a4&spaceId=a60d437f-ec9f-4f91-a164-6f05ff0d8852&width=1390&userId=&cache=v2', use_column_width=True)
    
  
    from konlpy.tag import Okt
    import re
    
    st.subheader("전처리를 하기 위해선 keyword, date, text, tok_text가 필요합니다. 컬럼 수정/생성/삭제 기능을 이용해주세요.")


    # 파일 업로드 위젯
    uploaded_file = st.file_uploader("파일을 선택해주세요.", type=["csv"])

    if uploaded_file is not None:
        # 업로드된 파일을 Pandas DataFrame으로 로드
        df = pd.read_csv(uploaded_file)
        
        # 데이터 프레임을 화면에 표시
        st.write(df)
        
        # Radio 버튼을 사용하여 원하는 기능 선택
        option = st.radio('원하는 기능을 선택하세요.',
                        ('컬럼명 수정', '컬럼 생성', '컬럼 삭제'))
        
        # 컬럼명 수정 기능
        if option == '컬럼명 수정':
            old_column_name = st.selectbox("수정할 컬럼명을 선택해주세요.", df.columns)
            new_column_name = st.text_input("새 컬럼명을 입력해주세요.", "")
            if st.button("컬럼명 수정"):
                df.rename(columns={old_column_name: new_column_name}, inplace=True)
                st.success(f"'{old_column_name}' 컬럼명이 '{new_column_name}'(으)로 수정되었습니다.")
                st.write(df)
        
        # 새로운 컬럼 추가 기능
        elif option == '컬럼 생성':
            new_col_name = st.text_input("새로 생성할 컬럼명을 입력해주세요.", key='new_col')
            new_col_value = st.text_input("새 컬럼의 값을 입력해주세요.", key='new_col_val')
            if st.button("컬럼 생성", key='create_col'):
                df[new_col_name] = new_col_value
                st.success(f"'{new_col_name}' 컬럼이 생성되었습니다.")
                st.write(df)
        
        # 컬럼 삭제 기능
        elif option == '컬럼 삭제':
            del_column_name = st.selectbox("삭제할 컬럼명을 선택해주세요.", df.columns, key='del_col')
            if st.button("컬럼 삭제", key='delete_col'):
                df.drop(del_column_name, axis=1, inplace=True)
                st.success(f"'{del_column_name}' 컬럼이 삭제되었습니다.")
                st.write(df)


        # 전처리 시작 버튼
        import numpy as np
        import re
        from konlpy.tag import Okt, Komoran, Mecab, Hannanum, Kkma
        from datetime import datetime

        
        if st.button("전처리 시작"):
            

            # 'text' 컬럼을 공백 기준으로 토큰화하여 'tok_text' 컬럼 생성
            df['tok_text'] = df['text'].fillna('').apply(lambda x: " ".join(x.split()))
            
            # 중복값 제거
            df = df.drop_duplicates(subset=['text'])

            # 결측치 처리
            df['text'] = df['text'].fillna('')
            
            
            
            
            # 불용어 리스트 정의
            stopwords = ["가", "가까스로", "가령", "각", "각각", "각자", "각종", "갖고말하자면", "같다", "같이", "개의치않고", "거니와", "거바", "거의", "것", "것과 같이", "것들", "게다가", "게우다", "겨우", "견지에서", "결과에 이르다", "결국", "결론을 낼 수 있다", "겸사겸사", "고려하면", "고로", "곧", "공동으로", "과", "과연", "관계가 있다", "관계없이", "관련이 있다", "관하여", "관한", "관해서는", "구", "구체적으로", "구토하다", "그", "그들", "그때", "그래", "그래도", "그래서", "그러나", "그러니", "그러니까", "그러면", "그러므로", "그러한즉", "그런 까닭에", "그런데", "그런즉", "그럼", "그럼에도 불구하고", "그렇게 함으로써", "그렇지", "그렇지 않다면", "그렇지 않으면", "그렇지만", "그렇지않으면", "그리고", "그리하여", "그만이다", "그에 따르는", "그위에", "그저", "그중에서", "그치지 않다", "근거로", "근거하여", "기대여", "기점으로", "기준으로", "기타", "까닭으로", "까악", "까지", "까지 미치다", "까지도", "꽈당", "끙끙", "끼익", "나", "나머지는", "남들", "남짓", "너", "너희", "너희들", "네", "넷", "년", "논하지 않다", "놀라다", "누가 알겠는가", "누구", "다른", "다른 방면으로", "다만", "다섯", "다소", "다수", "다시 말하자면", "다시말하면", "다음", "다음에", "다음으로", "단지", "답다", "당신", "당장", "대로 하다", "대하면", "대하여", "대해 말하자면", "대해서", "댕그", "더구나", "더군다나", "더라도", "더불어", "더욱더", "더욱이는", "도달하다", "도착하다", "동시에", "동안", "된바에야", "된이상", "두번째로", "둘", "둥둥", "뒤따라", "뒤이어", "든간에", "들", "등", "등등", "딩동", "따라", "따라서", "따위", "따지지 않다", "딱", "때", "때가 되어", "때문에", "또", "또한", "뚝뚝", "라 해도", "령", "로", "로 인하여", "로부터", "로써", "륙", "를", "마음대로", "마저", "마저도", "마치", "막론하고", "만 못하다", "만약", "만약에", "만은 아니다", "만이 아니다", "만일", "만큼", "말하자면", "말할것도 없고", "매", "매번", "메쓰겁다", "몇", "모", "모두", "무렵", "무릎쓰고", "무슨", "무엇", "무엇때문에", "물론", "및", "바꾸어말하면", "바꾸어말하자면", "바꾸어서 말하면", "바꾸어서 한다면", "바꿔 말하면", "바로", "바와같이", "밖에 안된다", "반대로", "반대로 말하자면", "반드시", "버금", "보는데서", "보다더", "보드득", "본대로", "봐", "봐라", "부류의 사람들", "부터", "불구하고", "불문하고", "붕붕", "비걱거리다", "비교적", "비길수 없다", "비로소", "비록", "비슷하다", "비추어 보아", "비하면", "뿐만 아니라", "뿐만아니라", "뿐이다", "삐걱", "삐걱거리다", "사", "삼", "상대적으로 말하자면", "생각한대로", "설령", "설마", "설사", "셋", "소생", "소인", "솨", "쉿", "습니까", "습니다", "시각", "시간", "시작하여", "시초에", "시키다", "실로", "심지어", "아", "아니", "아니나다를가", "아니라면", "아니면", "아니었다면", "아래윗", "아무거나", "아무도", "아야", "아울러", "아이", "아이고", "아이구", "아이야", "아이쿠", "아하", "아홉", "안 그러면", "않기 위하여", "않기 위해서", "알 수 있다", "알았어", "앗", "앞에서", "앞의것", "야", "약간", "양자", "어", "어기여차", "어느", "어느 년도", "어느것", "어느곳", "어느때", "어느쪽", "어느해", "어디", "어때", "어떠한", "어떤", "어떤것", "어떤것들", "어떻게", "어떻해", "어이", "어째서", "어쨋든", "어쩌라고", "어쩌면", "어쩌면 해도", "어쩌다", "어쩔수 없다", "어찌", "어찌됏든", "어찌됏어", "어찌하든지", "어찌하여", "언제", "언젠가", "얼마", "얼마 안 되는 것", "얼마간", "얼마나", "얼마든지", "얼마만큼", "얼마큼", "엉엉", "에", "에 가서", "에 달려 있다", "에 대해", "에 있다", "에 한하다", "에게", "에서", "여", "여기", "여덟", "여러분", "여보시오", "여부", "여섯", "여전히", "여차", "연관되다", "연이서", "영", "영차", "옆사람", "예", "예를 들면", "예를 들자면", "예컨대", "예하면", "오", "오로지", "오르다", "오자마자", "오직", "오호", "오히려", "와", "와 같은 사람들", "와르르", "와아", "왜", "왜냐하면", "외에도", "요만큼", "요만한 것", "요만한걸", "요컨대", "우르르", "우리", "우리들", "우선", "우에 종합한것과같이", "운운", "월", "위에서 서술한바와같이", "위하여", "위해서", "윙윙", "육", "으로", "으로 인하여", "으로서", "으로써", "을", "응", "응당", "의", "의거하여", "의지하여", "의해", "의해되다", "의해서", "이", "이 되다", "이 때문에", "이 밖에", "이 외에", "이 정도의", "이것", "이곳", "이때", "이라면", "이래", "이러이러하다", "이러한", "이런", "이럴정도로", "이렇게 많은 것", "이렇게되면", "이렇게말하면", "이렇구나", "이로 인하여", "이르기까지", "이리하여", "이만큼", "이번", "이봐", "이상", "이어서", "이었다", "이와 같다", "이와 같은", "이와 반대로", "이와같다면", "이외에도", "이용하여", "이유만으로", "이젠", "이지만", "이쪽", "이천구", "이천육", "이천칠", "이천팔", "인 듯하다", "인젠", "일", "일것이다", "일곱", "일단", "일때", "일반적으로", "일지라도", "임에 틀림없다", "입각하여", "입장에서", "잇따라", "있다", "자", "자기", "자기집", "자마자", "자신", "잠깐", "잠시", "저", "저것", "저것만큼", "저기", "저쪽", "저희", "전부", "전자", "전후", "점에서 보아", "정도에 이르다", "제", "제각기", "제외하고", "조금", "조차", "조차도", "졸졸", "좀", "좋아", "좍좍", "주룩주룩", "주저하지 않고", "줄은 몰랏다", "줄은모른다", "중에서", "중의하나", "즈음하여", "즉", "즉시", "지든지", "지만", "지말고", "진짜로", "쪽으로", "차라리", "참", "참나", "첫번째로", "쳇", "총적으로", "총적으로 말하면", "총적으로 보면", "칠", "콸콸", "쾅쾅", "쿵", "타다", "타인", "탕탕", "토하다", "통하여", "툭", "퉤", "틈타", "팍", "팔", "퍽", "펄렁", "하", "하게될것이다", "하게하다", "하겠는가", "하고 있다", "하고있었다", "하곤하였다", "하구나", "하기 때문에", "하기 위하여", "하기는한데", "하기만 하면", "하기보다는", "하기에", "하나", "하느니", "하는 김에", "하는 편이 낫다", "하는것도", "하는것만 못하다", "하는것이 낫다", "하는바", "하더라도", "하도다", "하도록시키다", "하도록하다", "하든지", "하려고하다", "하마터면", "하면 할수록", "하면된다", "하면서", "하물며", "하여금", "하여야", "하자마자", "하지 않는다면", "하지 않도록", "하지마", "하지마라", "하지만", "하하", "한 까닭에", "한 이유는", "한 후", "한다면", "한다면 몰라도", "한데", "한마디", "한적이있다", "한켠으로는", "한항목", "할 따름이다", "할 생각이다", "할 줄 안다", "할 지경이다", "할 힘이 있다", "할때", "할만하다", "할망정", "할뿐", "할수있다", "할수있어", "할줄알다", "할지라도", "할지언정", "함께", "해도된다", "해도좋다", "해봐요", "해서는 안된다", "해야한다", "해요", "했어요", "향하다", "향하여", "향해서", "허", "허걱", "허허", "헉", "헉헉", "헐떡헐떡", "형식으로 쓰여", "혹시", "혹은", "혼자", "훨씬", "휘익", "휴", "흐흐", "흥", "힘입어"]

            # 불용어 제거 함수
            def remove_stopwords(text):
                tokens = text.split()  # 텍스트를 공백 기준으로 토큰화
                tokens_filtered = [word for word in tokens if not word in stopwords]  # 불용어가 아닌 토큰만 선택
                return " ".join(tokens_filtered)  # 토큰들을 다시 공백으로 결합하여 반환

            # 전처리 함수 수정
            def preprocess_text(text):
                text = re.sub(r'<[^>]+>', '', text)  # HTML 태그 제거
                text = re.sub(r'\n', ' ', text)  # 개행문자 제거
                text = re.sub(r'\s+', ' ', text)  # 여러 개의 공백을 하나의 공백으로
                text = re.sub(r'[0-9]+갤', '개월', text)  # 숫자+갤 -> 숫자개월
                text = re.sub(r'[^가-힣.]', ' ', text)  # 한글과 마침표를 제외한 모든 문자 제거
                text = re.sub(r'\.\s+', '.', text)  # 마침표 다음의 공백 제거
                text = re.sub(r'\.{2,}', '.', text)  # 여러 개의 마침표를 하나로
                text = re.sub(r'\s+', ' ', text)  # 다시 한 번 여러 개의 공백을 하나의 공백으로
                text = remove_stopwords(text)  # 불용어 제거
                return text

            # 기존의 코드에서 'text' 컬럼을 전처리하는 부분에 불용어 제거 기능이 추가됨
            df.loc[:, 'text'] = df['text'].apply(preprocess_text)

            # 형태소 분석기 설정
            def get_tokenizer(tokenizer_name):
                if tokenizer_name == "komoran":
                    return Komoran()
                elif tokenizer_name == "okt":
                    return Okt()
                elif tokenizer_name == "mecab":
                    return Mecab()
                elif tokenizer_name == "hannanum":
                    return Hannanum()
                elif tokenizer_name == "kkma":
                    return Kkma()
                else:
                    return Okt()

            tokenizer = get_tokenizer("okt")
            
            def pos_tagging_and_filter(text):
                pos_tokens = tokenizer.pos(text)
                filtered_tokens = [token for token, pos in pos_tokens if pos != 'Josa']
                return filtered_tokens
            
            # 전처리 및 형태소 분석 실행
            df['text'] = df['text'].apply(preprocess_text)
            df['tok_text'] = df['text'].apply(pos_tagging_and_filter)

            # 형태소 분석 결과를 CSV 파일로 저장
            now = datetime.now()
            formatted_time = now.strftime("%Y%m%d_%H%M%S")
            filename = f"형태소분석_{formatted_time}.csv"
            df.to_csv(filename, index=False, encoding='utf-8-sig')

            # 완료 메시지 및 파일 다운로드 링크
            st.success('데이터 전처리가 완료되었습니다.')
            st.download_button('결과 다운로드', data=df.to_csv(index=False), file_name=filename, mime='text/csv')

            
            # 필요한 컬럼만 선택
            df = df[['keyword', 'date', 'text', 'tok_text']]
            
            st.success("전처리가 완료되었습니다.")
            st.write(df)

            
    st.markdown("---")  # 세로 구분선
    
    
    # GPT-2로 은행 관련 여부 판단 함수
    def is_bank_related_gpt2(text):
        prompt = f"이 텍스트는 은행, 금융 등에 관련된 내용과 관련이 있습니까? 예, 아니요로 대답하시오. '{text}'"
        inputs = tokenizer.encode(prompt, return_tensors='pt', truncation=True, max_length=950, padding=True)
        try:
            outputs = headmodel.generate(inputs, max_length=inputs.shape[1] + 15, num_return_sequences=1)
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            return "예" in response
        except IndexError as e:
            st.write(f"Error: {e}, Input Text: {text}")
            return False
    
    #  train_and_predict_model 함수가 정의
    # 모델 학습 및 예측을 위한 함수
    def train_and_predict_model(model, X_labeled, y_labeled, X_unlabeled):
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        model.fit(X_labeled, y_labeled, epochs=10, batch_size=32, validation_split=0.2)
        return (model.predict(X_unlabeled) > 0.5).astype(int)


    # GPT-2 모델에서 임베딩 추출 함수
    def get_embeddings(input_ids):
        input_tensor = torch.tensor([input_ids], dtype=torch.long)
        with torch.no_grad():
            outputs = model(input_tensor)
        return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()

    #####################
    #유효 문장 식별 AI
    st.image('https://brassy-dolphin-f22.notion.site/image/https%3A%2F%2Fprod-files-secure.s3.us-west-2.amazonaws.com%2Fa60d437f-ec9f-4f91-a164-6f05ff0d8852%2Fe060f141-f071-4186-a13a-8e4943ab6393%2FUntitled.png?table=block&id=9191834a-8351-48a4-a6a7-37bb893f385c&spaceId=a60d437f-ec9f-4f91-a164-6f05ff0d8852&width=870&userId=&cache=v2', width=500)
    
    uploaded_file = st.file_uploader("CSV 파일 업로드", type="csv")

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.write("데이터 미리보기:", df.head())

        # 토크나이저 및 모델 불러오기
        tokenizer = PreTrainedTokenizerFast.from_pretrained("skt/kogpt2-base-v2", bos_token='</s>', eos_token='</s>', unk_token='<unk>', pad_token='<pad>', mask_token='<mask>')
        model = GPT2Model.from_pretrained("skt/kogpt2-base-v2")
        headmodel = GPT2LMHeadModel.from_pretrained("skt/kogpt2-base-v2")
        
        # 레이블링을 위한 샘플 추출 및 임베딩 추출
        labeling_df = pd.DataFrame(df['text'].sample(frac=0.03))
        new_data = [get_embeddings(tokenizer.encode(text, truncation=True, max_length=1024, padding=True)) for text in labeling_df['text']]
        embeddings = np.array(new_data)
        index = faiss.IndexFlatL2(embeddings.shape[1])
        index.add(embeddings)

        # 레이블링 진행
        labeled_data = [(text, 1 if is_bank_related_gpt2(text) else 0) for text in labeling_df['text']]
        df_labeled = pd.DataFrame(labeled_data, columns=['text', 'Label'])
        df_merged = pd.merge(df, df_labeled, on='text', how='left')

        # 이하 코드는 원본 코드와 동일하며, eval 함수 사용이 필요한 부분에 대해서는 주의하여 사용합니다.
        # 데이터 준비 및 모델 학습, 예측 코드...
    


        # LSTM과 RNN 학습을 위한 데이터 준비
        texts = df_merged['tok_text'].apply(eval).astype(str)
        labels = df_merged['Label']
        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(texts)
        sequences = tokenizer.texts_to_sequences(texts)
        max_seq_length = max(len(seq) for seq in sequences)
        padded_sequences = pad_sequences(sequences, maxlen=max_seq_length)
        labeled_indices = labels.notna()
        unlabeled_indices = ~labeled_indices
        X_labeled = padded_sequences[labeled_indices]
        y_labeled = labels[labeled_indices].astype(int)
        X_unlabeled = padded_sequences[unlabeled_indices]

        # LSTM을 통한 학습 및 예측
        model_LSTM = Sequential([
            Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=100, input_length=max_seq_length),
            LSTM(128),
            Dense(1, activation='sigmoid')
        ])
        LSTM_pseudo_labels = train_and_predict_model(model_LSTM, X_labeled, y_labeled, X_unlabeled)
        df_merged.loc[unlabeled_indices, 'LSTM_Label'] = LSTM_pseudo_labels.flatten()

        # RNN을 통한 학습 및 예측
        model_RNN = Sequential([
            Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=100, input_length=max_seq_length),
            SimpleRNN(128),
            Dense(1, activation='sigmoid')
        ])
        RNN_pseudo_labels = train_and_predict_model(model_RNN, X_labeled, y_labeled, X_unlabeled)
        df_merged.loc[unlabeled_indices, 'RNN_Label'] = RNN_pseudo_labels.flatten()

        # LSTM과 RNN의 예측 결과가 다를 경우, GRU를 사용하여 재분류
        different_indices = df_merged[(df_merged['LSTM_Label'] != df_merged['RNN_Label']) & (unlabeled_indices)].index
        if different_indices.any():
            X_different = padded_sequences[different_indices]
            model_GRU = Sequential([
                Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=100, input_length=max_seq_length),
                GRU(128),
                Dense(1, activation='sigmoid')
            ])
            model_GRU.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
            model_GRU.fit(X_labeled, y_labeled, epochs=10, batch_size=32, validation_split=0.2)
            gru_pseudo_labels = (model_GRU.predict(X_different) > 0.5).astype(int)
            df_merged.loc[different_indices, 'GRU_Label'] = gru_pseudo_labels.flatten()
            df_merged['final_label'] = np.where(df_merged['LSTM_Label'] == df_merged['RNN_Label'], df_merged['LSTM_Label'], df_merged['GRU_Label'])
        else:
            df_merged['final_label'] = df_merged['LSTM_Label']
        
        df_merged = df_merged[(df_merged['final_label'] != 0) & (df_merged['final_label'].notna())]
        st.write("레이블링된 데이터:", df_merged)
        st.download_button(label="CSV로 다운로드", data=df_merged.to_csv(index=False).encode('utf-8'), file_name='labeled_data.csv', mime='text/csv')

        
        
    
    

        
            
            
elif menu == '📊클러스터링':
    st.image('https://brassy-dolphin-f22.notion.site/image/https%3A%2F%2Fprod-files-secure.s3.us-west-2.amazonaws.com%2Fa60d437f-ec9f-4f91-a164-6f05ff0d8852%2F28c7399d-ab78-40ef-a2c4-7b713329de8e%2FUntitled.png?table=block&id=b93be364-a1f9-426c-8121-50b0ebb989fe&spaceId=a60d437f-ec9f-4f91-a164-6f05ff0d8852&width=1390&userId=&cache=v2', use_column_width=True)
   
    # st.title('클러스터링')
    st.image('https://brassy-dolphin-f22.notion.site/image/https%3A%2F%2Fprod-files-secure.s3.us-west-2.amazonaws.com%2Fa60d437f-ec9f-4f91-a164-6f05ff0d8852%2F1456364b-eefa-4158-b9d1-106f8356e69e%2FUntitled.png?table=block&id=7e20f50a-85d5-47af-b5cd-3971e0a2db96&spaceId=a60d437f-ec9f-4f91-a164-6f05ff0d8852&width=850&userId=&cache=v2', width=900)
    # 파일 업로드 부분
    st.title("파일 업로드")
    uploaded_file = st.file_uploader("CSV file", type=['csv'])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        desired_columns = ['keyword', 'date', 'text', 'tok_text']
        df = df[desired_columns]
        df = df.dropna(subset=['text']).reset_index(drop=True)

        # Doc2Vec 모델 훈련
        tagged_data = [TaggedDocument(words=_d, tags=[str(i)]) for i, _d in enumerate(df['tok_text'])]
        model = Doc2Vec(vector_size=50, min_count=2, epochs=40)
        model.build_vocab(tagged_data)
        model.train(tagged_data, total_examples=model.corpus_count, epochs=model.epochs)
        df['vector'] = [model.dv[str(i)].tolist() for i in range(len(tagged_data))]

        # PCA 수행
        vector_array = np.array(df['vector'].tolist())
        pca = PCA(n_components=2)
        df[['PC1', 'PC2']] = pca.fit_transform(vector_array)

        # KMeans 클러스터링
        X = df[['PC1', 'PC2']]
        sse = []
        silhouette_coefficients = []
        for k in range(3, 11):
            kmeans = KMeans(n_clusters=k, random_state=42)
            kmeans.fit(X)
            sse.append(kmeans.inertia_)
            score = silhouette_score(X, kmeans.labels_)
            silhouette_coefficients.append(score)
        top_k_indices = sorted(range(len(silhouette_coefficients)), key=lambda i: silhouette_coefficients[i], reverse=True)[:3]
        optimal_k = min([index + 3 for index in top_k_indices])

        # 최적의 k로 K-means 클러스터링 수행
        kmeans = KMeans(n_clusters=optimal_k, random_state=42)
        df['Cluster'] = kmeans.fit_predict(X)

        # 클러스터링 결과 시각화
        fig, ax = plt.subplots()
        scatter = ax.scatter(df['PC1'], df['PC2'], c=df['Cluster'], cmap='viridis', marker='o', edgecolor='black')
        legend1 = ax.legend(*scatter.legend_elements(), title="Clusters")
        ax.add_artist(legend1)
        st.pyplot(fig)

        # 클러스터별 TF-IDF 추출 및 상위 단어
        okt = Okt()
        def tokenize(text):
            return okt.nouns(text)

        vectorizer = TfidfVectorizer(tokenizer=tokenize, max_features=1000)
        X_tfidf = vectorizer.fit_transform(df['text'])
        tfidf_df = pd.DataFrame(X_tfidf.toarray(), columns=vectorizer.get_feature_names_out())
        top_tfidf_words_per_cluster = {}
        for cluster in range(optimal_k):
            cluster_indices = df[df['Cluster'] == cluster].index
            mean_tfidf = tfidf_df.loc[cluster_indices].mean(axis=0)
            top_words = mean_tfidf.nlargest(20)
            filtered_words = top_words[top_words.index.str.len() > 1]
            top_tfidf_words_per_cluster[cluster] = filtered_words
            
        # 클러스터별 상위 TF-IDF 단어 추출
        okt = Okt()
        def tokenize(text):
            return okt.nouns(text)

        vectorizer = TfidfVectorizer(tokenizer=tokenize, max_features=1000)
        X_tfidf = vectorizer.fit_transform(df['text'])
        tfidf_df = pd.DataFrame(X_tfidf.toarray(), columns=vectorizer.get_feature_names_out())
        
        top_tfidf_words_per_cluster = {}
        for cluster in range(optimal_k):
            cluster_indices = df[df['Cluster'] == cluster].index
            mean_tfidf = tfidf_df.loc[cluster_indices].mean(axis=0)
            top_words = mean_tfidf.nlargest(20)
            filtered_words = top_words[top_words.index.str.len() > 1]
            top_tfidf_words_per_cluster[cluster] = filtered_words

        # 모델과 토크나이저 로드
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = AutoModelForCausalLM.from_pretrained("MoaData/Myrrh_solar_10.7b_3.0").to(device)
        tokenizer = AutoTokenizer.from_pretrained("MoaData/Myrrh_solar_10.7b_3.0")
        
        # 주제 입력 받기
        topic = st.text_input("분석할 주제를 입력하세요 (예: 은행):")
        if topic:
            # 관련성 검사 함수 정의
            def check_relevance(words, topic):
                relevant_words = []
                model.eval()
                for word in words:
                    input_text = f"주제: {topic}, 단어: {word} - 서로 관련 있나? 예 또는 아니오로만 짧게 대답해라."
                    input_ids = tokenizer.encode(input_text, return_tensors='pt').to(device)
                    outputs = model.generate(input_ids, max_length=60)
                    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
                    if "예" in result:
                        relevant_words.append(word)
                return relevant_words

            # 클러스터별로 단어 검사 및 출력
            for cluster, words in top_tfidf_words_per_cluster.items():
                relevant_words = check_relevance(list(words.index), topic)
                st.write(f"클러스터 {cluster}에서 주제 '{topic}'과 관련된 단어들:", relevant_words)

            # 문장 relevancy 함수
            model_sentence = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

            def find_most_relevant_sentence(texts, word, topic):
                relevant_texts = [text for text in texts if word in text]
                if not relevant_texts:
                    return "해당 단어를 포함하는 문장 없음"

                topic_embedding = model_sentence.encode([topic], convert_to_tensor=True)
                text_embeddings = model_sentence.encode(relevant_texts, convert_to_tensor=True)
                cos_scores = util.pytorch_cos_sim(topic_embedding, text_embeddings)[0]
                highest_score_index = torch.argmax(cos_scores).item()
                return relevant_texts[highest_score_index]

            # 각 클러스터별로 가장 관련있는 문장 찾기
            cluster_dataframes = {}
            for cluster in range(optimal_k):
                data = []
                cluster_texts = df[df['Cluster'] == cluster]['text'].tolist()
                words = list(top_tfidf_words_per_cluster[cluster].index)
                for word in words:
                    most_relevant_sentence = find_most_relevant_sentence(cluster_texts, word, topic)
                    data.append({
                        'word': word,
                        'sentence': most_relevant_sentence
                    })
                cluster_dataframes[cluster] = pd.DataFrame(data)
                st.write(f"클러스터 {cluster}의 데이터:")
                st.dataframe(cluster_dataframes[cluster])

            # 클러스터 이름 생성
            def generate_cluster_name(texts):
                combined_text = " ".join(texts)
                prompt = f"다음 설명을 보고 해당 자연어 군집에 적합한 이름을 하나만 지어주세요: '{combined_text}'"
                inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=1024)
                outputs = model.generate(inputs, max_length=50)
                name = tokenizer.decode(outputs[0], skip_special_tokens=True)
                return name

            cluster_names = []
            for cluster, frame in cluster_dataframes.items():
                texts = frame['sentence'].tolist()
                cluster_name = generate_cluster_name(texts)
                cluster_names.append({'Cluster': cluster, 'Name': cluster_name})

            clustername_df = pd.DataFrame(cluster_names)
            st.write("클러스터 이름:")
            st.dataframe(clustername_df)
    
    
    
       
    
elif menu == '📈소셜 네트워크 분석':
    st.image('https://brassy-dolphin-f22.notion.site/image/https%3A%2F%2Fprod-files-secure.s3.us-west-2.amazonaws.com%2Fa60d437f-ec9f-4f91-a164-6f05ff0d8852%2Fea4d3e84-e5e4-400b-b04d-5c85677f600f%2FUntitled.png?table=block&id=5ab154fa-6728-491f-bda0-ee026ed9ad51&spaceId=a60d437f-ec9f-4f91-a164-6f05ff0d8852&width=1390&userId=&cache=v2', use_column_width=True)
    st.image('https://brassy-dolphin-f22.notion.site/image/https%3A%2F%2Fprod-files-secure.s3.us-west-2.amazonaws.com%2Fa60d437f-ec9f-4f91-a164-6f05ff0d8852%2Fe384251a-7f6a-4d60-ae2b-5a19493577ca%2F43105e6b-98c7-482a-aa41-2dbd834f1e72.png?table=block&id=24bbb6fa-6c9e-48ad-a0dd-590f78b7c047&spaceId=a60d437f-ec9f-4f91-a164-6f05ff0d8852&width=500&userId=&cache=v2', width=900)
    
    
    # 한글 폰트 설정
    plt.rcParams['font.family'] = 'Malgun Gothic'
    plt.rcParams['axes.unicode_minus'] = False

    # Streamlit 앱 제목
    st.title("Text Data Clustering and Visualization")

    # 파일 업로드
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)

        # 클러스터별로 데이터 분리
        clusters = df.groupby('Cluster')

        # Okt 객체 생성
        okt = Okt()

        # 명사 추출 함수
        def extract_nouns(text):
            tokens = okt.nouns(text)
            return [token for token in tokens if len(token) > 1]

        # 네트워크 생성 함수
        def create_network(texts):
            vectorizer = TfidfVectorizer(tokenizer=extract_nouns)
            tfidf_matrix = vectorizer.fit_transform(texts)
            cosine_sim = cosine_similarity(tfidf_matrix)
            words = vectorizer.get_feature_names_out()

            G = nx.Graph()
            n = min(len(words), cosine_sim.shape[0])
            for i in range(n):
                for j in range(i + 1, n):
                    if cosine_sim[i, j] > 0.1:
                        G.add_edge(words[i], words[j], weight=cosine_sim[i, j])
            return G

        # 커뮤니티 탐지 및 아이겐벡터 계산
        def detect_communities_and_eigenvector(G, resolution=0.9):
            import networkx.algorithms.community as nx_comm
            # resolution 파라미터를 조절하여 커뮤니티의 수를 조정
            communities = nx_comm.louvain_communities(G, resolution=resolution)
            eigenvector = nx.eigenvector_centrality_numpy(G)
            return communities, eigenvector

        # 네트워크 시각화
        def visualize_network(G, eigenvector, communities, cluster_id):
            pos = nx.spring_layout(G, seed=42)
            colors = list(mcolors.TABLEAU_COLORS.keys())
            community_color = {node: colors[i % len(colors)] for i, comm in enumerate(communities) for node in comm}
            base_size = 10000
            scale_factor = 2
            node_size = [eigenvector[node] * base_size * (scale_factor if eigenvector[node] > np.median(list(eigenvector.values())) else 1) for node in G]
            labels = {node: node if eigenvector[node] > np.median(list(eigenvector.values())) else '' for node in G}
            plt.figure(figsize=(20, 20))
            nx.draw_networkx(G, pos, node_color=[community_color[node] for node in G],
                            node_size=node_size, labels=labels, font_size=7, with_labels=True, edge_color='gray', font_family='Malgun Gothic')
            plt.title(f'Network Graph for Cluster {cluster_id}')
            plt.axis('off')
            st.pyplot(plt)

        # 클러스터별 처리 및 시각화 실행
        for cluster_id, group in clusters:
            st.subheader(f"Cluster {cluster_id}")
            texts = group['text'].values
            G = create_network(texts)
            communities, eigenvector = detect_communities_and_eigenvector(G)
            visualize_network(G, eigenvector, communities, cluster_id)

        # 클러스터별 처리 및 주요 키워드 데이터프레임 생성 및 저장
        for cluster_id, group in clusters:
            st.subheader(f"Cluster {cluster_id}")
            texts = group['text'].values
            G = create_network(texts)
            communities, eigenvector = detect_communities_and_eigenvector(G)
            # 각 커뮤니티별 데이터 프레임 생성 및 저장
            for i, community in enumerate(communities):
                community_words = {word: eigenvector[word] for word in community}
                community_df = pd.DataFrame(list(community_words.items()), columns=['Word', 'Eigenvector'])
                # 파일명 지정 (클러스터와 커뮤니티 번호 포함)
                filename = f'cluster_{cluster_id}_Actor_{i}_keywords.csv'
                # 데이터 프레임을 CSV 파일로 저장
                community_df.to_csv(filename, index=False)
                st.write(f'Data for Cluster {cluster_id}, Community {i} saved to {filename}.')
                # 데이터 프레임 표시
                st.dataframe(community_df)
                
        # 네트워크 생성 및 커뮤니티 탐지 함수
        def create_network_and_detect_communities(texts):
            vectorizer = TfidfVectorizer(tokenizer=extract_nouns)
            tfidf_matrix = vectorizer.fit_transform(texts)
            cosine_sim = cosine_similarity(tfidf_matrix)
            words = vectorizer.get_feature_names_out()
            G = nx.Graph()
            n = min(len(words), cosine_sim.shape[0])
            for i in range(n):
                for j in range(i + 1, n):
                    if cosine_sim[i, j] > 0.1:
                        G.add_edge(words[i], words[j], weight=cosine_sim[i, j])
            communities = nx.algorithms.community.louvain_communities(G, resolution=1)
            return communities, words, G

        # 클러스터별 커뮤니티 데이터프레임 생성 및 저장
        cluster_community_dataframes = defaultdict(list)
        for cluster_id, group in clusters:
            st.subheader(f"Processing Cluster {cluster_id}")
            texts = group['text'].values
            communities, words, G = create_network_and_detect_communities(texts)
            for idx, community in enumerate(communities):
                pattern = '|'.join([f"\\b{word}\\b" for word in community if word in words])
                community_df = group[group['text'].str.contains(pattern, regex=True)]
                community_df = community_df.reset_index(drop=True)
                filename = f"cluster_{cluster_id}_Actor_{idx}.csv"
                community_df.to_csv(filename, index=False)
                cluster_community_dataframes[cluster_id].append(community_df)
                st.write(f'Actor {idx} Data for Cluster {cluster_id} saved to {filename}.')
                st.dataframe(community_df)

elif menu == '📉토픽 모델링':
    st.image('https://brassy-dolphin-f22.notion.site/image/https%3A%2F%2Fprod-files-secure.s3.us-west-2.amazonaws.com%2Fa60d437f-ec9f-4f91-a164-6f05ff0d8852%2Fcefff13e-14df-4394-8bca-6a8ab9f58534%2FUntitled.png?table=block&id=06bd4623-7c03-4811-b2c1-d6c3d1657b05&spaceId=a60d437f-ec9f-4f91-a164-6f05ff0d8852&width=1390&userId=&cache=v2', use_column_width=True)
    st.image('https://brassy-dolphin-f22.notion.site/image/https%3A%2F%2Fprod-files-secure.s3.us-west-2.amazonaws.com%2Fa60d437f-ec9f-4f91-a164-6f05ff0d8852%2F114f8151-802b-4f9b-a95e-98072a74a92a%2FUntitled.png?table=block&id=c8cf6409-bde9-458a-999a-6c09213d3fc0&spaceId=a60d437f-ec9f-4f91-a164-6f05ff0d8852&width=1290&userId=&cache=v2', width=900)
   
 
    # Set Korean font
    @st.cache_data
    def set_korean_font():
        nanum_gothic_font_path = 'C:/WINDOWS/Fonts/NGULIM.TTF'
        font_name = fm.FontProperties(fname=nanum_gothic_font_path).get_name()
        plt.rc('font', family=font_name)

    set_korean_font()

    # Function to extract nouns
    def extract_nouns(text):
        okt = Okt()
        return okt.nouns(text)

    # Function to perform LDA
    def lda_modeling(noun_texts, random_seed=42):
        dictionary = corpora.Dictionary(noun_texts)
        if len(dictionary) == 0:
            return None, None, None
        corpus = [dictionary.doc2bow(text) for text in noun_texts]
        coherence_values = []
        for num_topics in range(2, 8):
            model = models.LdaModel(corpus, num_topics=num_topics, id2word=dictionary, passes=10, random_state=random_seed)
            coherence_model = CoherenceModel(model=model, texts=noun_texts, dictionary=dictionary, coherence='c_v')
            coherence_values.append(coherence_model.get_coherence())
        if coherence_values:
            optimal_num_topics = np.argmax(coherence_values) + 2
            model = models.LdaModel(corpus, num_topics=optimal_num_topics, id2word=dictionary, passes=10, random_state=random_seed)
            return model, corpus, dictionary
        return None, None, None

    def main():
        st.title("Topic Modeling and Visualization")

        uploaded_file = st.file_uploader("Choose a file", type=['csv'])
        
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            df['nouns'] = df['text'].apply(extract_nouns)
            model, corpus, dictionary = lda_modeling(df['nouns'])
            if model and corpus and dictionary:
                lda_display = gensimvis.prepare(model, corpus, dictionary, sort_topics=False)
                lda_html = pyLDAvis.prepared_data_to_html(lda_display)
                components.html(lda_html, height=800)  # Use Streamlit components to render HTML
                
                # Save topic data and keywords
                for topic_id in range(model.num_topics):
                    words = model.show_topic(topic_id, topn=10)
                    topic_df = pd.DataFrame(words, columns=['keyword', 'weight'])
                    st.download_button(label=f"Download keywords for Topic {topic_id}", data=topic_df.to_csv(index=False), file_name=f"topic_{topic_id}_keywords.csv", mime='text/csv')
            else:
                st.error("Insufficient data for LDA")

    if __name__ == "__main__":
        main()

     
            
elif menu == '💻기회점수':
    st.image('https://brassy-dolphin-f22.notion.site/image/https%3A%2F%2Fprod-files-secure.s3.us-west-2.amazonaws.com%2Fa60d437f-ec9f-4f91-a164-6f05ff0d8852%2F480fb0aa-9cc7-4554-b6ec-28d5821c5bbc%2FUntitled.png?table=block&id=fb560845-d5e5-45c7-b140-9f8b5b007ca1&spaceId=a60d437f-ec9f-4f91-a164-6f05ff0d8852&width=1390&userId=&cache=v2', use_column_width=True)
   
    # st.title('기회점수')
    
# 게시판 부분(커뮤니티)
elif menu == "🙎🏻DGB CAS 커뮤니티":
    st.image('https://brassy-dolphin-f22.notion.site/image/https%3A%2F%2Fprod-files-secure.s3.us-west-2.amazonaws.com%2Fa60d437f-ec9f-4f91-a164-6f05ff0d8852%2F45af589f-3fc4-475a-82bb-e1c1a4d6a067%2FUntitled.png?table=block&id=99ceb2ea-bf6a-484c-b3b8-381c352b6e49&spaceId=a60d437f-ec9f-4f91-a164-6f05ff0d8852&width=870&userId=&cache=v2',use_column_width=True)
    
    #
    # st.title('게시판')
    post_title = st.text_input('제목')
    post_content = st.text_area('내용', height=250)
    uploaded_file = st.file_uploader("파일을 첨부하세요.", type=['jpg', 'png', 'pdf', 'csv', 'xlsx'])
    submit_button = st.button('게시하기')

    if submit_button:
        # 파일명 처리
        file_name = uploaded_file.name if uploaded_file is not None else "파일 없음"
        # 세션 상태에 게시글 정보 추가
        st.session_state['posts'].append({'제목': post_title, '내용': post_content, '파일명': file_name})
        st.success('게시글이 성공적으로 올라갔습니다!')

    # 게시글 출력
    for i, post in enumerate(st.session_state['posts']):
        st.write(f"제목: {post['제목']}")
        st.write(f"내용: {post['내용']}")
        st.write(f"첨부파일: {post['파일명']}")
        if st.button('삭제', key=f'delete_{i}'):
            # 해당 게시글 삭제
            del st.session_state['posts'][i]
            st.experimental_rerun()  # 화면 새로고침

   