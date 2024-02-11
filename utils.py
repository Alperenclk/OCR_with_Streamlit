import json
import time
import base64
import PyPDF2
import warnings
from threading import Timer

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from doctr.io import DocumentFile
from doctr.models import ocr_predictor

warnings.filterwarnings("ignore")


def putMarkdown():
    svg_code = """<svg width="100%" height="5"><line x1="0" y1="5" x2="100%" y2="5" stroke="black" stroke-width="1"/></svg>"""
    # st.write(svg_code, unsafe_allow_html=True)
    st.markdown("<hr>", unsafe_allow_html=True)


def get_download_button(data, button_text, filename):
    # JSON verisini dosyaya yaz
    json_str = json.dumps(data, indent=4)
    b64 = base64.b64encode(json_str.encode()).decode()
    href = f'<a href="data:application/json;charset=utf-8;base64,{b64}" download="{filename}">{button_text}</a>'
    return href


def ocr(item):
    model = ocr_predictor("db_resnet50", "crnn_vgg16_bn", pretrained=True)
    result = model(item)
    json_output = result.export()
    return result, json_output



