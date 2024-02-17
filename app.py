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

st.title("Image to Text App")


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


def display(result, json_output, img):
    st.write("#### Downoad Json output")
    st.write("*⬇*" * 9)

    # Button of Download JSON
    download_button_str = get_download_button(json_output, "DOWNLOAD", "data.json")
    st.markdown(download_button_str, unsafe_allow_html=True)
    putMarkdown()

    # Show the result image
    st.image(img, caption="Original image")
    putMarkdown()

    synthetic_pages = result.synthesize()
    st.image(synthetic_pages, caption="Result of image")

    elapsed_time = time.time() - start_time
    putMarkdown()

    # Show the results
    whole_words = []
    per_line_words = []
    for block in json_output["pages"][0]["blocks"]:
        for line in block["lines"]:
            line_words = []
            for word in line["words"]:
                whole_words.append(word["value"])
                line_words.append(word["value"])
            per_line_words.append(line_words)

    # Put the whole Words
    st.write(f"## Whole Words:")
    st.write(word + " " for word in whole_words)
    putMarkdown()

    # Put the Words line by line
    st.write(f"## Line by Line:")
    for lineWords in per_line_words:
        st.write(word + " " for word in lineWords)
    putMarkdown()

    # Put the Words Word by Word
    st.write(f"## Word by Word:")
    for index, item in enumerate(whole_words):
        st.write(f"**Word {index}**:", item)
    putMarkdown()

    st.write(f"Successful! Passed Time: {elapsed_time:.2f} seconds")


def main():
    global start_time, seconds_elapsed, stop_time

    # Uploading an image file
    uploaded_file = st.file_uploader(
        "Choose a File", type=["jpg", "jpeg", "png", "pdf"]
    )

    st.write("#### or Put an URL")
    url = st.text_input("Please type an URL:")

    if st.button("Show The URL"):
        st.write("Typed URL:", url)
        start_time = time.time()

        single_img_doc = DocumentFile.from_url(url)
        result, json_output = ocr(single_img_doc)
        display(result, json_output)

    elif uploaded_file is not None:
        # start timer
        start_time = time.time()

        if uploaded_file.type == "application/pdf":
            image = uploaded_file.read()
            single_img_doc = DocumentFile.from_pdf(image)
        else:
            image = uploaded_file.read()
            single_img_doc = DocumentFile.from_images(image)

        result, json_output = ocr(single_img_doc)
        display(result, json_output, image)


if __name__ == "__main__":
    main()
