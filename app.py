"""
TODO: 
    - Daha cesitli bir sekilde sonuclari goster
    - hepsinin tek bir metin haline getir
    - Goruntu uzerinden bboxlari goster
    - Dockerize et
    - zamanlayici ekle
    - diger dosya tiplerini ekle
"""

import time
import warnings
import numpy as np
import streamlit as st
from threading import Timer
import matplotlib.pyplot as plt
from doctr.io import DocumentFile
from doctr.models import ocr_predictor

warnings.filterwarnings("ignore")

st.title("Image to Text App")


def putMarkdown():
    svg_code = """<svg width="100%" height="5"><line x1="0" y1="5" x2="100%" y2="5" stroke="black" stroke-width="1"/></svg>"""
    st.write(svg_code, unsafe_allow_html=True)
    st.markdown("<hr>", unsafe_allow_html=True)


def main():
    global start_time, seconds_elapsed, stop_time

    # Uploading an image file
    uploaded_file = st.file_uploader("Resim Seçin", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # start timer
        start_time = time.time()

        image = uploaded_file.read()

        model = ocr_predictor("db_resnet50", "crnn_vgg16_bn", pretrained=True)
        single_img_doc = DocumentFile.from_images(image)

        result = model(single_img_doc)
        json_output = result.export()

        # Show the results
        whole_words = []
        for block in json_output["pages"][0]["blocks"]:
            for line in block["lines"]:
                for word in line["words"]:
                    whole_words.append(word["value"])

        # Put the whole Words
        st.write(f"## Whole Words:")
        st.write(i + " " for i in whole_words)
        putMarkdown()

        for index, item in enumerate(whole_words):
            st.write(f"**Word {index}**:", item)

        # Put the markdown
        putMarkdown()

        # Show the result image
        synthetic_pages = result.synthesize()
        # new_width = 680
        # new_height = 960
        # img = np.resize(synthetic_pages[0], (new_width, new_height, 3))
        st.image(synthetic_pages, caption="Result of image")

        elapsed_time = time.time() - start_time
        putMarkdown()
        st.write(f"Successful! Passed Time: {elapsed_time:.2f} seconds")


if __name__ == "__main__":
    main()
