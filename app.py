"""
TODO: 
    - Daha cesitli bir sekilde sonuclari goster
    - hepsinin tek bir metin haline getir
    - Goruntu uzerinden bboxlari goster
    - Dockerize et
"""

import streamlit as st
from doctr.models import ocr_predictor
from doctr.io import DocumentFile


st.title("OCR App")

uploaded_file = st.file_uploader("Upload Image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = uploaded_file.read()

    model = ocr_predictor("db_resnet50", "crnn_vgg16_bn", pretrained=True)
    single_img_doc = DocumentFile.from_images(image)

    result = model(single_img_doc)
    json_output = result.export()

    # Show the results
    counter = 0
    for block in json_output["pages"][0]["blocks"]:
        for line in block["lines"]:
            counter += 1
            for word in line["words"]:
                st.write(f"### Word {counter}:", word["value"])
