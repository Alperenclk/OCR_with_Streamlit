import streamlit as st
from doctr.io import DocumentFile
from doctr.models import ocr_predictor


def ocr(item):
    model = ocr_predictor("db_resnet50", "crnn_vgg16_bn", pretrained=True)
    result = model(item)
    json_output = result.export()
    return result, json_output


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
