import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image


def main():
    icon = Image.open('src/assets/favicon.ico')
    st.set_page_config(
        page_title='BERTViz',
        page_icon=icon,
        layout='centered',
        menu_items={
            'Get Help': 'https://github.com/insdout/BertAttentionViz',
            'About': 'The app to visualize BERT attention weights.'}
    )

    c1, c2 = st.columns([0.32, 2])

    # The snowflake logo will be displayed in the first column, on the left.

    # The snowflake logo will be displayed in the first column, on the left.
    with c1:
        st.write(" ")
        st.write(" ")
        st.image(
            "src/assets/android-chrome-192x192.png",
            width=60,
        )

    # The heading will be on the right.
    with c2:
        st.title("Bert Attention Weight Visualizer")

    # Add a text input box to the sidebar with default text
    default_text = 'Attention is [mask] you need.'
    #input_text = st.sidebar.text_input('Enter text', default_text)

    with st.sidebar.form(key='input'):
        text_input = st.text_input(label='Enter some text', value=default_text)
        submit_button = st.form_submit_button(label='Submit')


    st.write(text_input)


if __name__ == "__main__":
    main()
