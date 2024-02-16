import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
from src.model import get_outputs
from src.plots import plot_all_attentions, plot_attention, plot_top_k, is_mask


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
    default_text = 'Attention is [MASK] you need.'

    with st.sidebar.form(key='input'):
        text_input = st.text_input(label='Enter some text', value=default_text, max_chars=500)
        submit_button = st.form_submit_button(label='Submit')
    

    outputs = get_outputs(text_input)
    # Plot all attentions
    st.subheader("Max Attention Weight")
    plot_all_attentions(outputs)

    # Dropdowns to select layer and head
    num_layers = 6
    num_heads = 12

    c21, c22 = st.columns([1, 1])

    with c21:
        layer = st.selectbox("Select layer:", range(num_layers))
        include_special_tokens = st.checkbox('Include Special Tokens')
    with c22:
        head = st.selectbox("Select head:", range(num_heads))
        attn_fig_size = st.slider('Figure size', min_value=1, max_value=6, step=1, value=3)

    # Plot attention for the selected layer and head
    st.subheader(f"Attention for Layer {layer}, Head {head}")
    plot_attention(outputs, layer, head, include_special_tokens=include_special_tokens, figsize=attn_fig_size)
    
    if is_mask(outputs):
        c31, c32 = st.columns([1, 1])
        with c31:
            top_k_fig_size = st.slider('TopK figure size', min_value=1, max_value=6, step=1, value=3)
        plot_top_k(outputs, figsize=top_k_fig_size)
    st.write(text_input)


if __name__ == "__main__":
    main()
