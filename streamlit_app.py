import streamlit as st
from PIL import Image
from src.model import get_outputs
from src.plots import plot_all_attentions, plot_attention, plot_top_k, is_mask, remove_special_tokens
import numpy as np


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

    with c1:
        st.write(" ")
        st.write(" ")
        st.image(
            "src/assets/android-chrome-192x192.png",
            width=60,
        )

    with c2:
        st.title("DistilBERT Attention Weights")

    # Add a text input box to the sidebar with default text
    default_text = 'Attention is [MASK] you need.'

    st.sidebar.title('Controls:')
    with st.sidebar.form(key='input'):
        text_input = st.text_input(label='Input sentence:', value=default_text, max_chars=200)
        submit_button = st.form_submit_button(label='Submit')

    with st.sidebar.container(border=True):
        st.write('Special Tokens:')
        include_special_tokens = st.checkbox('Include Special Tokens')

    with st.sidebar.container(border=True):
        # Dropdowns to select layer and head
        num_layers = 6
        num_heads = 12
        st.write('Attention Head plot settings:')
        layer = st.selectbox("Select layer:", range(1, num_layers + 1)) - 1
        head = st.selectbox("Select head:", range(1, num_heads + 1)) - 1
        attn_fig_size = st.slider('Attention figure size', min_value=1.0, max_value=6.0, step=0.1, value=2.0)

    with st.sidebar.container(border=True):
        st.write('TopK Head plot settings:')
        topk_fig_size = st.slider('TopK figure size', min_value=1.0, max_value=6.0, step=0.1, value=2.0)

    outputs = get_outputs(text_input)
    special_tokens_removed, mask = remove_special_tokens(outputs)
    if outputs is not None and np.any(mask):
        # Plot all attentions
        st.subheader("Max Attention Weights:")
        plot_all_attentions(outputs, include_special_tokens=include_special_tokens)

        # Plot attention for the selected layer and head
        st.subheader(f"Attention for Layer {layer + 1}, Head {head + 1}:")
        plot_attention(outputs, layer, head, include_special_tokens=include_special_tokens, figsize=attn_fig_size)

        if is_mask(outputs):
            st.subheader("TopK predictions for [MASK] tokens:")
            plot_top_k(outputs, figsize=topk_fig_size)
    else:
        st.error("Failed to get outputs. Please check your input and try again.")


if __name__ == "__main__":
    main()
