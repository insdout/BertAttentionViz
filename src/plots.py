import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

def remove_special_tokens(outputs):
    attention_weights = np.stack([layer_weights.detach().cpu().numpy().squeeze() for layer_weights in outputs['outputs']['attentions']])
    input_tokens = outputs['input_tokens'].cpu().numpy()
    special_tokens = {101, 102}
    # Create a mask to identify special tokens
    mask = np.isin(input_tokens, list(special_tokens))
    # Negate the mask to exclude special tokens
    attention_weights_no_special = attention_weights[:, :, ~mask, :]
    attention_weights_no_special = attention_weights_no_special[:, :, :, ~mask]
    return attention_weights_no_special, ~mask


def plot_all_attentions(outputs, include_special_tokens=True):
    fig, ax = plt.subplots(figsize=(3, 2))

    if include_special_tokens:
        attention_weights = np.stack([layer_weights.detach().cpu().numpy().squeeze() for layer_weights in outputs['outputs']['attentions']])
    else:
        attention_weights, _ = remove_special_tokens(outputs)
    max_attentions = np.max(attention_weights, axis=(-1, -2))
    ax.imshow(max_attentions, cmap='Blues')
    ax.set_yticks(range(attention_weights.shape[0]), range(1, attention_weights.shape[0]+1), fontsize=5)
    ax.set_xticks(range(attention_weights.shape[1]), range(1, attention_weights.shape[1]+1), fontsize=5)
    ax.set_xlabel('Heads', fontsize=5)
    ax.set_ylabel('Layers', fontsize=5)
    ax.set_title('Max Attention Weight', fontsize=7)
    st.pyplot(fig, use_container_width=False)

def plot_attention(outputs, layer, head, include_special_tokens=True, figsize=3):
    attention_weights = np.stack([layer_weights.detach().cpu().numpy().squeeze() for layer_weights in outputs['outputs']['attentions']])[layer][head]
    words = np.array(outputs['input_words'])

    if not include_special_tokens:
        attention_weights, mask = remove_special_tokens(outputs)
        attention_weights = attention_weights[layer][head]
        words = words[mask]

    fig, ax = plt.subplots(figsize=(figsize, figsize))
    ax.imshow(attention_weights, cmap='Blues')
    ax.set_xticks(range(len(words)), words, rotation=90, ha='right', fontsize=5)
    ax.set_yticks(range(len(words)), words, rotation=0, ha='right', fontsize=5)
    st.pyplot(fig, use_container_width=False)


def is_mask(outputs):
    top_k_words = np.array(outputs['top_k_words'])
    return top_k_words.size > 0


def plot_top_k(outputs, figsize=3):
    top_k_words = np.array(outputs['top_k_words'])
    if is_mask(outputs):
        top_k_probs = outputs['top_k_probs'].detach().cpu().numpy()
        if top_k_words.ndim > 1:
            n_plots = top_k_words.shape[0]
        else:
            n_plots = 1

        fig, ax = plt.subplots(n_plots, figsize=(figsize, figsize*n_plots))
        if n_plots == 1:
            ax.barh(top_k_words[0], top_k_probs[0], align='center', color='steelblue')
            ax.set_title(f'For [MASK] 0', fontsize=6)
            ax.tick_params(axis='both', which='major', labelsize=5)
        else:
            for i in range(n_plots):
                ax[i].barh(top_k_words[i], top_k_probs[i], align='center', color='steelblue')
                ax[i].set_title(f'For [MASK] {i}', fontsize=6)
                ax[i].tick_params(axis='both', which='major', labelsize=5)
        plt.subplots_adjust(hspace=0.5)  # Increase vertical space between subplots
        st.pyplot(fig, use_container_width=False)
