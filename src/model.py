from transformers import pipeline
import torch
import streamlit as st

class PipelineWrapper:
    """
    Wrapper class for Hugging Face Transformers pipeline.
    """

    def __init__(self, pipeline):
        """
        Initialize the PipelineWrapper instance.

        Args:
            pipeline (pipeline): Hugging Face Transformers pipeline instance.
        """
        self.pipeline = pipeline
        self.tokenizer = pipeline.tokenizer
        self.model = pipeline.model.distilbert
        self.clf = torch.nn.Sequential(
            pipeline.model.vocab_transform,
            pipeline.model.vocab_layer_norm,
            pipeline.model.vocab_projector
        )
        self.mask_token_id = pipeline.tokenizer.mask_token_id

    def decode(self, tokens: list[int]) -> str:
        """
        Decode token IDs to text.

        Args:
            tokens (list[int]): List of token IDs to decode.

        Returns:
            str: Decoded text.
        """
        return self.tokenizer.decode(tokens)

    def encode(self, text: str) -> dict[str, torch.Tensor]:
        """
        Encode text to token IDs.

        Args:
            text (str): Input text to encode.

        Returns:
            dict[str, torch.Tensor]:
                Dictionary containing the encoded inputs as PyTorch tensors.
        """
        return self.tokenizer.encode_plus(text, return_tensors='pt')

    def basemodel_predict(
            self,
            text: str,
            output_attentions: bool = True
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        """
        Perform prediction using the base model.

        Args:
            text (str): Input text for prediction.
            output_attentions (bool, optional): Whether to output attentions.
                Defaults to True.

        Returns:
            tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
                Tuple containing outputs from the model and the encoded text.
        """
        encoded_text = self.encode(text)
        with torch.no_grad():
            outputs = self.model(
                **encoded_text,
                output_attentions=output_attentions
            )
        return outputs, encoded_text

    def mask_predict(
            self,
            text: str,
            top_k: int = 5
    ) -> dict[str,  torch.Tensor]:
        """
        Perform masked token prediction.

        Args:
            text (str): Input text containing the masked token.
            top_k (int, optional): Number of top predictions to return.
                Defaults to 5.

        Returns:
            dict[str, torch.Tensor]: Dictionary containing the predicted words,
                token IDs, and probabilities.
        """
        outputs, encoded_text = self.basemodel_predict(text)
        encoded_tokens = encoded_text['input_ids']
        input_words = [self.decode(token) for token in encoded_tokens.squeeze()]
        mask_ind = torch.nonzero(
            encoded_tokens == self.mask_token_id,
            as_tuple=True
        )

        mask_hidden = outputs['last_hidden_state'][mask_ind]
        logits = self.clf(mask_hidden)

        probs = logits.softmax(dim=-1)
        top_k_probs, top_k_ind = probs.topk(top_k)
        top_k_probs = top_k_probs.squeeze()
        top_k_ind = top_k_ind.squeeze()
        top_k_words = [self.decode(token) for token in top_k_ind]
        return {
                'top_k_words': top_k_words,
                'top_k_tokens': top_k_ind,
                'top_k_probs': top_k_probs,
                'input_tokens': encoded_tokens.squeeze(),
                'input_words': input_words,
                'outputs': outputs
                }


def get_pipeline_wrapper(
        pipeline_name: str = 'distilbert-base-uncased',
        device: str = 'cpu'
) -> PipelineWrapper:
    """
    Get a PipelineWrapper instance for the specified pipeline.

    Args:
        pipeline_name (str, optional):
            Name of the Hugging Face Transformers pipeline.
            Defaults to 'distilbert-base-uncased'.
        device (str, optional):
            Device to use for inference ('cpu' or 'gpu'). Defaults to 'cpu'.

    Returns:
        PipelineWrapper: Instance of PipelineWrapper.
    """
    # Half precision only available on gpu:
    if device == 'cpu':
        torch_dtype = torch.float32
    else:
        torch_dtype = torch.float16
    pipe = pipeline(
        'fill-mask',
        model=pipeline_name,
        device=device,
        torch_dtype=torch_dtype
    )

    return PipelineWrapper(pipe)


@st.cache_data
def get_outputs(
    input_text: str,
    model_name: str = 'distilbert-base-uncased'
) -> dict[str, torch.Tensor]:
    """
    Retrieves outputs from a masked language model pipeline given an input text.

    Args:
        input_text (str): The input text to be processed by the model.
        model_name (str, optional): The name of the pre-trained language model to use.
            Defaults to 'distilbert-base-uncased'.

    Returns:
        dict[str, torch.Tensor]: A dictionary containing the model outputs.

    Note:
        This function is decorated with `@st.cache_data` to enable caching of the
        outputs based on the input arguments. Caching allows for efficient reuse
        of results, reducing computation time for repeated function calls with
        the same input arguments.

    Raises:
        RuntimeError: If the specified model name is not available.

    Example:
        >>> outputs = get_outputs("Hello, how are you?")
    """
    pipe = get_pipeline_wrapper(model_name)
    outputs = pipe.mask_predict(input_text)
    return outputs
