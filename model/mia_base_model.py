import torch
import torch.nn as nn
from configs.mia_globals import (
    MIA_EPSILON,
    MIA_GELU_MAGIC_FACTOR,
    MIA_GELU_MAGIC_SUMMAND
)
from .mia_attention_mechanism import MiaMultiHeadAttention


class MIABaseModel(nn.Module):
    """
    Transformer based decoder architecture for MIA Language Base Model. MIA implements only the
    decoder part of tranformer architecture. 
    """

    def __init__(self, mia_config: dict) -> None:
        """
        Constructor, initialize mia llm base model architecture based on the provided configuration.

        Parameters:
            mia_config (dict): A dictionary that contains mia llm architecture configuration parameters.
        Returns:
            None    
        """
        super().__init__()
        self.embeddings_layer = nn.Embedding(mia_config["mia_vocab_size"], mia_config["mia_emb_dim"])
        self.posencoding_layer = nn.Embedding(mia_config["mia_context_length"], mia_config["mia_emb_dim"])
        self.dropout_layer = nn.Dropout(mia_config["mia_dropout_rate"])
        self.decoder = nn.Sequential(
            *[MiaTransformerDecoderBlock(mia_config)
            for _ in range(mia_config["mia_total_decoder_layers"])]
        )
        self.normalization_layer = MiaNormalizationLayer(mia_config["mia_emb_dim"])
        self.output_layer = nn.Linear(mia_config["mia_emb_dim"], mia_config["mia_vocab_size"], bias = False)

    def forward(self, input_idx):
        """
        implements the logig of a forward pass through all architecture layers in mia llm. Basicly this follow 
        the transformer decoder architecture.
        """
        batch_size, sequence_length = input_idx.shape
        embeddings_tokens = self.embeddings_layer(input_idx)
        positional_encodigns = self.posencoding_layer(torch.arange(sequence_length, device=input_idx.device))
        x = embeddings_tokens + positional_encodigns
        x = self.dropout_layer(x)
        x = self.decoder(x)
        x = self.normalization_layer(x)
        logits = self.output_layer(x)

        return logits

class MiaTransformerDecoderBlock(nn.Module):
    def __init__(self, mia_config: dict) -> None:
        super().__init__()
        self.multi_head_attention = MiaMultiHeadAttention(
            input_dim = mia_config["mia_emb_dim"],
            output_dim = mia_config["mia_emb_dim"],
            context_length = mia_config["mia_context_length"],
            total_heads = mia_config["mia_total_attention_heads"],
            dropout_rate = mia_config["mia_dropout_rate"], 
            qkv_bias = mia_config["mia_qkv_bias"]
        )
        self.ffn_layer = MiaFeedForward(mia_config)
        self.first_normalization_layer = MiaNormalizationLayer(mia_config["mia_emb_dim"])
        self.second_normalization_layer = MiaNormalizationLayer(mia_config["mia_emb_dim"])
        self.shortcut_layer = nn.Dropout(mia_config["mia_dropout_rate"])

    def forward(self, input_x):
        shortcut_connection = input_x

        input_x = self.first_normalization_layer(input_x)
        input_x = self.multi_head_attention(input_x)
        input_x = self.shortcut_layer(input_x)

        input_x = input_x + shortcut_connection

        shortcut_connection = input_x
        input_x = self.second_normalization_layer(input_x)
        input_x = self.ffn_layer(input_x)
        input_x = self.shortcut_layer(input_x)
        input_x = input_x + shortcut_connection

        return input_x            

class MiaNormalizationLayer(nn.Module):
    def __init__(self, mia_emb_dim):
        super().__init__()
        self.epsilon = MIA_EPSILON
        self.scale = nn.Parameter(torch.ones(mia_emb_dim))
        self.shift = nn.Parameter(torch.zeros(mia_emb_dim))

    def forward(self, input_x):
        mean = input_x.mean(dim=-1, keepdim=True)
        variance = input_x.var(dim=-1, keepdim=True, unbiased=False)
        normalized_input_x = (input_x - mean) / torch.sqrt(variance + self.epsilon)

        return self.scale * normalized_input_x + self.shift

class MiaGeluActivationFunction(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, input_x):
        return MIA_GELU_MAGIC_FACTOR * input_x * (1 + torch.tanh(
            torch.sqrt(torch.tensor(2.0 / torch.pi)) *
            (input_x + MIA_GELU_MAGIC_SUMMAND * torch.pow(input_x, 3))
        ))

class MiaFeedForward(nn.Module):
    def __init__(self, mia_config: dict) -> None:
        super().__init__()
        self.ffn_layers = nn.Sequential(
            nn.Linear( mia_config["mia_emb_dim"], 4 * mia_config["mia_emb_dim"] ),
            MiaGeluActivationFunction(),
            nn.Linear(4 * mia_config["mia_emb_dim"], mia_config["mia_emb_dim"])
        )

    def forward(self, input_x):
        return self.ffn_layers(input_x)                                       