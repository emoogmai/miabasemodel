import argparse
from training.mia_base_model_trainer import MiaBaseModelTrainer
from dataloader.mia_spanish_corpora import MiaBaseModelSpanishCorpora
from torchtext.data import to_map_style_dataset
from torch.utils.data import DataLoader
import tiktoken
import pprint

from utils import logger

def show_dataset_metadata(trainer: MiaBaseModelTrainer, split_type: str):
    total_lines = 0
    total_chars = 0
    total_tokens = 0

    spanish_corpora_iterator = MiaBaseModelSpanishCorpora(root_data_dir=trainer.config['mia_settings_data_dir'], split_type=split_type)
    spanish_corpora_iterator = to_map_style_dataset(spanish_corpora_iterator)

    #Creates a basic tokenizer - PLEASE REVIEW ME a robust one should be used instead
    tokenizer = tiktoken.get_encoding("gpt2")

    data_loader = DataLoader(
        spanish_corpora_iterator,
        batch_size=2,
        shuffle=True,
        drop_last=True)

    for batch in data_loader:
        #pprint.pprint(f"Returned batch through {split_type} data loader: {batch}")
        total_lines = total_lines + len(batch)
        all_text_batch = ''.join(batch)
        #pprint.pprint(f"Full text after joining batch items: '{all_text_batch}'")
        total_chars = total_chars + len(all_text_batch)
        total_tokens = total_tokens + len(tokenizer.encode(all_text_batch))

    print("================================================================")
    print(f"| Description of {split_type} data                              |")
    print("================================================================")
    print(f"TOTAL LINES = {total_lines}")
    print(f"TOTAL CHARS = {total_chars}")
    print(f"TOTAL TOKENS = {total_tokens}")

def estimate_total_parameters(trainer: MiaBaseModelTrainer) -> int:
    params_embedding_layer = trainer.mia_config["mia_vocab_size"] * trainer.mia_config["mia_emb_dim"]
    params_multihead_attention = 4 * (trainer.mia_config["mia_emb_dim"]**2)
    params_ffn_layer = 2 * trainer.mia_config["mia_emb_dim"] * (4 * trainer.mia_config["mia_emb_dim"]) + trainer.mia_config["mia_emb_dim"] + (4 * trainer.mia_config["mia_emb_dim"])
    params_norm_layer = 2 * trainer.mia_config["mia_emb_dim"]

    return (12 * (params_multihead_attention + params_ffn_layer + (2 * params_norm_layer)) + params_embedding_layer)


def show_miamodel_metadata(trainer: MiaBaseModelTrainer) -> None:
    vocab_size = trainer.mia_config["mia_vocab_size"]
    context_length = trainer.mia_config["mia_context_length"]
    d_model = trainer.mia_config["mia_emb_dim"]
    attention_heads = trainer.mia_config["mia_total_attention_heads"]
    n_decoders = trainer.mia_config["mia_total_decoder_layers"]

    print("================================================================")
    print(f"| MIA llm base model configurations                            |")
    print("================================================================")
    print("Architecture: Transformer based, autoregresive only with decoder")
    print(f"VOCABULARY: {vocab_size}")
    print(f"CONTEXT: {context_length}")
    print(f"MODEL DIMENSION: {d_model}")
    print(f"ATTENTION HEADS: {attention_heads}")
    print(f"DECODER LAYERS: {n_decoders}")
    print(f"ESTIMATED MIA LLM TOTAL PARAMETERS: {estimate_total_parameters(trainer)}")


def show_train_metadata(trainer: MiaBaseModelTrainer) -> None:
    train_batch_size = trainer.config["mia_settings_train_batch_size"]
    valid_batch_size = trainer.config["mia_settings_valid_batch_size"]
    suffle = trainer.config["mia_settings_shuffle"]
    stride = trainer.config["mia_settings_stride"]
    epochs = trainer.config["mia_settings_epochs"]
    learning_rate = trainer.config["mia_settings_learning_rate"]

    print("================================================================")
    print(f"| MIA llm base model trainning settings                       |")
    print("================================================================")
    print(f"TRAINING EPOCHS: {epochs}")
    print(f"LEARNING RATE: {learning_rate}")
    print(f"SUFFLE BATCHES: {suffle}")
    print(f"TRAIN BATCH SIZE: {train_batch_size}")
    print(f"VALID BATCH SIZE: {valid_batch_size}")
    print(f"WINDOW STRIDE: {stride}")


def show_mia_metadata(trainer: MiaBaseModelTrainer) -> None:
    # Prints metadata about the dataset being used to train and validate mia llm base model
    show_dataset_metadata(trainer, 'train')
    show_dataset_metadata(trainer, 'valid')

    # Print metadata about mia llm itself
    show_miamodel_metadata(trainer)

    # Print training settings for mia llm base model
    show_train_metadata(trainer)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_settings_filepath', type=str, required=True, help='Training configuration file path (path + filename)')
    parser.add_argument('--mia_config_filepath', type=str, required=True, help='MIA llm base model configuration file path (path + filename)')
    parser.add_argument('--show_mia_meta', type=str, default="y", required=False, help="Shows MIA llm base model metadata about dataset, training and model itself if set to 'y' otherwise nothing is displayed")
    args = parser.parse_args()

    trainer = MiaBaseModelTrainer(args.train_settings_filepath, args.mia_config_filepath)
     
    if args.show_mia_meta == 'y':
        show_mia_metadata(trainer) 

    trainer.train()
