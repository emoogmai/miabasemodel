import os
import yaml
import json
import logging
import time
import torch
import torch.optim as optim
import numpy as np
import torch.nn as nn
import tiktoken
import matplotlib.pyplot as plot
from matplotlib.ticker import MaxNLocator

from torch.optim.lr_scheduler import LambdaLR

from torchtext.data import to_map_style_dataset
from functools import partial
from torch.utils.data import DataLoader

from model.mia_base_model import MIABaseModel
from dataloader.mia_spanish_corpora import MiaBaseModelSpanishCorpora

def collate_function(batch: any, text_pipeline: any, max_length: int, stride: int) -> tuple:
    """
    This collate function is invoked by dataloader in order to process a batch,
    batch is expected to be list of text paragrahs and text pipeline is a function
    that is applied to such `batch`. Based on the cbow algorithm the context is represented
    as N past words and N future words, here we are representing N as the MIA_CONTEXT_LENGTH
    value defined as a global constant. Also, it is important to mention that this collate
    function will truncate those paragraphs longer than a threshold defined by MIA_MAXSEQUENCE_LENGTH
    constant.
    """
    input_ids = []
    target_ids = []

    all_text_batch = ''.join(batch)
    token_ids = text_pipeline(all_text_batch)

    for i in range(0, len(token_ids) - max_length, stride):
        input_chunk = token_ids[i:i + max_length]
        target_chunk = token_ids[i + 1: i + max_length + 1]
        input_ids.append(torch.tensor(input_chunk))
        target_ids.append(torch.tensor(target_chunk))

    input_ids = torch.stack(input_ids, dim=0)
    target_ids = torch.stack(target_ids, dim=0)
    
    return input_ids, target_ids    


class MiaBaseModelTrainer:
    """
    Encapsulate the mia llms base model training process, this class is invoked from main script to train mia llm base model based on a 
    small spanish corpus for experiemental purposes. This trainer reads the provided configuration file in order to define its training 
    behavior. 
    """
    def __init__(self, train_settings_filepath: str, mia_config_filepath: str) -> None:
        """
        Constructor, initialize mia llm model trainer based on the provided configuration file.

        Parameters:
            config_filepath (str): Configuration filepath (path + filename) from where the trainer will 
            take its configuration values
        Returns:
            None    
        """
        self.logger = logging.getLogger(     os.getenv("GMAI_LOGDEF", "development") + "." + __name__)
        self.loss = {"train": [], "val": [], "tokens_seen": []} # Keep track both training and validation loss during training !!!

        self.logger.info(f"Loading mia llm base model trainer configuration from {train_settings_filepath} ...")
        with open(train_settings_filepath, 'r') as tsf:
            self.config = yaml.safe_load(tsf)        

        self.logger.info(f"Loading mia llm base model configuration from {mia_config_filepath} ...")
        with open(mia_config_filepath, 'r') as mcf:
            self.mia_config = yaml.safe_load(mcf)        
    

    def train(self):
        """
        Prepare and execute the training process for mia llm base model.
        """

        self.logger.info(f"Creating mia base model output folder ...")
        os.makedirs(self.config["mia_settings_model_dir"], exist_ok=True)

        self.logger.info(f"Creating training data loader for mia llm base model ...")
        # Creates the data loader for training data
        train_loader = self.create_data_loader('train', self.config["mia_settings_train_batch_size"], self.mia_config["mia_context_length"] , self.config["mia_settings_stride"], self.config["mia_settings_shuffle"], True, 1)

        self.logger.info(f"Creating validation data loader for mia llm base model ...")
        # Creates the data loader for validation data
        valid_loader = self.create_data_loader('valid', self.config["mia_settings_valid_batch_size"], self.mia_config["mia_context_length"] , self.config["mia_settings_stride"], False, False, 1)

        self.logger.info(f"Defining model architecture and creating an instance of mia llm base model to be trained ...")
        # Creates the model
        torch.manual_seed(123)
        mia_base_model = MIABaseModel(self.mia_config)

        self.logger.info(f"Creating loss function, model optimizer and learning rate strategy ...")
        loss_function = nn.CrossEntropyLoss() 
        optimizer = optim.AdamW(mia_base_model.parameters(), lr=self.config["mia_settings_learning_rate"], weight_decay=self.config["mia_settings_weight_decay"])
        #optimizer = optim.Adam(miaEmbModel.parameters(), lr=self.config["miaemb_learning_rate"])
        #learningRateScheduler = self.createLearningRateScheduler(optimizer, self.config["miaemb_epochs"], verbose=True)

        self.logger.info(f"Checking if a gpu is availabe to assign training task to it ...")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        tokenizer = tiktoken.get_encoding("gpt2")

        # Executes the training loop once it finish we will have our embeddings ready to save !!!
        self.logger.info(f"Execute training loop using previously created object, model, training dataloader, validation dataloader, loss function, optimizer, learning rate strategy and gpu device if available ...")
        mia_base_model.to(device)
        self.execute_trainning_process(mia_base_model, train_loader, valid_loader, loss_function, optimizer, None, device, self.config["mia_settings_epochs"], 
                                        self.config["mia_settings_eval_frequency"], self.config["mia_settings_eval_iterations"],
                                        "Palabras cualquiera dice", tokenizer)

        #Using mathplotlib plot the training results to have a visual representation to evaluate how thing going on
        epochs_tensor = torch.linspace(0, self.config["mia_settings_epochs"], len(self.loss['train']))
        self.plot_training_results(epochs_tensor, self.loss['tokens_seen'], self.loss['train'], self.loss['val'])
        
        # Save all the final generated artifact results of the model training (model, loss, configuration and vocabulary).
        self.logger.info(f"Model training finished. Saving metadata for mia llm trained model, loss, configuration and trainning settings as final results ...")

        self.save_final_trained_model(mia_base_model)
        self.save_loss()
        self.save_training_config(self.config)
        self.save_mia_base_model_config(self.mia_config)
        #self.saveTrainingVocabulary(vocabulary)

        self.logger.info("MIA llm base model trained and ready to be used for inference !!!")

    def execute_trainning_process(self, model: MIABaseModel, train_loader: DataLoader, valid_loader: DataLoader, 
                                    loss_function: any, optimizer: any, lr_scheduler: any, train_device: any,
                                    epochs: int, eval_frequency: int, eval_iterations: int, start_context: any,
                                    tokenizer: any):
        """
        Execute the model's training loop based on the number of configured epochs in trainer.

        Parameters:
            model (MiaEmbeddingsModel): The embeddings model based on cbow architecture.
            trainLoader (DataLoader): The train data loader.
            validLoader (DataLoader): The valid data loader.
            lossFunction (any): The model's loss function.
            optimizer (any): The model's optimizer.
            lrScheduler (any): A learning rate scheduler that modify such value during training.
            trainDevice (any): Either a cpu or gpu device if cude is available.
        """
        t1d = time.time()
        self.logger.info(f"Training model for {epochs} epochs")
        self.logger.info(f"Model training started: {t1d}")

        for epoch in range(epochs):
            self._train_epoch(model, train_loader, loss_function, optimizer, start_context, tokenizer, eval_frequency, train_device)
            self._validate_epoch(model, valid_loader, loss_function, train_device, eval_iterations)
            print(f"Epoch: {epoch}/{epochs}, Train Loss={self.loss['train'][-1]:.3f}, Val Loss={self.loss['val'][-1]:.3f}")

            # Print a sample text after each epoch
            self.generate_sample_text(
                model, tokenizer, train_device, start_context
            )

        t2d = time.time()
        self.logger.info(f"Model training finished: {t2d}")
        self.logger.info(f"Model training total duration: {str((t2d-t1d))}")        


    def plot_training_results(self, epochs_seen: int, tokens_seen: any, train_losses: any, val_lossses: any) -> None:
        figure, axis_xy = plot.subplots(figsize=(5, 3))
        axis_xy.plot(epochs_seen, train_losses, label="Training Loss")
        axis_xy.plot(epochs_seen, val_lossses, linestyle="-.", label="Validation Loss")

        axis_xy.set_xlabel("Epochs")
        axis_xy.set_ylabel("Losses")
        axis_xy.legend(loc="upper right")

        axis_xy.xaxis.set_major_locator(MaxNLocator(integer=True))

        twin_axis_xy = axis_xy.twiny()
        twin_axis_xy.plot(tokens_seen, train_losses, alpha=0)
        twin_axis_xy.set_xlabel("Tokens Seen")

        figure.tight_layout()
        plot.show()

        plot_path = os.path.join(self.config["mia_settings_model_dir"], "mia-train-plot.png")

        plot.savefig(plot_path, bbox_inches='tight')






    def _train_epoch(self, model: MIABaseModel, train_loader: DataLoader, loss_function: any, optimizer: optim.AdamW, start_context: any, tokenizer: any, eval_frequency: int, device: any) -> None:
        """
        Train model during one epoch.

        Parameters:
            model (MiaEmbeddingsModel): Model to be trained based on cbow architecture for embeddings.
            trainLoader (DataLoader): Train data loader, responsible to load training data.
            lossFunction (any): The model's loss function to be applied.
            optimizer (any): The model's optimizer to be applied during training.
            device (any): Either cpu or gpu if cuda is available.
        """
        model.train()
        running_loss = []
        tokens_seen = -1
        global_steps = 0


        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()  # Reset loss gradients from previous batch iteration
            input_batch = input_batch.to(device)
            target_batch = target_batch.to(device)

            # Predict outputs using model and compute the loss using the provided loss function !!!
            outputs = model(input_batch) 
            loss = loss_function(outputs.flatten(0,1), target_batch.flatten())
            loss.backward()  # Calculate loss gradients
            optimizer.step()  # Update model weights using loss gradients
            tokens_seen += input_batch.numel()
            global_steps += 1

            running_loss.append(loss.item())

            if global_steps % eval_frequency == 0:
                break

        epoch_loss = np.mean(running_loss)
        self.loss["train"].append(epoch_loss)
        self.loss["tokens_seen"].append(tokens_seen)


    def generate_sample_text(self, model: MIABaseModel, tokenizer: any, device: any, start_context: any):
        model.eval()
        context_size = model.posencoding_layer.weight.shape[0]
        encoded = self.text_to_token_ids(start_context, tokenizer).to(device)
        with torch.no_grad():
            token_ids = self.generate_text(
                model=model, idx=encoded,
                max_new_tokens=50, context_size=context_size
        )
        decoded_text = self.token_ids_to_text(token_ids, tokenizer)
        print(decoded_text.replace("\n", " "))  # Compact print format
        model.train()

    def text_to_token_ids(self, text: any, tokenizer: any):
        encoded = tokenizer.encode(text)
        encoded_tensor = torch.tensor(encoded).unsqueeze(0)  # add batch dimension
        return encoded_tensor


    def token_ids_to_text(self, token_ids: any, tokenizer: any):
        flat = token_ids.squeeze(0)  # remove batch dimension
        return tokenizer.decode(flat.tolist())


    def generate_text(self, model: MIABaseModel, idx: any, max_new_tokens: any, context_size: any):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):

            # Crop current context if it exceeds the supported context size
            # E.g., if LLM supports only 5 tokens, and the context size is 10
            # then only the last 5 tokens are used as context
            idx_cond = idx[:, -context_size:]

            # Get the predictions
            with torch.no_grad():
                outputs = model(idx_cond)

            # Focus only on the last time step
            # (batch, n_token, vocab_size) becomes (batch, vocab_size)
            outputs = outputs[:, -1, :]

            # Get the idx of the vocab entry with the highest outputs value
            idx_next = torch.argmax(outputs, dim=-1, keepdim=True)  # (batch, 1)

            # Append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1)  # (batch, n_tokens+1)

        return idx



    def _validate_epoch(self, model: MIABaseModel, valid_loader: DataLoader, loss_function: any, device: any, eval_iterations: any) -> None:
        """
        Validates one epoch of training.

        Parameters:
            model (MiaEmbeddingsModel): Model to be trained based on cbow architecture for embeddings.
            validLoader (DataLoader): Train data loader, responsible to load training data.
            lossFunction (any): The model's loss function to be applied.
            device (any): Either cpu or gpu if cuda is available.
        """
        model.eval()
        running_loss = []

        with torch.no_grad():
            for i, (input_batch, target_batch) in enumerate(valid_loader, 1):
                input_batch = input_batch.to(device)
                target_batch = target_batch.to(device)

                outputs = model(input_batch)
                loss = loss_function(outputs.flatten(0,1), target_batch.flatten())

                running_loss.append(loss.item())

                if i == eval_iterations:
                    break

            epoch_loss = np.mean(running_loss)
            self.loss["val"].append(epoch_loss)


    def save_final_trained_model(self, model: MIABaseModel) -> None:
        """
        Saves the trained model in the model directory.

        Parameters:
            model (MIABaseModel): The model to be saved.

        Returns:
            None    
        """
        model_path = os.path.join(self.config["mia_settings_model_dir"], "mia-base-model.pt")
        self.logger.info(f"Saving trained model in: {model_path}")
        torch.save(model, model_path)


    def save_loss(self) -> None:
        """
        Saves the computed losses during training both validation and training loss as json file in the model's directory.
        """
        loss_path = os.path.join(self.config["mia_settings_model_dir"], "mia-base-model-loss.json")
        self.logger.info(f"Saving model losses in: {loss_path}")
        with open(loss_path, "w") as fp:
            json.dump(self.loss, fp)


    def save_training_config(self, config: dict) -> None:
        """
        Saves the applied configuration during training in the model's directory.

        Parameters:
            config (dict): Configuration dictionary.
        """
        config_path = os.path.join(config["mia_settings_model_dir"], "config.yaml")
        self.logger.info(f"Saving trained model configuration parameters in: {config_path}")
        with open(config_path, "w") as cf:
            yaml.dump(config, cf)

    def save_mia_base_model_config(self, config: dict) -> None:
        """
        Saves the mia llm base model configuration in the model's directory.

        Parameters:
            config (dict): Configuration dictionary.
        """
        config_path = os.path.join(self.config["mia_settings_model_dir"], "mia_config.yaml")
        self.logger.info(f"Saving mia llm base model configuration parameters in: {config_path}")
        with open(config_path, "w") as cf:
            yaml.dump(config, cf)
        

    def createLearningRateScheduler(self, optimizer, totalEpochs: int, verbose: bool = True):
        """
        Scheduler to linearly decrease learning rate, so that learning rate after the last epoch is 0.
        """
        lr_lambda = lambda epoch: (totalEpochs - epoch) / totalEpochs
        learningRateScheduler = LambdaLR(optimizer, lr_lambda=lr_lambda, verbose=verbose)

        return learningRateScheduler

    def create_data_loader(self, dataset_type: str, batch_size: int, max_length: int, stride: int, shuffle: bool, drop_last: bool, num_workers: int) -> tuple:
        """
        Creates a dataloader for the provided dataset type (train or valid).

        Parameters:
            dataset_type (str): Either 'train' or 'valid'
            batch_size (int): Self explanatory, batch size for dataloader.

        Returns:
            A configured dataloader.    
        """

        #Creates an iterator for spanish corpora dataset
        self.logger.info(f"Converting miaspanishcorpora dataset to a map style in order dataloader can use it to created batches for traiing purpose ...")
        spanish_corpora_iterator = MiaBaseModelSpanishCorpora(root_data_dir=self.config['mia_settings_data_dir'], split_type=dataset_type)
        spanish_corpora_iterator = to_map_style_dataset(spanish_corpora_iterator)

        #Creates a basic tokenizer - PLEASE REVIEW ME a robust one should be used instead
        self.logger.info(f"Creating a tiktoken based tokenizer ...")
        tokenizer = tiktoken.get_encoding("gpt2")

        text_pipeline = lambda x: tokenizer.encode(x, allowed_special={"<|endoftext|>"})

        self.logger.info(f"Creating data loader using spanish corpora iterator, batch_size = {batch_size}, suffle and collate function ...")
        data_loader = DataLoader(
            spanish_corpora_iterator,
            batch_size=batch_size,
            shuffle= shuffle, #self.config["mia_settings_shuffle"],
            drop_last=drop_last,
            num_workers=num_workers,
            collate_fn=partial(collate_function, text_pipeline=text_pipeline, max_length=max_length , stride=stride))
   
        self.logger.info("Returning created data loader ...")
        return data_loader

