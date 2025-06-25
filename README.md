# MIA Large Language Model

A base implementation of MIA Language Model based on the Transformer Architecture and designed to explore how complex is to implement this kind of models, "Modelo para Inteligencias Autoregresivas". This is the code provided as part of my post - [Modelo de Lenguaje MIA](https://edwinmoo.substack.com/p/embeddings-en-el-modelo-de-lenguaje?r=233vmr).

## MIA LLM experiment details

The design objective of this architecture implementation is to allow us to experiment and document how easy or difficult can be the implementation of a Large Language Model based on Transformer Architecture, specifically for MIA Large Language Model.
It is important to note that this is only the POC for the real MIA Language Model that allows me to explore complexity about the training process, data collection etc., in addition through this experiment I was able to analyze what other routes can be taken in order to mitigate and/or eliminate the current problems that many LLMs are experiencing because the scaling approach followed for all of them.
As is the case for all the AI models I design and implement this POC provides small datasets for testing purporses, the required trainner for it and the dataloaders responsible to read the dataset in a generic way.

### MIA LLM architecture's diagram
![alt text](docs/mia_tokenizer_diagram.png)

## Project Structure

```
.
├── README.md
├── requirements.txt
├── main.py
├── configs
├── data
├── dataloader
├── model
│   └── mia_attention_mechanism.py
│   └── mia_base_model.py
├── docs
├── training
│   └── mia_base_model_trainer.py
├── utils
```

- **configs/** - Defines required configuration for MIA LLM, for instance, attention heads, number of decoder blocks, embeddings dimensions, etc. 
- **model/mia_base_model.py** - Module that defines all classes for MIA Large Language Model Architecture based on Transformer.
- **model/mia_attention_mechanism.py** - Module that implements the attention mechanism, a core component in the Transformer Architecture and of course in MIA. 
- **main.py** - Entry point to run the training process for MIA Language Model.

## How to run the training process for MIA

```
Run below commands if you environment is linux otherwise replace step 2 with this instruction 
'.\miabasemodelvenv\Scripts\activate.bat'

1.- python -m venv miabasemodelvenv
2.- source ./miabasemodelvenv/bin/activate
3.- pip install -r requirements.txt
4.- python main.py --train_settings_filepath ./configs/mia_train_settings.yml --mia_config_filepath ./configs/mia_config.yml

```

In case you want to run your own exeriments you can modify configuration files in order to adjust settings like number of attention heads, how many decoder blocks you want ot use, etc., also you can add more data to be used as source for training.

