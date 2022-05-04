# Learning to Respond to Reddit Writing Prompts from Human Feedback
> This is a project where I attempted to train transformer language models with reinforcement learning to respond to [Reddit Writing Prompts](reddit.com/r/writingprompts) and learn human preferences for better writing prompt responses.

## High-level overview


## How it works
Fine-tuning a language model via PPO (Proximal-Policy Optimization consists of roughly three steps:

1. **Rollout**: The language model generates a response or continuation based on query which could be the start of a sentence.
2. **Evaluation**: The query and response are evaluated with a function, model, human feedback or some combination of them. The important thing is that this process should yield a scalar value for each query/response pair.
3. **Optimization**: This is the most complex part. In the optimisation step the query/response pairs are used to calculate the log-probabilities of the tokens in the sequences. This is done with the model that is trained and and a reference model, which is usually the pre-trained model before fine-tuning. The KL-divergence between the two outputs is used as an additional reward signal to make sure the generated responses don't deviate to far from the reference language model. The active language model is then trained with PPO.


## Notebooks
This library is built with `nbdev` and as such all the library code as well as examples are in Jupyter notebooks. The following list gives an overview:

- `index.ipynb`: Generates the README and the overview page.
- `00-core.ipynb`: Contains the utility functions used throughout the library and examples.
- `01-gpt2-with-value-head.ipynb`: Implementation of a `transformer` compatible GPT2 model with an additional value head as well as a function to generate sequences.
- `02-ppo.ipynb`: Implementation of the PPOTrainer used to train language models.
- `03-bert-imdb-training.ipynb`: Training of BERT with `simpletransformers` to classify sentiment on the IMDB dataset.
- `04-gpt2-sentiment-ppo-training.ipynb`: Fine-tune GPT2 with the BERT sentiment classifier to produce positive movie reviews.
- `05-gpt2-sentiment-control.ipynb`: Fine-tune GPT2 with the BERT sentiment classifier to produce movie reviews with controlled sentiment.

## References

### Proximal Policy Optimisation
The PPO implementation largely follows the structure introduced in the paper **"Fine-Tuning Language Models from Human Preferences"** by D. Ziegler et al. \[[paper](https://arxiv.org/pdf/1909.08593.pdf), [code](https://github.com/openai/lm-human-preferences)].

### Language models
The language models utilize the `transformer` library by 🤗Hugging Face.

### trl
The [`trl` library](https://github.com/lvwerra/trl) by lverra was immensely helpful for a lot of the code that this project was based on. 
