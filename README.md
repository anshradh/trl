# Learning to Respond to Reddit Writing Prompts from Human Feedback
> This is a project where I attempted to train transformer language models with reinforcement learning to respond to [Reddit Writing Prompts](https://reddit.com/r/writingprompts) and learn human preferences for better writing prompt responses.

## High-level Overview of Reinforcement Learning from Human Feedback (RLHF)
The idea is to train a reward model to pick up on human preferences for completing a certain task (e.g. summarizing text, building a tower in minecraft, driving on an obstacle course). We do this by having humans pick which completions of a task they prefer (e.g. which summaries are best, which towers are best, which driving paths are best). We then subsequently train a policy model to optimize the reward outputted by that reward model. We can then use the outputs of the policy model to generate more data for human feedback to evaluate, and thus more data for our reward model to update on, and repeat the process as many times as we'd like.

## Applying RLHF to Language Models
Fine-tuning a language model via RLHF  consists of roughly three steps:

1. **Rollout**: The language model generates a response or continuation based on query which could be the start of a sentence.
2. **Evaluation**: The query and response are evaluated with a function, model, human feedback or some combination of them. The important thing is that this process should yield a scalar value for each query/response pair.
3. **Optimization**: This is the most complex part. In the optimisation step the query/response pairs are used to calculate the log-probabilities of the tokens in the sequences. This is done with the model that is trained and and a reference model, which is usually the pre-trained model before fine-tuning. The KL-divergence between the two outputs is used as an additional reward signal to make sure the generated responses don't deviate to far from the reference language model. The active language model is then trained with PPO (Proximal-Policy Optimization).


## Notebooks
There are 6 notebooks associated with this project:
1. `00-core.ipynb` is mostly just helper functions (some of which were transferred from the `trl` repo this project was forked from and are no longer relevant).
2. `01-model-with-value-head.ipynb` is mostly forked from `trl` and is meant to allow a "value head" to be attached on top of a language model (for PPO purposes), as well as an associated helper function to generate language model outputs.
3. `02-ppo.ipynb` is mostly forked from `trl`, with the addition of a separate "value model" that is optimized alongside the "policy model", instead of having updates to one network ruin the other (if they were combined). This was an update to the approach taken in the original **"Fine-Tuning Language Models from Human Preferences"** that **"Learning to Summarize from Human Feedback"** introduced.
4. `03-writing-prompt-reward-model-training.ipynb` trains a reward model to predict which responses to a r/writingprompts prompt are likely to have the highest comment score (and are thus predicted to be the "best" according to humans).
5. `04-writing-prompt-supervised-baseline-training.ipynb` takes a "behavioral cloning" approach to the task and simply fine-tunes a language model on an r/writingprompts dataset to generate responses to writing prompts.
6. Finally, `05-writing-prompts-rlhf.ipynb` puts everything together and uses PPO to train a policy model (starting off with the supervised baseline from the previous notebook) against the reward model trained earlier.

## References

### Proximal Policy Optimisation
The PPO implementation largely follows the structure introduced in the paper **"Fine-Tuning Language Models from Human Preferences"** by D. Ziegler et al. \[[paper](https://arxiv.org/pdf/1909.08593.pdf), [code](https://github.com/openai/lm-human-preferences)].

### Learning to Summarize from Human Feedback
The results from this paper most directly inspired the approach taken in this project.
\[[paper](https://arxiv.org/abs/2009.01325)]

### Language models
The language models utilize the `transformer` library by 🤗Hugging Face.

### trl
The [`trl` library](https://github.com/lvwerra/trl) by lverra was immensely helpful for a lot of the code that this project was based on. 

### Dataset
The [rewardsignal/reddit_writing_prompts](https://huggingface.co/datasets/rewardsignal/reddit_writing_prompts) dataset was used for this project.
