# Learning to Respond to Reddit Writing Prompts from Human Feedback
> This is a project where I attempted to train transformer language models with reinforcement learning to respond to [Reddit Writing Prompts](https://reddit.com/r/writingprompts) and learn human preferences for better writing prompt responses.

## High-level Overview of Reinforcement Learning from Human Feedback (RLHF)
The idea is to train a reward model to pick up on human preferences for completing a certain task (e.g. summarizing text, building a tower in minecraft, driving on an obstacle course). We do this by having humans pick which completions of a task they prefer (e.g. which summaries are best, which towers are best, which driving paths are best). We then subsequently train a policy model to optimize the reward outputted by that reward model. We can then use the outputs of the policy model to generate more data for human feedback to evaluate, and thus more data for our reward model to update on, and repeat the process as many times as we'd like.

## Applying RLHF to Language Models
Fine-tuning a language model via RLHF  consists of roughly three steps:

1. **Rollout**: The language model generates a response or continuation based on query which could be the start of a sentence.
2. **Evaluation**: The query and response are evaluated with a function, model, human feedback or some combination of them. The important thing is that this process should yield a scalar value for each query/response pair.
3. **Optimization**: This is the most complex part. In the optimisation step the query/response pairs are used to calculate the log-probabilities of the tokens in the sequences. This is done with the model that is trained and and a reference model, which is usually the pre-trained model before fine-tuning. The KL-divergence between the two outputs is used as an additional reward signal to make sure the generated responses don't deviate to far from the reference language model. The active language model is then trained with PPO (Proximal-Policy Optimization).

## Motivation for this project
Before I started this project, I didn't have a strong opinion on the likelihood of RLHF being an important part of a possible solution to alignment, but I did think that at the very least it was likely to be a [productive mistake](https://www.lesswrong.com/posts/ADMWDDKGgivgghxWf/productive-mistakes-not-perfect-answers) - I tend to think this is the case for most work involving attempting to [align narrowly-superhuman models](https://www.lesswrong.com/posts/PZtsoaoSLpKjjbMqM/the-case-for-aligning-narrowly-superhuman-models). I figured it would be worthwhile trying to understand this approach more deeply since it seems to be an essential component of work being done at most of the leading alignment organizations, and that they best way of understanding it better would be to get my hands dirty. It's possible I should have just stuck to a more straightforward reimplementation of one of the existing papers utilizing this approach, but I thought it might be interesting to see how the method worked for more "subjective" tasks like creative writing, compared to e.g. summarization.

## Notebooks
There are 6 notebooks associated with this project (in the [`nbs`](https://github.com/anshradh/trl_custom/tree/master/nbs) directory):
1. `00-core.ipynb` is mostly just helper functions (some of which were transferred from the `trl` repo this project was forked from and are no longer relevant).
2. `01-model-with-value-head.ipynb` is mostly forked from `trl` and is meant to allow a "value head" to be attached on top of a language model (for PPO purposes), as well as an associated helper function to generate language model outputs.
3. `02-ppo.ipynb` is mostly forked from `trl`, with the addition of a separate "value model" that is optimized alongside the "policy model", instead of having updates to one network ruin the other (if they were combined). This was an update to the approach taken in the original **"Fine-Tuning Language Models from Human Preferences"** that **"Learning to Summarize from Human Feedback"** introduced.
4. `03-writing-prompt-reward-model-training.ipynb` trains a reward model to predict which responses to a r/writingprompts prompt are likely to have the highest comment score (and are thus predicted to be the "best" according to humans).
5. `04-writing-prompt-supervised-baseline-training.ipynb` takes a "behavioral cloning" approach to the task and simply fine-tunes a language model on an r/writingprompts dataset to generate responses to writing prompts.
6. Finally, `05-writing-prompts-rlhf.ipynb` puts everything together and uses PPO to train a policy model (starting off with the supervised baseline from the previous notebook) against the reward model trained earlier.

## Links to Trained Models on Huggingface Hub
[`distilgpt2 Reward Model`](https://huggingface.co/anshr/distilgpt2_reward_model_final)
[`distilgpt2 Supervised Learning Model`](https://huggingface.co/anshr/distilgpt2_supervised_model_final)
[`distilgpt2 RL Trained Policy Model`](https://huggingface.co/anshr/distilgpt2_trained_policy_model_final)

## Example Comparison of Outputs from Fine-Tuned Model and Trained Policy Model:
**Prompt:** 

Writing Prompt: The demon you conjured up 20 years ago is at your front door. She says that her and Lucifer got in a fight and she needs a place to stay for awhile. Response:

**Reference Model Fine-Tuned on Writing Prompt Dataset Output:**

The dust has failed me, another!

The here is a mental health lawsuit.Emotion to you suddenly get this pretty easy and not life saving the day. This was it however, was won!

11:17am

A poster has surfaced via 



\-----------


Anyways, go check out the lineup of treats and tools you need to know to stop your serious daughter

You nevermind and lose it.

"Shit." repeated Carol Gates, one mile from me

I saw a reporter at hundreds of pairs of hands tightened their shoulders, crafted the human body easier for the next 10 minutes while holding their breath.

U. Hulk

I wasn't about to protest!

WE ARE ABUSE OF ACCOUNTED

&# begin streaming pirating

Today is the last few days of retroactive to release link to file attachments most necessary

The traveler train enters view re: what

**RL Trained Policy Model Output:**

bottom-famyan drawback 2 minutes late!
The leader behind these lists are the current updates

&#

This is my third time telling me about some of the emerging traits that make, life of gods and fallen angels. You are not welcome

Ignoring the toothfalls people use to grow hinds of the oppressors who are annoying them to you around everyday. A simple test, all you are seeing is passersmeters and the stars best to her.

You shall fail.


It's a must be pragmatic move. There is a lot of time, I've maybe some Ted Hayes and I teamed up again for the LiveJournal just a couple of days ago.

This mission statement.

First time posting.

I see what you are thinking is my favorite way to take a turn coming this weekend, my name is Maurice Goodwin.

Ok, I need some extra provocation.

The first generation of humans, not needing

## Directions for Future Work
The most straightforward extensions involve simply turning the compute crank (larger models, larger batches of data, more training steps, etc.) and/or collecting more feedback from humans and iterating the reward and policy model training steps. I'd also like to come up with a more objective metric of writing quality, though I'm currently unsure what that might be.

Another direction to go in might be to train on different forms of feedback, rather than purely binary comparisons. It seems like both OpenAI and Deepmind have been heading in the direction of training their language models on natural language feedback (sometimes generated by other language models!) to their outputs. I think in general having a more pluralistic approach to reward modeling/learning seems very promising to me, since it seems unlikely that one particular way of providing human feedback will scale for all tasks (see examples of different kinds of feedback that could be provided to learn a reward model and a unifying formalism for this kind of work [here](https://arxiv.org/abs/2002.04833)).

I think it might also be interesting to look into ways to make the reward model more interpretable/decomposable. We might (and should) be worried that we may learn a misaligned reward function based off of the human feedback we receive, since [humans can be assigned any values whatsoever](https://www.lesswrong.com/posts/ANupXf8XfZo2EJxGv/humans-can-be-assigned-any-values-whatsoever). The straightforward ways I can think of getting around this involve digging into our reward models with interpretability tools or attempting to [recursively decompose our rewards](https://arxiv.org/abs/1811.07871) for tasks into simpler and simpler pieces until we're sure that each component is aligned. It might be useful to break down what aspects of a writing prompt response are relevant for its evaluation and see how we might generate a transparent and aligned reward model for the task of creative writing.

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
