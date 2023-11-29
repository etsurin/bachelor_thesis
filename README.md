# Intro
This is the repository of my bachelor thesis project. 

## Research Part
Text segmentation is one of the methods to generate summaries for long inputs. However, previous segmentation methods ignore the concrete information of the text. Moreover, there is confusion in the training samples due to the cut-off of language models. Our work researched on how to optimize text segmentation methods to generate summaries of higher-quality.

We designed an unsupervised topic segmentation algorithm of text using pre-trained GPT2, which makes divide decision based on computing the loss for the next sentence.

We did experiments by reproducing the SUMM^N framework (Now it is opensourced. https://github.com/psunlpgroup/Summ-N) and improving it by applying our topic segmentation method. 

## Development Part

### Introduction
We used PyQt to implement a summarization system. 

It has following functions:

1. Detect any txt file in user's device
2. Preview the selected txt file and generate summary for the selected device. With pretrained summarization models available at here.
3. Calculate the rouge score if reference is provided.

### Run the summarization system
```
cd system
python call_ui.py
```
Then you will see the following interfaces
