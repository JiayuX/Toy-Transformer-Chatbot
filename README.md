# Toy-Transformer-Chatbot

In this project, we handcraft a transformer model from scratch using PyTorch. Transformer model is based on the idea of 'self-attention' which allows it to extract a very rich contextual representation of the words in the sentences. Those vectorial representations are then used in different NLP tasks. Soon after the invention of the original transformer architecture, BERT and GPT models were derived from the transformer encoder or decoder for transfer learning in various NLP tasks. The original transformer containing a encoder and a decoder was for sequence-to-sequence (seq2seq) tasks like language translation. Here we apply the original transformer achitecture to build another seq2seq model, a toy conversational chatbot!

Due to the limitation of the computational resource on my local laptop, I only use a few sentences to train the model to demonstrate how the model is trained and used to make predictions. A more intelligent chatbot would require a much bigger corpus as the training data.

Here is the training curve:
<img src="https://raw.githubusercontent.com/JiayuX/Toy-Transformer-Chatbot/main/1.png" width="350"/>

Here are some sample conversions with the trained chatbot:
<img src="https://raw.githubusercontent.com/JiayuX/Toy-Transformer-Chatbot/main/2.png" width="600"/>

