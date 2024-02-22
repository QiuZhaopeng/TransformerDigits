# TransformerDigits

This repo demonstrates translation from digits to french using Transformer

## Introduction

This repo showcases how a transformer model translate a digital integer to french words. As a popular seq-to-seq model, transformer as well as its variants are used in more and more fields. One can read the famous paper ["Attention is all you need"](https://papers.neurips.cc/paper/7181-attention-is-all-you-need.pdf) for more knowledge about it.

The vocabulary in this application is rather small:
![Image description](https://dev-to-uploads.s3.amazonaws.com/uploads/articles/londcorj05z9k842dc3x.png)

Below is an example how a interger and its translation tokens are coded as sequences:
![Image description](https://dev-to-uploads.s3.amazonaws.com/uploads/articles/iaso2a4lp5v66gkzarck.png)

Padding the sequences to get vectors (vector has fixed length):

![Image description](https://dev-to-uploads.s3.amazonaws.com/uploads/articles/jdt3wylzv4o7gzriw11e.png)

During the training, the text and token vectors from corpus will be input to embedding layer and added by their positional encoding vector. One can read more details in the paper *"Attention is all you need"*. As we can see, because of this tiny vocabulary, dimensions of the model layers could be in quite low scale. Thus we have a quite small number of model parameters to train. It is so small that we can train the model with only a low-performance cpu. This is exactly the purpose of this repo: **anyone can try training a tranformer without using large language corpus and expensive GPUs**.

## Dependencies
* tensorflow >= 2.1
* num2words
* json
* numpy
and some other packages

## Quick start with trained model
With the pre-trained model, one can start the testing with the following command line:

```bash
python myTransformer.py

```

If everything goes well, one should see the output simalar to:

```bash
====================== Testing the model ============================
Testing the model now...
ID =  1061    20327875.1 :  vingt millions trois cent vingt-sept mille huit cent soixante-quinze virgule un
 - tranlation result:  ['vingt million trois cent vingt - sept mille huit cent soixante - quinze virgule un <END> <END> <END> ... <END> <END>']

Testing a fresh new data...
ID =  0    61250001 :  soixante et un millions deux cent cinquante mille un
fresh_x [[7, 2, 3, 6, 1, 1, 1, 2]]
fresh_y [[41, 33, 39, 13, 36, 14, 34, 32, 35, 13, 42]]
 - translation result:  ['soixante et un million deux cent cinquante mille un <END> <END> <END> ... <END> <END> <END> <END>']
```


## Translate an input number
With the pre-trained model, one can use it to translate a number (positive integer is supported)  with `--translate` option as:


```bash
python myTransformer.py  --translate 2021

```


## Train the model
```bash
python myTransformer.py --training
```

