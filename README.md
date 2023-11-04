# TransformerDigits

This repo demonstrates translation from digits to french using Transformer

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

