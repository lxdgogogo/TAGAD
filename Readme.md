# Towards Anomaly Detection on Text-Attributed Graphs


## Dependencies

- Pytorch 2.1.2
- DGL 2.4.0
- sklearn
- imblearn
- Numpy


***

## Dataset

The datasets are in the "datasets" folder.First unzip these datasets.

***

## Hwo to run

Global module:

```
cd Global

python main.py --dataset cora --epoch 200 --lr 5e-3 \ 
--dropout 0.6 --batch_size 10000 --num_layers 1 --patience 5\
--weight_decay 5e-4 --device cuda
```


Local module:

```
cd Local

[//]: # zero shot
python zero_shot.py --dataset cora --epsilon 0.5 --lamb 0.4 --alpha 0.7 --batch_size 10000 --num_layers 1 --device cuda

[//]: # few shot
python few_shot.py --dataset cora --shot_num 3 --epoch 200 --lr 5e-3 --dropout 0.6 --epsilon 0.5 --lamb 0.4 --alpha 0.7 --batch_size 10000 --num_layers 1 --patience 5 --weight_decay 5e-4  --device cuda

```

