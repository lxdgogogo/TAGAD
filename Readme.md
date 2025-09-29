# Towards Anomaly Detection on Text-Attributed Graphs


## Dependencies

- Pytorch 2.1.2
- DGL 2.4.0
- sklearn
- imblearn
- Numpy


## Model:

![](https://raw.githubusercontent.com/lxdgogogo/TAGAD/master/Figure/model.png)

***

## Dataset

The datasets are in the "datasets" folder. First, unzip these datasets.

***

## How to run

Global module:

```
cd Global

python main.py --dataset cora --epoch 100 --alpha 0.8 --lr 1e-3 \
--dropout 0.6 --batch_size 10000 --num_layers 1 --patience 10 \
--weight_decay 5e-4 --device cuda
```


Local module:

```
cd Local

[//]: # zero shot
python zero_shot.py --dataset cora --dropout 0.6 --epsilon 0.9 --lamb 0.3 --alpha 0.8 --batch_size 10000 --num_layers 1 --device cuda

[//]: # few shot
python few_shot.py --dataset cora --dropout 0.6 --shot_num 2 --epoch 20 --lr 5e-3 --lamb 0.3 --alpha 0.8 --batch_size 10000 --num_layers 1 --weight_decay 5e-4  --device cuda
```

