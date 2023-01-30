# KRDN

Source code for Knowledge-refined Denoising Network for Robust Recommendation

## Environment Requirements

- Ubuntu OS
- Python >= 3.8 (Anaconda3 is recommended)
- PyTorch 1.7+
- A Nvidia GPU with cuda 11.1+

## Datasets

We use three processed datasets: Alibaba-iFashion, Yelp2018 and Last-FM. 
* We follow the paper "[Learning Intents behind Interactions with Knowledge Graph for Recommendation](https://arxiv.org/abs/2102.07057)" to process data.
* You can find the full version of recommendation datasets via [Alibaba-iFashion](https://github.com/wenyuer/POG), [Yelp2018](https://www.heywhale.com/mw/dataset/5ecbc342fac16e0036ec41a0) and [Last-FM](http://www.cp.jku.at/datasets/LFM-1b/).

## Training

- Alibaba-iFashion dataset
```
python main.py --dataset alibaba-ifashion --lr 0.0001 --context_hops 3 --num_neg_sample 200 --margin 0.6 --max_iter 2
```

- Yelp2018 dataset
```
python main.py --dataset yelp2018 --lr 0.0001 --context_hops 2 --num_neg_sample 400 --margin 0.8 --max_iter 1
```

- Last-FM dataset
```
python main.py --dataset last-fm --lr 0.0001 --context_hops 2 --num_neg_sample 400 --margin 0.7 --max_iter 2
```