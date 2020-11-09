# Tiny_ImageNet_Challenge

[Tiny ImageNet Challenge](https://tiny-imagenet.herokuapp.com/): This is a miniature of ImageNet classification Challenge.

# Getting Started
Download git and dataset
```Shell
git clone https://github.com/cjf8899/SSD_ResNet_Pytorch.git

cd Tiny_ImageNet_Challenge
cd data
sh download_and_unzip.sh

# Pretrain model download(ImageNet)
wget http://data.lip6.fr/cadene/pretrainedmodels/se_resnext50_32x4d-a260b3a4.pth

```

the structures would like
```
~/Tiny_ImageNet_Challenge/data/
    -- tiny-imagenet-200
    -- se_resnext50_32x4d-a260b3a4.pth
  
```
# Train

I used wandb and various other transforms as well.

The wandb code is included in this repositories.

[Wandb Guide](https://greeksharifa.github.io/references/2020/06/10/wandb-usage/)

All of the special transforms I used are included in [this repo](https://github.com/cjf8899/simple_tool_pytorch)

# Results


I used SE-ResNeXt50_32x4d and the best performance is 82.36%

|              Implementation              |    top-1     |    top-5     |
| :--------------------------------------: | :---------: | :---------: |
| SE-ResNeXt50 |   82.36   |   94.96   |



