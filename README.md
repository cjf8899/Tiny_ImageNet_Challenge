# Tiny_ImageNet_Challenge

[Tiny ImageNet Challenge](https://tiny-imagenet.herokuapp.com/): This is a miniature of ImageNet classification Challenge.

# Getting Started
Download git and dataset
```Shell
git clone https://github.com/cjf8899/Tiny_ImageNet_Challenge.git

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

All of the special transforms I used are included in [this repositories](https://github.com/cjf8899/simple_tool_pytorch).

# Results


I used SE-ResNeXt50_32x4d and the best performance is 82.54%

|              Implementation              |    top-1     |    top-5     |
| :--------------------------------------: | :---------: | :---------: |
| SE-ResNeXt50 |   82.54   |   94.96   |

<p align="center"><img src="https://user-images.githubusercontent.com/53032349/98533514-6d722c00-22c6-11eb-89ab-a73be0d8384b.png" width="50%" height="50%" title="70px" alt="memoryblock"></p>
