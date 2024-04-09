### Introduction
We train a model to detect the diffusion model-based AI images.

#### References

https://github.com/zoie-ui/AI-Generated-Image-Detection

https://github.com/grip-unina/DMimageDetection

### Usage

#### Load your own dataset 

You can collect the image from the illustration community, such as [Pixiv](https://www.pixiv.net/).

Please copy your non-AI images into `data/images` and the AI images into `data/images_ai`, and run the dataset construction script:

```shell
cd data
python gen_set.py
```

#### Data preprocessing
```shell
python preprocess.py
```

#### Model Training 

Execute the training script at the root directory

```shell
python dual_net_train.py
```

#### Test

```shell
python test.py
```
