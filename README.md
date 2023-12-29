# Contrastively Enforcing Distinctiveness for Multi-label Image Classification

<br> [Paper](https://www.sciencedirect.com/science/article/pii/S0925231223007282) |
[Pretrained models](https://drive.google.com/file/d/1JS79pCAv1ky3JK37YgG96mJzTWHck8eB/view?usp=sharing)

Official PyTorch Implementation

> Son D.Dao, He Zhao, Dinh Phung, Jianfei Cai <br/> Department of Data Science and AI, Monash University

**Abstract**

Recently, as an effective way of learning latent representations, contrastive
learning has been increasingly popular and successful in various domains.
The success of contrastive learning in single-label classifications motivates
us to leverage this learning framework to enhance distinctiveness for better
performance in multi-label image classification. In this paper, we show that
a direct application of contrastive learning can hardly improve in multi-label
cases. Accordingly, we propose a novel framework for multi-label classification with contrastive learning in a fully supervised setting, which learns multiple representations of an image under the context of different labels. This
introduces a simple yet intuitive adaption of contrastive learning into our
model to boost its performance in multi-label image classification. Extensive
experiments on four benchmark datasets show that the proposed framework
achieves state-of-the-art performance in the comparison with the advanced
methods in multi-label classification.

## Installation

Please install the following packages:
- Python==3.8
- Install the appropriate Pytorch version. 
- numpy==1.19.2
- randaugment
- scikit-learn
- tqdm

## Dataset

We use data processing code for MS-COCO from [ML-GCN] (https://github.com/megvii-research/ML-GCN). Please download the project and run the file "demo_coco_gcn.py". The important output files are: 
- "category.json": contains class name and class id.
- "train_anno.json" and "val_anno.json": contain the image names and ground truth labels of train and test set, respectively.


## Training Scripts

Train stage one:
```
python train_attention_ema_multic.py --dataroot <Path_to_ML-GCN>/data/coco/ --num_classes 80 --load_size 448 --dataset_mode new_coco --model coco_att_stage_one --batch_size 32 --lr 0.0002 --gpu_ids 0,1 --name coco_stage_one --lr_policy cosine --niter_decay 25 --warm
```

Train stage two:
```
python train_augmented_ema.py --dataroot <Path_to_ML-GCN>/data/coco/ --load_size 448 --gpu_ids 0,1 --num_classes 80 --dataset_mode con_coco --model coco_att_con_stage_two --batch_size 24 --lr 0.01 --gpu_ids 0,1 --name coco_stage_two --lr_policy step --lr_decay_iters 20 --pretrain_folder checkpoints/coco_stage_one/ --epoch 15 --ema
```

## Testing Scripts

Inference on MS-COCO:
```
python test_new_ap.py --eval --dataroot <Path_to_ML-GCN>/data/coco/ --num_classes 80 --load_size 448 --dataset_mode new_coco --model coco_test  --name coco_model --data_type val --epoch 15 --batch_size 64 --ema
```

## Pretrained Models
We provide pre-trained models on MS-COCO dataset [here](https://drive.google.com/file/d/1JS79pCAv1ky3JK37YgG96mJzTWHck8eB/view?usp=sharing). 

## Citation
```
 @article{dao2023contrastively,
  title={Contrastively enforcing distinctiveness for multi-label image classification},
  author={Dao, Son D and Zhao, He and Phung, Dinh and Cai, Jianfei},
  journal={Neurocomputing},
  volume={555},
  pages={126605},
  year={2023},
  publisher={Elsevier}
}
```

## Contact
Feel free to contact if there are any questions or issues - Son Dao (sondaoduy@gmail.com).
