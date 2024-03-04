## self-supervised Domain Adaptation with Significance-Oriented Masking (DASOM): A new solution for pelvic organ prolapse (POP) detection.

<p align="center">
  <img src="[https://user-images.githubusercontent.com/11435359/146857310-f258c86c-fde6-48e8-9cee-badd2b21bd2c.png](https://github.com/xiaolilaoli/dasom/blob/main/framework.png)" width="480">
</p>


This is a PyTorch/GPU re-implementation of the paper "Self-Supervised Domain Adaptation with Significance-Oriented Masking for Pelvic Organ
Prolapse Detection".

Alright, you can't find this article online yet,  because it's still in its (very) long review process.:


This article uses deep learning to detect pelvic floor dysfunction(POP) diseases. In this article, we have completed two works that we find very interesting.

The first work is data processing. Our pelvic floor dataset is imbalanced and has few samples. We processed the data using resampling and strong random data augmentation. This data augmentation method comes from AutoAugment, and we used their reinforcement learning approach to search for augmentation strategies suitable for the pelvic floor dataset. 

The reason why this work is interesting is that it actually worked. Beyond our imagination, in fact, this is a very difficult-to-classify dataset. Before this data processing, the highest classification accuracy could only reach 70%-80%. And this data processing can improve the detection accuracy by 10-20%. I am sure there are some secrets behind this. But there is no way. I am about to graduate soon, so just leave this work to someone who is destined to explore it.

The second work is to explore the masking method for the masked image modeling(MIM) task. In fact, this is a direction worth studying. In the MIM task, what is the best way to mask? Is random masking effective? Perhaps the answer can only be obtained through exploration using reinforcement learning on a large-scale dataset. Of course, require a lot of resources.


Our work is based on [VIT (Vision Transformer)](https://arxiv.org/abs/2010.11929) and  [Masked Autoencoders Are Scalable Vision Learners](https://arxiv.org/abs/2111.06377). 
Therefore, our model also loads the [pre-trained parameters](https://dl.fbaipublicfiles.com/mae/visualize/mae_visualize_vit_base.pth) provided by them. Alright, you can see that many of our code also comes from them.
Very sorry that due to privacy reasons, we are unable to provide our dataset.

But we still have our own unique features.The main running file for the work is 
```
python myFinetune.py
```
In it, you can experiment with different mask ratios, different training models, or different decoder depths.

You can find our data processing approach in 
```
model_utils/data.py  
```
and
```
 model_utils/my_aug.py
```
If you are interested in how we implemented masking, get it in 

```
models_mae.py (MaskedAutoencoderViT.random_masking)
```

other aspects need to be explored on your own. Wish you happiness every day.
