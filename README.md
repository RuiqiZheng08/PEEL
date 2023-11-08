# PEEL
This is the Pytorch implementation for PEEL


## Environment Requirement
The code has been tested under Python 3.6.9. The required packages are as follows:
* pytorch == 1.3.1
* numpy == 1.18.1
* scipy == 1.3.2
* sklearn == 0.21.3


## Structure Design
![PEEL](/Fig/PEEL.jpg)

PEEL has three main components: **full embedding pretraining**, **personalized elastic embedding learning**, and **on-device ranking**. 

1. In the pretraining stage, we pretrain recommendation models with collected data instances and utilize a diversity-driven regularizer to output the item embedding table. Users are clustered into groups based on user embeddings. 

2. In the PEE learning stage, for every user group, a copy of the item embedding table is refined based on the local data instances within the user group, and a controller learns weights for embedding blocks, indicating the contributions of each embedding block to the local recommendation performance. The interdependent embedding table refining and weights optimization are conducted in a bi-level optimization manner, instead of a conventional attention mechanism to prevent overfitting. Finally, 

3. In the on-device ranking stage. The learned PEEs for all items and only one specific user embedding are deployed on the corresponding device. A parameter-free similarity function is implemented to rank all the items and output the recommendations. 



## Code Example:

```shell
python main.py
```

After this command item embedding item_emb_Reg.npy and user embedding user_emb_Reg.npy will be produced. The files are already uploaded to the GitHub repository for convenience. 

```shell
python test.py
```

It will output the recommendation performance of on-device recommendation under specific memory constraints. The memory budgets and other parameters are set in PEEL/NGCF/config_all.py.



Note: If you have any questions or suggestions, please contact: *ruiqi dot zheng at uq dot edu dot au*
