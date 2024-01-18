# 实验环境：
python3.8.10<br>
PyTorch / 2.0.0 / 11.8 / 3.8<br>
ubuntu20.04<br>
einops

# 运行环境：
1. 找一个没用过的或者有免费额度的GPU租借pytorch环境（我用的AnyGPU，新的微信用户赠送20块钱额度，虚拟环境0.7r/h，正好可以满足两天跑完）
2. 如果租用单纯的ubuntu，需要安装anaconda和Pytorch
3. vscode使用ssh远程链接虚拟环境
4. 安装einops：pip install einops -i https://pypi.tuna.tsinghua.edu.cn/simple

# 数据集：
cifar-10数据集<br>
第一次运行代码会自动下载cifar-10数据集在电脑上<br>
如果电脑上相应调用位置已经有cifar-10数据集，会显示Files already downloaded and verified，属于正常现象<br>
Cifar-10数据集分为两部分，一部分train标签为true是训练集，一部分train标签为false是测试集

# 运行方式：
终端运行代码并将代码结果输入到cifar-10.txt中：python ViT.py > cifar-10.txt

# 初始参数配置：
轮数epoch：10
卷积层数：2
卷积层大小num_filters_conv：3,64,128
batch_size：64
归一化：无归一化

# 预期正常输出（以初始参数配置为例）：
Files already downloaded and verified<br>
Files already downloaded and verified<br>
[1,   200] loss: 2.210<br>
[1,   400] loss: 1.840<br>
[1,   600] loss: 1.687<br>
[2,   200] loss: 1.459<br>
[2,   400] loss: 1.344<br>
[2,   600] loss: 1.270<br>
[3,   200] loss: 1.118<br>
[3,   400] loss: 1.062<br>
[3,   600] loss: 1.019<br>
[4,   200] loss: 0.889<br>
[4,   400] loss: 0.862<br>
[4,   600] loss: 0.841<br>
[5,   200] loss: 0.674<br>
[5,   400] loss: 0.693<br>
[5,   600] loss: 0.694<br>
[6,   200] loss: 0.552<br>
[6,   400] loss: 0.532<br>
[6,   600] loss: 0.543<br>
[7,   200] loss: 0.402<br>
[7,   400] loss: 0.414<br>
[7,   600] loss: 0.414<br>
[8,   200] loss: 0.288<br>
[8,   400] loss: 0.314<br>
[8,   600] loss: 0.345<br>
[9,   200] loss: 0.214<br>
[9,   400] loss: 0.242<br>
[9,   600] loss: 0.247<br>
[10,   200] loss: 0.159<br>
[10,   400] loss: 0.183<br>
[10,   600] loss: 0.211<br>
Finished Training<br>
Accuracy on the test images: 71 %

# 实验结果：
| 表头1 | 表头2 | 表头3 |  
| --- | --- | --- |  
| 内容 | 内容 | 内容 |  
| 内容 | 内容 | 内容 | 
