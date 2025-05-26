# 最重要
标签值：1为钓鱼，0为良性
钓鱼网站下载csv:http://data.phishtank.com/data/online-valid.csv.gz

数据集收集：
1、采集数据：datacollect/main.py
2、清洗数据：datacollect/clean_data.py
3、证书数据转为特征：model/RGCN/feature_resolve.py
4、html数据转为特征：model/RGCN/html_process.py
5、最终数据集构建：model/CMACM/create_dataset.py
6、开始训练：model/CMACM/main.py

# 1. 项目架构

## 1.1  data collect

&emsp;&emsp;数据采集模块，包含HTML代码提取、TLS证书提取、页面截图

&emsp;&emsp;&emsp;&emsp;应该首先提取TLS证书，如果提取不到则不需要再提取HTML代码和页面截图，不然太慢了

    html&screen_collect.py: 从URL中提取HTML代码及页面截图<一次>

证书链解析，需要修改resolver，让其返回证书链的字节数据，以此可以将证书保存于本地,该部分依赖

    https://github.com/rkoopmans/python-certificate-chain-resolver项目修改

得到，修改后的项目存储于

    https://github.com/y1174804262/python-certificate-chain-resolver/tree/master

# 1.2 data process

&emsp;&emsp;数据处理模块，包含HTML代码解析、TLS证书解析、页面截图解析

解析证书信息，将证书中的信息提取出来，以供后续存储为图格式

    cert_resolv.py: 从证书链中提取证书链长度、证书链中证书的长度、证书链中证书的信息
    cert2graph.py: 将证书链信息存储为图格式

证书链转变为图格式，每个证书唯一标识为证书的序列号（序列号，并不一定是唯一的，因此可以后续考虑变换）

证书链结构

    序列号-[father]->序列号
    序列号-[]->
    序列号-[Subject]->主体
    序列号-[issue]->颁发者
    序列号-[validity]->有效期
    序列号-[public key]->公钥
    序列号-[signature]->签名算法
    序列号-[extension——count]->扩展数量《未考虑好如何处理》

cert2graph 存储图结构，以上信息为节点的索引，同时每个节点存在一个feature属性值，此为使用bert将数据转换为向量的结果，方便后续进行训练

边同样存在一个feature属性值，此为使用独热编码将数据转换为向量的结果，方便后续进行训练

feature_transform.py: 使用独热编码，将边上的信息转换为向量

bert_embedding.py: 使用bert将数据转换为向量的结果

# 1.3 model

该类存储各种训练模型，是模型训练的核心

## 1.3.1 GCN
GCN模型的相关东西，使用的GCN模型为其变体RGCN，由于关系需要处理，因此使用了RGCN

    data_loader.py: 读取数据
    feature_resolv.py: 读取数据
    gcn_main.py: GCN模型的训练
    gcn_model.py: GCN模型的定义
    gcn_predict.py: GCN模型的预测
    gcn_train.py: GCN模型的训练

RGCN模型的输入：节点特征，边特征<关系>，边的索引

# 1.4 HTML资源处理

HTML由于其包含无关信息太多，钓鱼网站是由开发者所编写的，因此仅考虑开发者自创的信息

css代码可以完全剔除
js代码仅仅保留url标签头

其余信息，首先提取标签中存在href的



    html_process.py: HTML代码的处理

# 2. 数据集构建
数据集以json格式构建，html和证书信息已经以特征的形式存储，因此只需要加载特征即可

用csv格式存储会更简单

```
id, url, certificate_feature_path, html_feature_path, label
```

# 2.碎碎念

首先url使用简单的bert模型，huggingface上有一个专门的识别钓鱼url的预训练模型，拿过来直接玩得了

TLS证书，证书使用图结构进行处理，更准确说使用知识图谱的结构，因为证书中的属性可以以关系进行映射，因为是有关系的图所以使用了RGCN，
RGCN的优缺点我也不知道，具体什么原理我也看不懂，但是这似乎是处理带关系的图的最好办法

TLS证书的属性目前只提取了那些简单的属性，还没有考虑证书的拓展属性，因为拓展的信息太过乱和庞杂，具体就是一个证书的拓展数量可能不是固定的，这个还好处理，
后期将得到的证书数据进行一个统计，看看拓展具体有哪些，拓展可能有十多种。关键问题是拓展内容信息也比较大，其实也能做，截断，把特征内容处理好就行，
我们要用的是特征，并不是他的具体内容。另外留的一个是证书主体和签发者的那些信息，这些东西看看吧，后面如果效果不好再给加上，其实这部分信息是存在意义的，
因为这些主体中可能有的含有组织验证，有的不含组织验证，组织验证也是区分钓鱼网站的一个重要信息，看效果吧，效果不好再改

HTML代码处理，HTML代码主要问题是HTML页面太大，直接丢给bert之类的模型必然会被截断，现在先找了个token较大的大模型，先拿来试试看好不好用，
不好用的话就做向量拼接，拼接将HTML页面中的HTML代码提取出来，整体的丢入一个能处理较长信息的模型中，返回一个向量，
然后对于js代码，分块，一个js脚本丢一个模型，返回一个向量，这会得出多个向量，这个向量数量是不固定的，有多少个js脚本就有多少个向量，这个再思索怎么处理吧
（简单方法，把这几个向量拼接做卷积卷积成特定维度的，假设他就5个js<感觉这个数差不多>超过5个的随机丢弃剩下的，少了的填充空的，但是这肯定是有问题的， 所以有没有可能能将n维卷积成1维的，应该也有，到时候再看）

大模型太慢了，前面的方法很难实现，最终决定将html源码转换为dom树，仍然以GCN进行处理


好了大模型下完了，拿来玩吧



