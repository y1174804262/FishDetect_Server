example_1: url->bert, certificate->RGCN

    example_1_1:使用注意力机制，二者结果加入一个新的线性层
    example_1_2:不使用注意力机制，二者结果直接相加，不通过中间线性层

example_2: url->bert, certificate->bert

    example_2_1:证书只提取subject内容
    example_2_2:不使用注意力机制，二者结果直接相加，不通过中间线性层