# events2-graphnets


## 图结构说明

http://note.youdao.com/noteshare?id=ac6e881bb469d1e08fbc7273e86fdb4a

## 生成数据

```corpus/prepare_graph.py```

## 训练 

```test_graphnets/test_event.py```

## 输入 输出举例

https://github.com/2877992943/financialReport_event

## 评估标准如下 准确召回

predict:11  14  0  0  57  5

y      :11  45  0  0  0   5

accuracy=2/4 # pad id=0的不计入准确召回. predict中4个非零

recall= 2/3   # target 中3个非零

## loss accuracy recall

```
 step 810 loss 0.115877 acc 0.779412 recall 0.898305
 step 820 loss 0.091656 acc 0.837209 recall 0.986301
 step 830 loss 0.106382 acc 0.946808 recall 0.936842
 step 840 loss 0.074234 acc 0.958333 recall 0.793103
 step 850 loss 0.116466 acc 0.857143 recall 0.843750
 step 860 loss 0.065692 acc 0.986111 recall 1.000000
 step 870 loss 0.067061 acc 0.925926 recall 1.000000
 step 880 loss 0.067702 acc 0.909091 recall 0.800000
 step 890 loss 0.080887 acc 0.918919 recall 0.985507
 step 900 loss 0.069226 acc 0.876712 recall 1.000000
 step 910 loss 0.117248 acc 0.800000 recall 0.986301
 step 920 loss 0.061834 acc 0.988095 recall 0.965116
 step 930 loss 0.106629 acc 0.753846 recall 0.924528
 step 940 loss 0.095609 acc 0.870968 recall 0.964286
 step 950 loss 0.071993 acc 0.805970 recall 0.981818
 step 960 loss 0.055524 acc 0.982758 recall 0.934426
 ```

