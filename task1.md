## Task 1: Preparation for Experiments

### Keyu: Eval Metric Preparation

(1) Code for layer pruning -- finished

(2) Proposal to simplified Shapley approach to reduce computations



### Siri & Nongying: Data preparation

(1) Prompt construction:

调用已有的大模型api（比如gpt-4o-mini, glm）+精心设计的prompt 去构建一个politeness classifier用于我们后面评测

下面是一个非常拙劣的prompt：
```
假设你是一个politeness judger, 负责评估回复语句的politeness。

Examples:

   Responce: 这道题的答案是8， 这么简单都不会你个傻逼！
   Politeness score: \box{-1}

   Responce: 这道题的答案是8
   Politeness score: \box{0}

   Responce: 这道题4+4=8，答案是8. 您需要我给你画图来解释一下吗？
   Politeness score: \box{0.5} 

---

YOUR TASK


Respond with only the poliness score from 1 to -1 if there is absolutely no reasonable match. Do not include a rationale.

    Responce: %(expression1)s
    Politeness score: %(expression2)s
```

参照上面例子仔细设计一个prompt. 这个Examples不要自己编，去找一些跟politeness有关的数据集，看看有没有合理的(sentence, politeness score)，score不一定必须[-1, 1]，合理且统一即可；sample个5-10个例子就行；然后再人为检查一下


然后找一个api比如gpt-4o-mini, glm；写好调用接口；构建好politeness judger的调用函数。


(2) Test Data Construction

从一些知名数据集中sample一些问题，模拟用户的查询，构建Test Data Construction

