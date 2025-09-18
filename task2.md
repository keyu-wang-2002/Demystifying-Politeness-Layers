TL;DR: 构建不同细粒度的polite classifier, 提供更全面的评测

我们之前构建的polite classifier是针对整个输出句子进行打分，是paragraph-level的。 除此之外，我们还可以构建word-level和sentence-level的classifier

- word level polite classifier example:
```
假设你是....， 请统计礼貌用词出现频率 (礼貌用词数/句数)

例子：please, could you, would you mind, thank you, sorry, appreciate...

...
```
礼貌用词的例子多找一些

- sentence level polite classifier example
```
假设你是....， 请统计礼貌句子出现频率 (?)

条件句（Could you…?）、疑问句、缓和词（hedges, maybe, possibly）

...
```
礼貌用句的例子多找一些

- paragrah level polite classifier
除了我们自己构建的，之后我们再加一个某个官方做的https://huggingface.co/Intel/polite-guard 来评估
