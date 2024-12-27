---
# You can also start simply with 'default'
theme: default
# random image from a curated Unsplash collection by Anthony
# like them? see https://unsplash.com/collections/94734566/slidev
# background: https://cover.sli.dev
# some information about your slides (markdown enabled)
title: LLMBasics
info: |
  ## LLM分享
  By lzr.
# apply unocss classes to the current slide
class: text-center
# https://sli.dev/features/drawing
drawings:
  persist: false
# slide transition: https://sli.dev/guide/animations.html#slide-transitions
# transition: none
# enable MDC Syntax: https://sli.dev/features/mdc

mdc: true
---

# LLM基础知识


<!--
The last comment block of each slide will be treated as slide notes. It will be visible and editable in Presenter Mode along with the slide. [Read more in the docs](https://sli.dev/guide/syntax.html#notes)
-->

---
layout: image-right

# the image source
image: elephant.png
---

# 如何训练大模型
<div class="flex flex-col items-center justify-center h-80">
  <v-click>


1. 把大模型放进显存里
2. 训练模型
3. 把大模型拿出来

  </v-click>
</div>

---

# Adam优化器

原理回顾

**一阶动量**:
  $$m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t$$

**二阶动量**:
  $$v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2$$

  - $m_t$、$m_{t-1}$: 一阶动量
  - $v_t$、$v_{t-1}$: 二阶动量
  - $g_t$: 梯度
  - $\beta_1$、$\beta_2$: 衰减系数超参数，一般是$0.9$、$0.999$


---

# Adam优化器

原理回顾

<div class="grid grid-cols-2 gap-4">
  <div>

**偏差纠正**:
   $$\hat{m_t} = \frac{m_t}{1 - \beta_1^t}$$
   $$\hat{v_t} = \frac{v_t}{1 - \beta_2^t}$$

  </div>
  <div>
    <iframe src="https://www.desmos.com/calculator/nst73wha4p?embed" width="400" height="180" style="border: 1px solid #ccc" frameborder=0></iframe>
  </div>
</div>

**参数更新**:
   $$\theta_{t+1} = \theta_t - \alpha \cdot \frac{\hat{m_t}}{\sqrt{\hat{v_t}} + \epsilon} $$

   - $\theta_t$、$\theta_{t+1}$: 模型参数
   - $\alpha$: Learning rate

---

# 显存构成-Adam

以Adam混合精度为例，16$\times$模型参数量+bs$\times$激活值<sup><a href="https://arxiv.org/pdf/1910.02054">1</a></sup>

```mermaid {theme: 'default',themeVariables: {lineColor: '#000',ontSize: '22px'}}
graph TD
    classDef paramGroup fill:#fff2cc,stroke:#333
    classDef activGroup fill:#e2efd9,stroke:#333
    classDef others fill:#d9e2f3,stroke:#333
    classDef disabled fill:#ededed,stroke:#888,color:#aaa
    classDef outline stroke:#f00,stroke-width:3px

    A[显存]
    A --> B[模型]
    A --> D["Adam"]

    B --> E["梯度<br>(fp16)<br>2x"]
    B --> F["参数<br>(fp16)<br>2x"]
    
    D --> I["全精度参数<br>(fp32)<br>4x"]
    D --> G["一阶动量<br>(fp32)<br>4x"]
    D --> H["二阶动量<br>(fp32)<br>4x"]
    
    A --> M["2\*bs\*激活值<br>(fp16)"]
    A --> C["其他开销<br>(通信/显存分配)"]

    class E,F,I,G,H paramGroup;
    class M activGroup;
    class A,B,C,D others;
```

---

# 显存构成-SGD

12$\times$模型参数量+bs$\times$激活值

```mermaid {theme: 'default',themeVariables: {lineColor: '#000',fontSize: '22px'}}
graph TD
    classDef paramGroup fill:#fff2cc,stroke:#333
    classDef activGroup fill:#e2efd9,stroke:#333
    classDef others fill:#d9e2f3,stroke:#333
    classDef disabled fill:#ededed,stroke:#888,color:#aaa

    A[显存]
    A --> B[模型]
    A --> D["优化器"]

    B --> E["梯度<br>(fp16)<br>2x"]
    B --> F["参数<br>(fp16)<br>2x"]
    
    D --> I["全精度参数<br>(fp32)<br>4x"]
    D --> G["一阶动量<br>(fp32)<br>4x"]
    D --> H["二阶动量<br>(fp32)<br>4x"]
    
    A --> M["2\*bs\*激活值<br>(fp16)"]
    A --> C["其他开销<br>(通信/显存分配)"]

    class E,F,I,G,H paramGroup;
    class H disabled;
    class M activGroup;
    class A,B,C,D others;
```

---

# 显存构成-全精度

全精度下激活值占用更大

```mermaid {theme: 'default',themeVariables: {lineColor: '#000',fontSize: '22px'}}
graph TD
    classDef paramGroup fill:#fff2cc,stroke:#333
    classDef activGroup fill:#e2efd9,stroke:#333
    classDef others fill:#d9e2f3,stroke:#333
    classDef disabled fill:#ededed,stroke:#888,color:#aaa

    A[显存]
    A --> B[模型]
    A --> D["优化器"]

    B --> E["梯度<br>**(fp32)**<br>**4x**"]
    B --> F["参数<br>(fp16)<br>2x"]
    
    D --> I["全精度参数<br>(fp32)<br>4x"]
    D --> G["一阶动量<br>(fp32)<br>4x"]
    D --> H["二阶动量<br>(fp32)<br>4x"]
    
    A --> M["**4**\*bs\*激活值<br>**(fp32)**"]
    A --> C["其他开销<br>(通信/显存分配)"]

    class E,F,I,G,H paramGroup;
    class H,F disabled;
    class M activGroup;
    class A,B,C,D others;
```

---

# Data Parallel & Tensor Parallel

DP：batch层面切分，TP：进一步从矩阵运算上切分

![aa](/tp.png)

---

# Data Parallel & Tensor Parallel

DP/TP节省的都是激活值，模型/优化器仍然需要全量加载

```mermaid {theme: 'default',themeVariables: {lineColor: '#000',fontSize: '22px'}}
graph TD
    classDef paramGroup fill:#fff2cc,stroke:#333
    classDef activGroup fill:#e2efd9,stroke:#333
    classDef others fill:#d9e2f3,stroke:#333
    classDef disabled fill:#ededed,stroke:#888,color:#aaa
    classDef outline stroke:#f00,stroke-width:3px

    A[显存]
    A --> B[模型]
    A --> D["Adam"]

    B --> E["梯度<br>(fp16)<br>2x"]
    B --> F["参数<br>(fp16)<br>2x"]
    
    D --> I["全精度参数<br>(fp32)<br>4x"]
    D --> G["一阶动量<br>(fp32)<br>4x"]
    D --> H["二阶动量<br>(fp32)<br>4x"]
    
    A --> M["2\*bs\*激活值<br>(fp16)"]
    A --> C["其他开销<br>(通信/显存分配)"]

    class E,F,I,G,H paramGroup;
    class M activGroup;
    class A,B,C,D others;
    class M outline;
```

---

# Model Parallel & Pipeline Parallel

模型权重“纵向”切分存放在不同GPU，forward和backword之间GPU不可避免地存在空闲时间

![](/model_parallel.png)

---

# GPU间的通信

注意到刚刚介绍的并行化过程涉及到大量的GPU间通信，这是怎么实现的？


<div class="grid grid-cols-3 gap-4">
  <div>


````md magic-move
```python
...

def forward(x):
    x = self.module_a(x)
    x = self.module_b(x)
    ...
```
```python
...

def forward(x):
    x = self.module_a(x)
    x = x.to('cuda:1')
    x = self.module_b(x)
    ...
```
```python
class Model(nn.Module):
    ...

    def forward(x):
        x = self.module_a(x)
        x = self.module_b(x)
        ...

model0 = Model().to('cuda:0')
model1 = Model().to('cuda:1')
```
````

  </div>
  <div class="col-span-2">

<v-click>

<img src="/gpu_comm.png" width=600 />

</v-click>
  </div>
</div>


<v-click>

NVIDIA Collective Communications Library<sup><a href="https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/operations.html">1</a></sup>提供了几种通信原语（primitives）:
- AllReduce
- Broadcast
- Reduce
- AllGather
- ReduceScatter

</v-click>


<!-- 实际上在大规模训练中，GPU间通信已经称为主要瓶颈之一，因此Nvidia早就推出了高效通讯的库，总结了常用的通讯模式 -->


---

# NCCL Primitives

<span></span>

**AllReduce**: 把每个GPU上的数值求和或平均后发回每个GPU

<img src="/allreduce.png" width=500/>

恰好是DP所需要的操作

<img src="/ddp.png" width=600/>

---

# NCCL Primitives

<span></span>

**Reduce**: 把所有GPU上的数值求和或平均汇总到一个GPU

<img src="/reduce.png" width=600/>

**Broadcast**: 把一个GPU上的数据发送给所有GPU

<img src="/broadcast.png" width=600/>

<v-click>
<p v-after class="floatmarker absolute right-86 top-75 opacity-80 transform">+</p>
<p v-after class="floatmarker absolute right-10 top-82 opacity-80 transform -rotate-10">= AllReduce!</p>
</v-click>

<style>
  .floatmarker {
    font-size: 40px;
  }
</style>

---

# NCCL Primitives

<span></span>

**ReduceScatter**: 把每个GPU上的数值分块求和或平均，结果分块存储在各GPU上

<img src="/reducescatter.png" width=600/>

**AllGather**: 把散落在各GPU上的分块同步

<img src="/allgather.png" width=600/>

<v-click>
<p v-after class="floatmarker absolute right-86 top-75 opacity-80 transform">+</p>
<p v-after class="floatmarker absolute right-10 top-82 opacity-80 transform -rotate-10">= AllReduce!</p>
</v-click>

<style>
  .floatmarker {
    font-size: 40px;
  }
</style>

---

# Ring AllReduce

一种高效的AllReduce算法，以3GPU为例

<div style="overflow-x:scroll;white-space:nowrap">
  <img src="/ddp0.png" />
  <img src="/ddp1.png" />
  <img src="/ddp2.png" />
  <img src="/ddp3.png" />
  <img src="/ddp4.png" />
</div>

<style>
  img {
    display: inline-block;
    width: 400px;
  }
</style>

---

# DBTree AllReduce

适用于更多GPU 的 Double Binary Tree 算法

<img src="/DBtree.png" width="750px" />

---

# 从DP到DDP

Data Parallel -> Distributed Data Parallel

以PyTorch的实现为例，DDP可以看作一种更规范的DP实现

<span></span>

<div class="grid grid-cols-2 gap-4">
  <div>

- **DP：单进程，多线程**  
  - 一个主进程，每个GPU对应一个线程
  - 进程内共享内存，以CPU为中枢进行通信
  - 受GIL影响不能充分利用CPU/GPU性能

<span></span><!-- prevent slidev bug -->

  </div>
  <div>

- **DDP：多进程**  
  - 每个GPU对应一个进程
  - GPU之间通过NCCL等框架通信，允许多机多卡
  - 平衡且充分地利用每个GPU

<span></span><!-- prevent slidev bug -->

  </div>
</div>

<img src="/ddp.png" width=700/>

---

# 从DDP到FSDP

Fully Sharded Data parallel 是ZeRO(Zero Redundancy Optimizer)在Pytorch框架下的一种优化实现<sup><a href="https://arxiv.org/pdf/2304.11277">1</a></sup>

- **Splitting**：“纵向”切割为串行的单元
- **Sharding**：“横向”切割每个单元中的参数，分布到不同的GPU上

<img src="/fsdp.png" width=800/>

---

# FSDP流程解析

FSDP Forward<sup><a href="https://blog.clika.io/fsdp-1/">image source</a></sup>

<div style="overflow-x:scroll;white-space:nowrap">
  <span id="crop0" class="crop"><img src="/fsdp0.svg" /></span>
  <span id="crop1" class="crop"><img src="/fsdp1.svg" /></span>
  <span id="crop2" class="crop"><img src="/fsdp2.svg" /></span>
  <span id="crop3" class="crop"><img src="/fsdp3.svg" /></span>
  <span id="crop4" class="crop"><img src="/fsdp4.svg" /></span>
  <span id="crop5" class="crop"><img src="/fsdp5.svg" /></span>
</div>

<style>
  img {
    display: inline-block;
    /* width: 400px; */
    height: 400px;
    max-width: none;
  }
  .crop {
    display: inline-block;
    overflow: hidden;
  }
  #crop1 img {
    margin-left: -300px;
  }
  #crop2 img {
    margin-left: -290px;
  }
  #crop5 img {
    margin-left: -280px;
  }
</style>

---

# FSDP流程解析

FSDP Backward<sup><a href="https://blog.clika.io/fsdp-1/">image source</a></sup>

<div style="overflow-x:scroll;white-space:nowrap;direction:rtl">
  <span id="crop0" class="crop"><img src="/fsdp_b1.svg" /></span>
  <span id="crop1" class="crop"><img src="/fsdp_b2.svg" /></span>
  <span id="crop2" class="crop"><img src="/fsdp_b3.svg" /></span>
  <span id="crop3" class="crop"><img src="/fsdp_b3.svg" /></span>
  <span id="crop4" class="crop"><img src="/fsdp_b4.svg" /></span>
</div>

<style>
  img {
    display: inline-block;
    /* width: 400px; */
    height: 400px;
    max-width: none;
  }
  .crop {
    display: inline-block;
    overflow: hidden;
  }
  #crop2 img {
    margin-right: -360px;
  }
  #crop3 img {
    margin-left: -400px;
  }
</style>

---

# FSDP流程解析

FSDP Optimizer Update<sup><a href="https://blog.clika.io/fsdp-1/">image source</a></sup>

<img src="/fsdp_os.svg" />

<style>
  img {
    display: inline-block;
    /* width: 400px; */
    height: 400px;
    max-width: none;
  }
</style>

<!-- 至此解决了第一个问题-如何把大模型放到显存里 -->

---

# 如何选择并行策略

Llama3: 我全都要

<img src="/llama_parallel.png" width=800 />

<p class="floatmarker absolute right-100 bottom-60 opacity-50 transform -rotate-10">4D Parallel</p>

<v-click>

*小规模微调一般FSDP就够了

</v-click>


<style>
  .floatmarker {
    font-size: 50px;
  }
</style>

---

# 大模型训练范式

预训练 -> 微调 -> 提示词

<sup><a href="https://www.researchgate.net/figure/Overview-of-LLM-training-process-LLMs-learn-from-more-focused-inputs-at-each-stage-of_fig1_373642018">image source</a></sup>
<img src="/llm_training.png">

<v-click>
<p class="floatmarker absolute right-150 bottom-9 opacity-80 transform rotate-45">→</p>
<p class="floatmarker absolute left-98 bottom-6 opacity-80">SFT、DPO、PPO</p>
</v-click>

<style>
  .floatmarker {
    font-size: 24px;
  }
</style>

---

# 词表的建立

BPE (Byte Pair Encoding)


<div class="grid grid-cols-2 gap-4">
  <div>

BPE<sup><a href="https://paperswithcode.com/method/bpe">1</a></sup>通过识别高频出现的相邻token对来构建词表

```python
text = 'ababcabc'
vocab = ['a', 'b', 'c']
for iteration in range(k):
    token1, token2 = most_frequent_pair(vocab, text)
    vocab.append(token1 + token2)
```

<span></span><!-- prevent slidev bug -->

  </div>
  <div>

![](/bpe.png)

<span></span><!-- prevent slidev bug -->

  </div>
</div>

<v-click>

<div class="grid grid-cols-2 gap-4">
  <div>

GPT4o的tokenizer<sup><a href="https://platform.openai.com/tokenizer">2</a></sup>:
- <span class="tokenizer-tkn tokenizer-tkn-0">To</span><span class="tokenizer-tkn tokenizer-tkn-1"> the</span><span class="tokenizer-tkn tokenizer-tkn-2"> stars</span><span class="tokenizer-tkn tokenizer-tkn-3"> and</span><span class="tokenizer-tkn tokenizer-tkn-4"> to</span><span class="tokenizer-tkn tokenizer-tkn-0"> the</span><span class="tokenizer-tkn tokenizer-tkn-1"> depths</span>
- <span class="tokenizer-tkn tokenizer-tkn-0">Ad</span><span class="tokenizer-tkn tokenizer-tkn-1"> ast</span><span class="tokenizer-tkn tokenizer-tkn-2">ra</span><span class="tokenizer-tkn tokenizer-tkn-3"> abyss</span><span class="tokenizer-tkn tokenizer-tkn-4">os</span><span class="tokenizer-tkn tokenizer-tkn-0">que</span>
- <span class="tokenizer-tkn tokenizer-tkn-0">你好</span><span class="tokenizer-tkn tokenizer-tkn-1">谢谢</span><span class="tokenizer-tkn tokenizer-tkn-2">小</span><span class="tokenizer-tkn tokenizer-tkn-34">笼</span><span class="tokenizer-tkn tokenizer-tkn-0">包</span><span class="tokenizer-tkn tokenizer-tkn-1">再</span><span class="tokenizer-tkn tokenizer-tkn-2">见</span>
- <span class="tokenizer-tkn tokenizer-tkn-0">“</span><span class="tokenizer-tkn tokenizer-tkn-1">给主人留下些什么吧</span><span class="tokenizer-tkn tokenizer-tkn-2">”</span><span class="tokenizer-tkn tokenizer-tkn-3">这</span><span class="tokenizer-tkn tokenizer-tkn-4">句话</span><span class="tokenizer-tkn tokenizer-tkn-0">翻</span><span class="tokenizer-tkn tokenizer-tkn-1">译</span><span class="tokenizer-tkn tokenizer-tkn-2">成</span><span class="tokenizer-tkn tokenizer-tkn-3">英文</span>


<span></span><!-- prevent slidev bug -->

  </div>
  <div>

<span></span><!-- prevent slidev bug -->

  </div>
</div>

<p v-after class="floatmarker absolute right-130 bottom-30 opacity-80">← 英语vs小语种</p>
<p v-after class="floatmarker absolute right-50 bottom-22 opacity-80">← 不同词频的词语得到了不同程度的合并，笼=[6006, 120]</p>
<p v-after class="floatmarker absolute right-80 bottom-13 opacity-80">← 语料清洗不充分导致的bug</p>

</v-click>



<style>

.tokenizer-tkn-0 {
    background: rgba(107,64,216,.3)
}

.tokenizer-tkn-1 {
    background: rgba(104,222,122,.4)
}

.tokenizer-tkn-2 {
    background: rgba(244,172,54,.4)
}

.tokenizer-tkn-3 {
    background: rgba(239,65,70,.4)
}

.tokenizer-tkn-4 {
    background: rgba(39,181,234,.4)
}

.tokenizer-tkn-34 {
  background: linear-gradient(
        to right, 
        rgba(239, 65, 70, 0.4) 50%, 
        rgba(39, 181, 234, 0.4) 50%
    );
}
</style>

---
layout: two-cols-header
---

# 文字到向量的转换

Tokenization + Embedding -> 神经网络能够处理的向量序列

<img src="/tok_emb.png" />

::left::

**Tokenization**:
- 把文本切分成token组成的序列
- *词表*由语料统计得出

::right::

**Embedding**:
- 把每个token编码为向量，得到向量序列
- *Embedding*属于模型参数的一部分，随机初始化后随着训练更新

---

# 自回归生成

如何对大量语料中的文本分布情况进行建模

对整个文本序列进行建模是很难的

$$x=\text{"Why did we play Haruhikage"}$$

但拆分成对每一个token进行建模，就可以化简为一个token层面的分类器模型

$$
\begin{align*}
P(x) =     & \prod_{i=1}^N P(x_i | x_{<i}) \\
     =     & P(\text{"Why"} \mid \emptyset) \\
     \cdot & P(\text{"did"} \mid \text{"Why"}) \\
     \cdot & P(\text{"we"} \mid \text{"Why did"}) \\
     \cdot & P(\text{"play"} \mid \text{"Why did we"}) \\
     \cdot & P(\text{"Haruhikage"} \mid \text{"Why did we play"})
\end{align*}
$$

<!-- 所以所有基于GPT的大模型本质上都是分类器 -->

---

# 预训练

无监督的预训练：建模样本概率分布

目标：最大化预训练语料上的概率

<div class="grid grid-cols-2 gap-4">
  <div>

$$P_{\theta}(x) = \prod_{i=1}^N P_{\theta}(x_i | x_{<i})$$

<span></span><!-- prevent slidev bug -->

  </div>
  <div>

$$L_{\text{pretrain}} = -\sum_{i=1}^N \log P_{\theta}(x_i | x_{<i})$$

<span></span><!-- prevent slidev bug -->

  </div>
</div>

- 其中$P_{\theta}(x_i | x_{<i})$就是以$\theta$为参数的模型

<v-click>

那些没有出现在训练语料中的文本呢?

注意到$\Sigma P = 1$，所以最大化样本概率就等于隐式地最小化非样本的概率

<span style="color:#385723">

$$P(\text{"今天也是好天气"}) \uparrow$$

</span>

<span style="color:#843c0c">

$$P(\text{"搶曽峠杺嫿"}) \downarrow$$

</span>

</v-click>

---

# 预训练的规模

Scaling Laws for Neural Language Models <sup><a href="https://arxiv.org/abs/2001.08361">1</a></sup>

<img src="/scaling.webp" />

<v-click>

Scaling Law 的含义：
- ❌ 对LLM未来发展的预测
- ✔ 对投入资源的合理预估

</v-click>

---

# 从预训练到SFT

Supervised Fine-Tuning

有监督的SFT：将文本划分为输入和输出，建模条件概率分布

<div class="grid grid-cols-2 gap-4">
  <div>

$$P_{\theta}(y | x) = \prod_{i=1}^N P_{\theta}(y_i | y_{<i}, x)$$

<span></span><!-- prevent slidev bug -->

  </div>
  <div>

$$L_{\text{SFT}} = -\sum_{i=1}^N \log P_{\theta}(y_i | y_{<i}, x)$$

<span></span><!-- prevent slidev bug -->

  </div>
</div>

<v-click>

通常是针对特定下游任务进行的，例如类似于ChatGPT的对话任务

$$x=\text{"993-7=?"}, y=\text{"986"}$$

需要注意，$P_{\theta}$还是同一个$P_{\theta}$，只是数据的形式变了

</v-click>

<v-click>

**能不能跳过预训练直接SFT？** 数据少，收敛慢

</v-click>

<!-- SFT通常是针对特定应用场景进行的，比如问答，因此在数据上把语料划分为了输入和输出，对条件概率进行建模 -->

---

# 没有足够的标注？

用户评价也是一种标注

<img src="/rating.png"/>

RLHF（Reinforcement learning from human feedback）：
- 用用户评价+强化学习进行模型微调
- 对数据要求更低
- 标注过程只需要对模型的输出评判好坏
- 训练成本稍高

---

# RLHF

强化学习如何与LLM对应

<div class="grid grid-cols-6 gap-4">
  <div class="col-span-3">

- 策略模型（Policy） $\rightarrow$ LLM
- 动作 $\rightarrow$ 预测下一个token
- 动作空间 $\rightarrow$ 词表
- 轨迹 $\rightarrow$ LLM生成的字符串
- Reward Function $\rightarrow$ 人类反馈？

<span></span>

  </div>
  <div v-click="+5" class="col-span-3">
    <img src="/rlhf.jpg" />
  </div>
</div>

<v-click>

从Reward Function到Reward Model

</v-click>

<div class="grid grid-cols-6 gap-4">
  <div v-after class="col-span-2">

```python
def reward_function(state):
    # 传统的强化学习通过规则判定reward
    if i_win(state):
        return 1.0
    else:
        return -1.0
```

  </div>
  <div v-click class="col-span-2">

```python
def reward_function(state):
    # 每次都找标注人员问一下？效率太低了
    print(state)
    reward = input("请输入你的评价：")
    return float(reward)
```

<span></span>

  </div>
  <div v-click class="col-span-2">

```python
def reward_function(state):
    # 太好了，是*深度学习*，我们有救了
    reward = model(state)
    return reward
```

<span></span>

  </div>
</div>

<p v-click class="floatmarker absolute right-130 top-60 opacity-90">—————　Reward Model</p>

---

# PPO

比较常见的在线强化学习方法<sup><a href="https://huggingface.co/docs/trl/v0.11.4/en/ppo_trainer">1</a></sup>

<img src="/ppo.png" width=800 />

---

# PPO的显存占用

我有一个绝妙的算法，但是CUDA out of memory

对于7B的LlAMA，bs=1，PPO占用了220GB显存<sup><a href="https://arxiv.org/pdf/2309.00754">1</a></sup>

- Actor（LLM本身）：推理+训练
- Value Model：推理+训练
- Reference Model：推理
- Reward Model：推理

<v-click>

**DPO** : 如果我们把LLM本身看作一个Reward Model，就能极大地简化问题<sup><a href="https://arxiv.org/pdf/2305.18290">1</a></sup>

<img src="/rlhf_dpo.png"/>

</v-click>

---

# DPO

Direct Preference Optimization

**Preference**：对于同一个输入$x$，输出$y_w$（win）比$y_l$（lose）更好。

$$
\mathcal{L}_\text{DPO}(\pi_{\theta}; \pi_\text{ref}) = -\mathbb{E}_{(x, y_w, y_l)\sim \mathcal{D}}\left[\log \sigma \left(\beta \log \frac{\pi_{\theta}(y_w\mid x)}{\pi_\text{ref}(y_w\mid x)} - \beta \log \frac{\pi_{\theta}(y_l\mid x)}{\pi_\text{ref}(y_l\mid x)}\right)\right].
$$

考虑$\log \sigma (a - b)$的性质：

<iframe src="https://www.desmos.com/calculator/jlaitcglrm?embed" width="500" height="200" style="border: 1px solid #ccc" frameborder=0></iframe>

可以发现DPO的目标是增大$y_\text{w}$相对于$y_\text{l}$的优势

---

# 不同的Feedback形式

如果数据不是成对的呢？

<div class="grid grid-cols-4 gap-4">
  <div v-after class="col-span-2">

<img src="/pref_pair.png" />

<div v-click="1">

两个候选，二选一：Bradley–Terry模型<sup><a href="https://en.wikipedia.org/wiki/Bradley%E2%80%93Terry_model">1</a></sup>

$\rightarrow$ DPO

</div>

  </div>
  <div v-after class="col-span-2">

<img src="/pref_single.png" />

<div v-click="1">

只有一个候选，选或者不选：Kahneman & Tversky “展望理论”<sup><a href="https://zh.wikipedia.org/zh-cn/%E5%B1%95%E6%9C%9B%E7%90%86%E8%AE%BA">2</a></sup>

$\rightarrow$ KTO<sup><a href="https://arxiv.org/abs/2402.01306">3</a></sup>

</div>

  </div>
</div>


---

# 开源模型是怎样炼成的

大多数模型都会结合多种微调方式

| Model          | 训练流程                      |
|----------------|---------------------------------------------|
| **Llama3**<sup><a href="https://arxiv.org/pdf/2407.21783">1</a></sup>     | Pretrain → Train RM → SFT → DPO → Model Average |
| **Qwen2.5**<sup><a href="https://arxiv.org/pdf/2412.15115">2</a></sup>    | Pretrain → SFT → DPO → Train RM → GRPO (PPO 改进版) |
| **Deepseek v2**<sup><a href="https://arxiv.org/pdf/2405.04434">3</a></sup>| Pretrain → SFT → Train RM → GRPO                      |

一般开源模型都会提供预训练版本和微调版本

<img src="/qwencard.png" width=700/>

---

# 我们可以做什么

SFT案例：遥感智能体

<img src="/agent.png" />

<div class="grid grid-cols-2 gap-4">
  <div>

任务特点：
- 可以借助GPT构造出大量输入-输出数据
- 可以自动筛选出正确数据

<span></span>

  </div>
  <div>

微调开源模型
- 采用FSDP进行SFT
- 2张A800，3000+数据，耗时10h
- 得到在遥感灾害智能体任务上比GPT4omini表现更好的模型

<span></span>

  </div>
</div>

---

# 开源框架

<span></span>

🤗🤗🤗🤗🤗🤗

<div class="grid grid-cols-3 gap-4">
  <div>

datasets<sup><a href="https://huggingface.co/docs/datasets/en/index">1</a></sup>，evaluate<sup><a href="https://huggingface.co/docs/evaluate/index">2</a></sup>:
- 数据集加载、预处理
- 标准化的评价指标

transformers<sup><a href="https://huggingface.co/docs/transformers/index">3</a></sup>: 
- AutoModel: 预训练模型的加载

TRL<sup><a href="https://huggingface.co/docs/trl/index">5</a></sup>:
- SFT Trainer<sup><a href="https://huggingface.co/docs/trl/en/sft_trainer">6</a></sup>
- DPO Trainer<sup><a href="https://huggingface.co/docs/trl/dpo_trainer">7</a></sup>
- PPO Trainer<sup><a href="https://huggingface.co/docs/trl/ppo_trainer">8</a></sup>

<span></span>

  </div>
  <div class="col-span-2">
<v-click>

使用例

````md magic-move {lines: true}
```python
from transformers import AutoModelForCausalLM
from datasets import load_dataset
from trl import SFTConfig, SFTTrainer
```
```python {2,5}
from transformers import AutoModelForCausalLM
from datasets import load_dataset
from trl import SFTConfig, SFTTrainer

dataset = load_dataset("stanfordnlp/imdb", split="train")
```
```python {1,7}
from transformers import AutoModelForCausalLM
from datasets import load_dataset
from trl import SFTConfig, SFTTrainer

dataset = load_dataset("stanfordnlp/imdb", split="train")

model = AutoModelForCausalLM.from_pretrained("facebook/opt-350m")
```
```python {3,9-15}
from transformers import AutoModelForCausalLM
from datasets import load_dataset
from trl import SFTConfig, SFTTrainer

dataset = load_dataset("stanfordnlp/imdb", split="train")

model = AutoModelForCausalLM.from_pretrained("facebook/opt-350m")

training_args = SFTConfig(output_dir="/tmp")
trainer = SFTTrainer(
    model,
    train_dataset=dataset,
    args=training_args,
)
trainer.train()
```
```python
from transformers import AutoModelForCausalLM
from datasets import load_dataset
from trl import SFTConfig, SFTTrainer

dataset = load_dataset("stanfordnlp/imdb", split="train")

model = AutoModelForCausalLM.from_pretrained("facebook/opt-350m")

training_args = SFTConfig(output_dir="/tmp")
trainer = SFTTrainer(
    model,
    train_dataset=dataset,
    args=training_args,
)
trainer.train()
```
````

</v-click>

<span v-after>

```bash
python -m torch.distributed.launch train.py
```

</span>
  </div>
</div>

<style>
h1 {
  background-color: #2B90B6;
  background-image: linear-gradient(45deg, #4EC5D4 10%, #146b8c 20%);
  background-size: 100%;
  -webkit-background-clip: text;
  -moz-background-clip: text;
  -webkit-text-fill-color: transparent;
  -moz-text-fill-color: transparent;
}
</style>


---

# 并非万能

LLM的大厦已经基本建成，只是天边漂浮着两朵乌云

<!-- 微调能实现什么，不能实现什么？ -->

The Reversal Curse<sup><a href="https://arxiv.org/pdf/2309.14402">1</a></sup>

<img src="/reverse1.png" width=700 />

微调后进行推理：

<img src="/reverse2.png" width=700 />

---

# 大模型的知识

大模型真的掌握“知识”吗？节选自*Physics of Language Models*<sup><a href="https://arxiv.org/pdf/2309.14316">1</a></sup>

设计验证能否通过FineTune从预训练大模型中“提取”知识：

1. bioS

<img src="/anya_bio.png" />

对bioS进行的数据增强：multi / permute / full name

2. QA

<img src="/anya_qa.png" />

---

# 大模型的知识

实验节选

设定：使用Bio进行预训练，50%QA进行微调，剩余的QA进行测试

| 预训练          | 微调       | 测试准确率 |
|-----------------|------------|------------|
| bioS + 50%QA    | /          | ✔          |
| bioS            | 50%QA      | ❌          |
| bioS + 各种增强 | 50%QA      | ✔          |

<v-click>

结论：
1. 单纯在预训练中“记住”的知识并不能被有效提取
2. 以多样表述、多种顺序重复提及的知识才能被有效提取

如果想让大模型学会知识，就需要对同一个知识在训练数据中提供尽可能多样的表述方式

</v-click>

---

# 大模型的逻辑能力

大模型发展出了真正的逻辑，还是只是在套用模板？<sup><a href="https://arxiv.org/pdf/2407.20311">1</a></sup>

设定：小学水平的数学题目，合成数据集，严格避免数据泄露，分布外（OOD）测试

<img src="/logic1.png">

<v-click>

✔ Level1: 确实掌握了逻辑，而非套用回答模板

</v-click>

---

# 大模型的逻辑能力

大模型会思考吗

V-Probing：探索大模型“内心”的想法

<img src="/logic2.png" width=700 />

---

# 大模型的逻辑能力

大模型会思考吗

Probing内容

- **nece(A)**：参数A对于计算答案是是否必要
- **dep(A, B)**：当参数A是否依赖于参数B
- **known(A)**：参数A是否已知。
- **value(A)**：参数A的值
- **can next(A)**：参数A是否可以在下一步中求出来
- **nece next(A)**：`can next(A)`且`nece(A)`

<v-click>

✔ Level2: LLM在内部已经构建了所有参数之间的依赖关系

</v-click>

---

# Open Questions

既然大模型不一定真的掌握知识，那么...

人类是如何掌握知识的？

人类能真的学会知识吗？

<img src="/xilanai.jpg" />

---
layout: center
class: text-center
---

# The End

感谢倾听
