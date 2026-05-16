# Nanotron 项目架构文档与学习路径

> **Nanotron** 是由 Hugging Face 开发的极简大语言模型（LLM）预训练框架，专注于简洁性、性能和可扩展性。
> 版本：0.4 | Python 要求：~=3.10

---

## 目录

- [一、项目概览](#一项目概览)
- [二、项目目录结构](#二项目目录结构)
- [三、核心架构设计](#三核心架构设计)
  - [3.1 整体架构图](#31-整体架构图)
  - [3.2 配置系统](#32-配置系统)
  - [3.3 分布式训练引擎（Trainer）](#33-分布式训练引擎trainer)
  - [3.4 并行上下文（ParallelContext）](#34-并行上下文parallelcontext)
- [四、三维并行体系](#四三维并行体系)
  - [4.1 张量并行（Tensor Parallelism）](#41-张量并行tensor-parallelism)
  - [4.2 流水线并行（Pipeline Parallelism）](#42-流水线并行pipeline-parallelism)
  - [4.3 数据并行（Data Parallelism）](#43-数据并行data-parallelism)
  - [4.4 专家并行与上下文并行](#44-专家并行与上下文并行)
- [五、模型架构层](#五模型架构层)
  - [5.1 模型基类 NanotronModel](#51-模型基类nanotronmodel)
  - [5.2 支持的模型](#52-支持的模型)
  - [5.3 神经网络组件（nn 模块）](#53-神经网络组件nn-模块)
- [六、参数管理系统](#六参数管理系统)
  - [6.1 NanotronParameter](#61-nanotronparameter)
  - [6.2 分片参数（Sharded Parameters）](#62-分片参数sharded-parameters)
  - [6.3 绑定参数（Tied Parameters）](#63-绑定参数tied-parameters)
- [七、数据流水线](#七数据流水线)
  - [7.1 数据集构建](#71-数据集构建)
  - [7.2 数据加载器](#72-数据加载器)
  - [7.3 数据流图](#73-数据流图)
- [八、优化器与梯度管理](#八优化器与梯度管理)
  - [8.1 优化器架构](#81-优化器架构)
  - [8.2 梯度累积器](#82-梯度累积器)
  - [8.3 ZeRO 优化器](#83-zero-优化器)
- [九、FP8 量化支持](#九fp8-量化支持)
- [十、检查点与序列化](#十检查点与序列化)
- [十一、推理与生成](#十一推理与生成)
- [十二、评估系统](#十二评估系统)
- [十三、参数初始化与缩放](#十三参数初始化与缩放)
- [十四、完整训练数据流](#十四完整训练数据流)
- [十五、系统学习路径](#十五系统学习路径)
  - [阶段一：基础入门](#阶段一基础入门)
  - [阶段二：核心机制](#阶段二核心机制)
  - [阶段三：并行策略](#阶段三并行策略)
  - [阶段四：高级特性](#阶段四高级特性)
  - [阶段五：实战与扩展](#阶段五实战与扩展)

---

## 一、项目概览

Nanotron 的核心设计原则：

| 原则 | 描述 |
|------|------|
| **简洁性** | 提供简单灵活的 API 来预训练模型 |
| **性能** | 使用最新技术优化训练速度和效率 |
| **可扩展性** | 支持 3D 并行（TP/PP/DP）及更多并行策略 |

**核心能力**：
- 支持 Llama、Qwen2、Starcoder2 等主流模型架构
- 完整的 3D 并行训练（张量并行 + 流水线并行 + 数据并行）
- FP8 混合精度训练
- MoE（混合专家）架构支持
- ZeRO 优化器与梯度累积
- 检查点保存/恢复与 S3 上传
- 与 LightEval 集成的评估系统
- 文本生成与推理

---

## 二、项目目录结构

```
nanotron/
├── src/nanotron/                  # 核心源码
│   ├── config/                    # 配置系统
│   │   ├── config.py              # 主配置类（Config、LoggingArgs、TokensArgs 等）
│   │   ├── models_config.py       # 模型配置（LlamaConfig、Qwen2Config、MoEConfig 等）
│   │   ├── parallelism_config.py  # 并行配置（ParallelismArgs）
│   │   ├── lighteval_config.py    # 评估配置
│   │   └── utils_config.py        # 配置工具（枚举转换、序列化）
│   ├── data/                      # 数据处理
│   │   ├── nanoset.py             # Nanoset 数据集实现
│   │   ├── dataloader.py          # 数据加载与校验
│   │   ├── dataloader_builder.py  # 数据加载器构建器
│   │   ├── clm_collator.py        # CLM 数据整理器
│   │   ├── samplers.py            # 采样器
│   │   ├── processing.py          # 数据预处理
│   │   ├── sft_processing.py      # SFT 数据预处理
│   │   └── nemo_dataset/          # NeMo 格式数据集支持
│   ├── models/                    # 模型实现
│   │   ├── base.py                # NanotronModel 抽象基类
│   │   ├── llama.py               # Llama 模型实现
│   │   ├── qwen.py                # Qwen2 模型实现
│   │   └── starcoder2.py          # Starcoder2 模型实现
│   ├── nn/                        # 神经网络组件
│   │   ├── attention.py           # 注意力机制（Flash、Flex、Ring）
│   │   ├── activations.py         # 激活函数
│   │   ├── layer_norm.py          # 层归一化（TritonRMSNorm、LlamaRMSNorm）
│   │   ├── rotary.py              # 旋转位置编码（RoPE）
│   │   ├── moe.py                 # MoE 组件（Router、GroupedMLP）
│   │   ├── ring_attention.py      # Ring Attention 实现
│   │   └── flex_attention.py      # Flex Attention 实现
│   ├── parallel/                  # 并行策略
│   │   ├── context.py             # ParallelContext（进程组管理）
│   │   ├── parameters.py          # NanotronParameter（参数元数据）
│   │   ├── sharded_parameters.py  # 分片参数管理
│   │   ├── tied_parameters.py     # 绑定参数管理
│   │   ├── tensor_parallel/       # 张量并行
│   │   │   ├── nn.py              # TP 线性层（Column/Row Linear、Embedding）
│   │   │   ├── functional.py      # TP 函数操作
│   │   │   ├── enum.py            # TP 模式枚举
│   │   │   └── distributed_differentiable_primitives.py  # 可微通信原语
│   │   ├── pipeline_parallel/     # 流水线并行
│   │   │   ├── block.py           # PipelineBlock（流水线块）
│   │   │   ├── engine.py          # PipelineEngine（AFAB/1F1B 调度）
│   │   │   ├── p2p.py             # 点对点通信
│   │   │   ├── state.py           # 流水线状态管理
│   │   │   ├── tensor_pointer.py  # TensorPointer（张量指针）
│   │   │   └── context_manager.py # 流水线上下文管理
│   │   └── data_parallel/         # 数据并行
│   │       └── utils.py           # DP 梯度同步工具
│   ├── optim/                     # 优化器
│   │   ├── base.py                # BaseOptimizer 抽象类
│   │   ├── zero.py                # ZeRO 分布式优化器
│   │   ├── gradient_accumulator.py # 梯度累积器
│   │   ├── clip_grads.py          # 梯度裁剪
│   │   └── named_optimizer.py     # 命名优化器
│   ├── fp8/                       # FP8 量化
│   │   ├── parameter.py           # FP8Parameter
│   │   ├── tensor.py              # FP8Tensor
│   │   ├── linear.py              # FP8Linear 层
│   │   ├── kernel.py              # FP8 矩阵乘法内核
│   │   ├── meta.py                # FP8 元数据
│   │   └── dtypes.py              # FP8 数据类型定义
│   ├── serialize/                 # 检查点序列化
│   │   ├── main.py                # 保存/加载入口
│   │   ├── weights.py             # 权重保存/加载
│   │   ├── optimizer.py           # 优化器状态保存/加载
│   │   ├── metadata.py            # 训练元数据
│   │   └── random.py              # 随机状态保存/加载
│   ├── generation/                # 文本生成
│   │   ├── decode.py              # 解码逻辑
│   │   ├── sampler.py             # 采样策略（Greedy/TopK/TopP）
│   │   └── generate_store.py      # KV Cache 存储
│   ├── eval/                      # 评估系统
│   │   ├── evaluation_tasks.py    # 评估任务
│   │   └── one_job_runner.py      # 单任务运行器
│   ├── scaling/                   # 参数缩放
│   │   └── parametrization.py     # 标准与 Spectral μP 参数化
│   ├── logging/                   # 日志系统
│   │   ├── base.py                # 日志基础
│   │   ├── logmixin.py            # 日志混入类
│   │   └── timers.py              # 计时器
│   ├── s3_checkpoints/            # S3 检查点上传
│   ├── distributed.py             # 分布式通信封装
│   ├── trainer.py                 # 核心训练器
│   ├── helpers.py                 # 辅助函数
│   ├── constants.py               # 常量定义
│   ├── random.py                  # 随机状态管理
│   ├── sanity_checks.py           # 健全性检查
│   ├── metrics_logging.py         # 指标日志
│   └── utils.py                   # 通用工具
├── examples/                      # 示例与扩展
│   ├── llama/                     # Llama 转换与训练
│   ├── mamba/                     # Mamba 模型支持
│   ├── moe/                       # MoE 训练示例
│   ├── doremi/                    # DoReMi 训练示例
│   ├── mup/                       # μParametrization 示例
│   └── inference/                 # 推理示例
├── tests/                         # 测试套件
├── docs/                          # 文档
├── scripts/                       # 辅助脚本
├── tools/                         # 工具（数据预处理）
├── run_train.py                   # 训练入口
├── run_generate.py                # 生成入口
├── run_evals.py                   # 评估入口
└── pyproject.toml                 # 项目配置
```

---

## 三、核心架构设计

### 3.1 整体架构图

```
┌─────────────────────────────────────────────────────────────────────┐
│                         用户入口层                                    │
│  run_train.py  │  run_generate.py  │  run_evals.py                  │
└───────┬────────┴────────┬──────────┴────────┬───────────────────────┘
        │                 │                    │
        ▼                 ▼                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    配置系统 (config/)                                 │
│  Config │ ParallelismArgs │ LlamaConfig │ Qwen2Config │ ...         │
└───────────────────────────┬─────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────────┐
│              分布式训练引擎 (DistributedTrainer)                      │
│  ┌─────────────┐  ┌──────────────┐  ┌────────────────────────┐     │
│  │ParallelCtx  │  │ Model Init   │  │ Optimizer/LR Scheduler │     │
│  └──────┬──────┘  └──────┬───────┘  └───────────┬────────────┘     │
│         │                │                       │                   │
│  ┌──────┴────────────────┴───────────────────────┴────────────┐     │
│  │                    训练循环                                  │     │
│  │  DataLoader → Forward → Backward → Grad Accum → Optim Step │     │
│  └────────────────────────────────────────────────────────────┘     │
└─────────────────────────────────────────────────────────────────────┘
                            │
        ┌───────────────────┼───────────────────┐
        ▼                   ▼                   ▼
┌──────────────┐  ┌──────────────┐  ┌──────────────────┐
│  并行策略层   │  │   模型层      │  │   数据层          │
│  TP / PP /   │  │  Llama /     │  │  Nanoset /       │
│  DP / EP /   │  │  Qwen2 /    │  │  Dataloader /    │
│  CP          │  │  Starcoder2 │  │  Collator        │
└──────────────┘  └──────────────┘  └──────────────────┘
        │                   │                   │
        ▼                   ▼                   ▼
┌──────────────┐  ┌──────────────┐  ┌──────────────────┐
│  通信原语     │  │  NN 组件      │  │  序列化/检查点    │
│  P2P /       │  │  Attention / │  │  Safetensors /   │
│  AllReduce / │  │  MLP /       │  │  S3 Upload /     │
│  AllGather   │  │  RoPE / MoE  │  │  Optim State     │
└──────────────┘  └──────────────┘  └──────────────────┘
```

### 3.2 配置系统

配置系统采用 `dataclass` 层级结构，通过 YAML 文件加载：

```
Config (根配置)
├── general: GeneralArgs              # seed、project、run 名称
├── model: ModelArgs                  # 模型配置 + 初始化方法
│   ├── model_config: LlamaConfig / Qwen2Config / Starcoder2Config
│   └── init_method: RandomInit / SpectralMupInit / ExistingCheckpointInit
├── parallelism: ParallelismArgs      # 并行策略配置
│   ├── tp, pp, dp                    # 三维并行度
│   ├── pp_engine                     # AFAB / 1F1B
│   ├── tp_mode                       # ALL_REDUCE / REDUCE_SCATTER
│   ├── expert_parallel_size          # 专家并行度
│   └── context_parallel_size         # 上下文并行度
├── tokens: TokensArgs                # 训练 token 相关配置
├── optimizer: OptimizerArgs          # 优化器 + LR 调度器
├── checkpoints: CheckpointArgs       # 检查点配置
├── logging: LoggingArgs              # 日志级别
├── data_stages: List[DataStageArgs]  # 多阶段数据配置
├── lighteval: LightEvalConfig        # 评估配置
└── s3_upload: S3UploadArgs           # S3 上传配置
```

**关键代码路径**：
- 配置加载：[config.py](file:///c:/Users/Administrator/nanotron/src/nanotron/config/config.py) → `get_config_from_file()`
- 模型配置：[models_config.py](file:///c:/Users/Administrator/nanotron/src/nanotron/config/models_config.py) → `LlamaConfig`、`Qwen2Config`
- 并行配置：[parallelism_config.py](file:///c:/Users/Administrator/nanotron/src/nanotron/config/parallelism_config.py) → `ParallelismArgs`

### 3.3 分布式训练引擎（Trainer）

`DistributedTrainer` 是整个框架的核心调度器，位于 [trainer.py](file:///c:/Users/Administrator/nanotron/src/nanotron/trainer.py)。

**初始化流程**：

```
DistributedTrainer.__init__()
│
├── 1. 加载配置 (Config)
├── 2. 初始化并行上下文 (ParallelContext)
├── 3. 设置日志级别
├── 4. 设置随机种子（每个 TP rank 不同）
├── 5. 初始化随机状态
├── 6. 初始化模型 (init_model)
│   └── CONFIG_TO_MODEL_CLASS 映射：
│       LlamaConfig → LlamaForTraining
│       Starcoder2Config → Starcoder2ForTraining
│       Qwen2Config → Qwen2ForTraining
├── 7. 初始化优化器和梯度累积器
├── 8. 初始化学习率调度器
├── 9. 加载检查点（如果存在）
├── 10. 初始化指标日志
└── 11. 初始化 S3 上传和评估运行器
```

**训练循环核心逻辑**：

```python
# 伪代码表示
for iteration_step in range(start_step, total_steps):
    # 1. 获取数据批次
    batch = next(dataloader)

    # 2. 前向传播（通过 PipelineEngine 调度）
    #    - AFAB: All Forward All Backward
    #    - 1F1B: One Forward One Backward
    loss = pipeline_engine.forward_backward(batch, model, grad_accumulator)

    # 3. 梯度同步（跨 DP）
    sync_gradients_across_dp()

    # 4. 梯度裁剪
    clip_grad_norm()

    # 5. 优化器步进
    optimizer.step()

    # 6. 学习率调度
    lr_scheduler.step()

    # 7. 检查点保存（按间隔）
    if should_save:
        save(model, optimizer, lr_scheduler, metadata)

    # 8. 评估（按间隔）
    if should_eval:
        lighteval_runner.eval()
```

### 3.4 并行上下文（ParallelContext）

`ParallelContext` 位于 [parallel/context.py](file:///c:/Users/Administrator/nanotron/src/nanotron/parallel/context.py)，是整个分布式训练的基础设施。

**进程组拓扑**：

```
World Size = EP × PP × DP × CP × TP

进程组布局 (5D reshape):
ranks[EP][PP][DP][CP][TP]

派生的进程组：
├── tp_pg   : TP 进程组（同 PP/DP/CP/EP 内的 TP 组）
├── pp_pg   : PP 进程组（同 TP/DP/CP/EP 内的 PP 组）
├── dp_pg   : DP 进程组（同 TP/PP/CP/EP 内的 DP 组）
├── cp_pg   : CP 进程组（同 TP/PP/DP/EP 内的 CP 组）
├── ep_pg   : EP 进程组（同 TP/PP/DP/CP 内的 EP 组）
├── mp_pg   : 模型并行组（TP + PP + EP 的组合）
├── dp_cp_pg: DP+CP 组合进程组
└── local_pg: 本地节点进程组
```

**关键约束**：`TP × PP × DP × CP × EP == WORLD_SIZE`

---

## 四、三维并行体系

### 4.1 张量并行（Tensor Parallelism）

张量并行的核心思想是将模型的权重矩阵切分到多个 GPU 上，每个 GPU 只持有部分权重。

**核心组件**：

| 组件 | 文件 | 功能 |
|------|------|------|
| `TensorParallelColumnLinear` | [nn.py](file:///c:/Users/Administrator/nanotron/src/nanotron/parallel/tensor_parallel/nn.py) | 切分输出维度（列切分） |
| `TensorParallelRowLinear` | [nn.py](file:///c:/Users/Administrator/nanotron/src/nanotron/parallel/tensor_parallel/nn.py) | 切分输入维度（行切分） |
| `TensorParallelEmbedding` | [nn.py](file:///c:/Users/Administrator/nanotron/src/nanotron/parallel/tensor_parallel/nn.py) | 词表切分嵌入 |
| `TensorParallelLinearMode` | [enum.py](file:///c:/Users/Administrator/nanotron/src/nanotron/parallel/tensor_parallel/enum.py) | ALL_REDUCE / REDUCE_SCATTER |

**TP 模式对比**：

```
模式一：ALL_REDUCE（经典 TP）
┌──────────┐     ┌──────────┐
│  GPU 0   │     │  GPU 1   │
│ W[0:N/2] │     │W[N/2:N]  │
└────┬─────┘     └────┬─────┘
     │ AllReduce       │
     └────────┬────────┘
              ▼
         完整输出

模式二：REDUCE_SCATTER（序列并行）
┌──────────┐     ┌──────────┐
│  GPU 0   │     │  GPU 1   │
│ W[0:N/2] │     │W[N/2:N]  │
└────┬─────┘     └────┬─────┘
     │ ReduceScatter   │
     ▼                 ▼
  输出[0:S/2]      输出[S/2:S]
```

**Column Linear 数据流**（以 MLP 的 gate_up_proj 为例）：

```
输入 X [seq, batch, hidden]
        │
        ▼ AllGather（如果 REDUCE_SCATTER 模式）
   X 完整 [seq, batch, hidden]
        │
        ▼ 本地矩阵乘
   X @ W_local → [seq, batch, intermediate/tp_size]
        │
        ▼ AllReduce / ReduceScatter
   输出（聚合后完整或按序列切分）
```

### 4.2 流水线并行（Pipeline Parallelism）

流水线并行将模型的不同层分配到不同 GPU 上，形成流水线。

**核心组件**：

| 组件 | 文件 | 功能 |
|------|------|------|
| `PipelineBlock` | [block.py](file:///c:/Users/Administrator/nanotron/src/nanotron/parallel/pipeline_parallel/block.py) | 流水线块，定义在哪个 rank 上计算 |
| `PipelineEngine` | [engine.py](file:///c:/Users/Administrator/nanotron/src/nanotron/parallel/pipeline_parallel/engine.py) | 流水线调度引擎（AFAB/1F1B） |
| `P2P` | [p2p.py](file:///c:/Users/Administrator/nanotron/src/nanotron/parallel/pipeline_parallel/p2p.py) | 点对点通信原语 |
| `TensorPointer` | [tensor_pointer.py](file:///c:/Users/Administrator/nanotron/src/nanotron/parallel/pipeline_parallel/tensor_pointer.py) | 张量指针（跨 rank 引用） |
| `PipelineTrainBatchState` | [state.py](file:///c:/Users/Administrator/nanotron/src/nanotron/parallel/pipeline_parallel/state.py) | 训练批次状态管理 |

**PipelineBlock 机制**：

```python
# PipelineBlock 是最细粒度的流水线单元
# 所有 PipelineBlock 定义存在于每个 rank，但只在指定 rank 上实例化

class PipelineBlock(nn.Module):
    def __init__(self, p2p, module_builder, module_kwargs, module_input_keys, module_output_keys):
        ...

    def build_and_set_rank(self, pp_rank):
        """只在当前 rank 等于 pp_rank 时实例化模块"""
        if pp_rank == dist.get_rank(self.p2p.pg):
            self.pp_block = self.module_builder(**self.module_kwargs)

    def forward(self, **kwargs):
        """通过 TensorPointer 机制传递张量"""
        if dist.get_rank(self.p2p.pg) != self.rank:
            # 非 compute rank：发送/跳过
            ...
        else:
            # Compute rank：接收并执行计算
            ...
```

**流水线调度策略**：

```
AFAB (All Forward All Backward):
时间 →  Rank 0    Rank 1    Rank 2
       F(mb0)    F(mb0)    F(mb0)
       F(mb1)    F(mb1)    F(mb1)
       F(mb2)    F(mb2)    F(mb2)
       B(mb0)    B(mb0)    B(mb0)
       B(mb1)    B(mb1)    B(mb1)
       B(mb2)    B(mb2)    B(mb2)

1F1B (One Forward One Backward):
时间 →  Rank 0    Rank 1    Rank 2
       F(mb0)    F(mb0)    F(mb0)
       F(mb1)    F(mb1)    F(mb1)
       B(mb0)    F(mb2)    F(mb2)
       F(mb2)    B(mb0)    B(mb0)
       B(mb1)    B(mb1)    B(mb1)
       B(mb2)    B(mb2)    B(mb2)
```

**TensorPointer 机制**：

```
TensorPointer 是一个轻量级对象，表示张量存在于另一个 rank 上。

当 PipelineBlock 的 forward 被调用时：
1. 如果当前 rank 不是 compute rank：
   - 输入中的 Tensor → 通过 P2P 发送到 compute rank
   - 输入中的 TensorPointer → 跳过（已在其他地方处理）
2. 如果当前 rank 是 compute rank：
   - 从前一个 rank 接收 Tensor
   - 执行实际计算
   - 输出 Tensor 或 TensorPointer
```

### 4.3 数据并行（Data Parallelism）

数据并行是最基础的并行策略，每个 GPU 持有完整的模型副本，处理不同的数据批次。

**关键机制**：
- 梯度同步：`sync_gradients_across_dp()` — 在 DP 进程组内 AllReduce 梯度
- DDP 集成：使用 PyTorch 的 `DistributedDataParallel`
- ZeRO Stage 1：优化器状态分片（见[优化器章节](#83-zero-优化器)）

**数据分配策略**：
```
每个 DP rank 获取不同的数据批次
DP rank 0: batch_0, batch_N, batch_2N, ...
DP rank 1: batch_1, batch_N+1, batch_2N+1, ...
...
梯度在所有 DP rank 间 AllReduce 同步
```

### 4.4 专家并行与上下文并行

**专家并行（Expert Parallelism, EP）**：
- 专门用于 MoE 模型
- 将不同的专家分配到不同 GPU 上
- `expert_parallel_size` 控制并行度
- 通过 `ep_pg` 进程组管理通信

**上下文并行（Context Parallelism, CP）**：
- 用于超长序列训练
- 将序列长度切分到多个 GPU
- 配合 Ring Attention 使用
- `context_parallel_size` 控制并行度

---

## 五、模型架构层

### 5.1 模型基类 NanotronModel

`NanotronModel` 位于 [base.py](file:///c:/Users/Administrator/nanotron/src/nanotron/models/base.py)，是所有模型的抽象基类。

```python
class NanotronModel(nn.Module, LoggingCollectorMixin, metaclass=ABCMeta):
    parallel_context: ParallelContext
    config: NanotronConfigs
    input_pp_rank: int     # 输入所在的 PP rank
    output_pp_rank: int    # 输出所在的 PP rank

    # 抽象方法
    @abstractmethod
    def init_model_randomly(self, config): ...

    # 可选覆写
    def tie_custom_params(self) -> None: ...
    def get_embeddings_lm_head_tied_names(self) -> list[str]: ...
    def before_tbi_sanity_checks(self) -> None: ...
    def after_tbi_sanity_checks(self) -> None: ...
```

**模型构建流程**：

```
build_model(model_config, parallel_context, ...)
│
├── 1. 实例化模型类（如 LlamaForTraining）
├── 2. 调用 model.init_model_randomly() 初始化参数
├── 3. 绑定参数（tie_parameters）
│   └── 例如：embedding 和 lm_head 权重共享
├── 4. 为绑定权重创建进程组
├── 5. 健全性检查（sanity_check）
└── 6. 包装 DDP（如果 DP > 1）
```

### 5.2 支持的模型

#### Llama（[llama.py](file:///c:/Users/Administrator/nanotron/src/nanotron/models/llama.py)）

```
LlamaForTraining
├── model: LlamaModel
│   ├── token_position_embeddings: PipelineBlock
│   │   ├── token_embedding: TensorParallelEmbedding
│   │   └── rotary_embedding: RotaryEmbedding / LlamaRotaryEmbedding
│   ├── layers: List[PipelineBlock]  (每个 Transformer 层)
│   │   └── LlamaDecoderLayer
│   │       ├── input_layernorm: TritonRMSNorm
│   │       ├── self_attn: CoreAttention + TP QKV Linear
│   │       ├── post_attention_layernorm: TritonRMSNorm
│   │       └── mlp: MLP (gate_up_proj + down_proj)
│   └── final_norm: PipelineBlock → TritonRMSNorm
└── lm_head: PipelineBlock → TensorParallelColumnLinear
```

**Llama Decoder Layer 数据流**：

```
hidden_states
    │
    ▼ input_layernorm
    │
    ▼ QKV Projection (TensorParallelColumnLinear)
    │   Q: [seq, batch, num_heads/tp, head_dim]
    │   K: [seq, batch, num_kv_heads/tp, head_dim]
    │   V: [seq, batch, num_kv_heads/tp, head_dim]
    │
    ▼ Rotary Embedding (RoPE)
    │
    ▼ CoreAttention (Flash Attention)
    │   output: [seq, batch, hidden_size]
    │
    ▼ O Projection (TensorParallelRowLinear)
    │
    ▼ Residual Connection
    │
    ▼ post_attention_layernorm
    │
    ▼ MLP
    │   gate_up_proj (TensorParallelColumnLinear) → GLU → down_proj (TensorParallelRowLinear)
    │
    ▼ Residual Connection
    │
    ▼ 输出 hidden_states
```

#### Qwen2（[qwen.py](file:///c:/Users/Administrator/nanotron/src/nanotron/models/qwen.py)）

- 支持 GQA（Grouped Query Attention）
- 支持 MoE（Mixture of Experts）
- 支持 Sliding Window Attention
- 支持 Flex Attention（文档级掩码）
- 支持 Ring Attention（长序列）

#### Starcoder2（[starcoder2.py](file:///c:/Users/Administrator/nanotron/src/nanotron/models/starcoder2.py)）

- 代码生成模型架构
- 支持注意力偏置

### 5.3 神经网络组件（nn 模块）

| 组件 | 文件 | 描述 |
|------|------|------|
| `CoreAttention` | [attention.py](file:///c:/Users/Administrator/nanotron/src/nanotron/nn/attention.py) | 支持 Flash/Flex/Ring 多种注意力后端 |
| `RotaryEmbedding` | [rotary.py](file:///c:/Users/Administrator/nanotron/src/nanotron/nn/rotary.py) | 旋转位置编码 |
| `TritonRMSNorm` | [layer_norm.py](file:///c:/Users/Administrator/nanotron/src/nanotron/nn/layer_norm.py) | Triton 加速的 RMSNorm |
| `GLUActivation` | [activations.py](file:///c:/Users/Administrator/nanotron/src/nanotron/nn/activations.py) | 门控线性单元激活 |
| `Router` | [moe.py](file:///c:/Users/Administrator/nanotron/src/nanotron/nn/moe.py) | MoE 路由器（Top-K 选择） |
| `GroupedMLP` | [moe.py](file:///c:/Users/Administrator/nanotron/src/nanotron/nn/moe.py) | MoE 分组 MLP |

---

## 六、参数管理系统

### 6.1 NanotronParameter

`NanotronParameter` 位于 [parameters.py](file:///c:/Users/Administrator/nanotron/src/nanotron/parallel/parameters.py)，继承自 `nn.Parameter`，增加了分布式元数据。

```python
class NanotronParameter(nn.Parameter):
    # 两种分布式属性（可同时具有）
    is_sharded: bool       # 是否跨设备分片
    is_tied: bool          # 是否与其他参数绑定

    # 分片信息
    sharded_info: ShardedInfo
    │   ├── global_ranks: Tuple[int, ...]        # 持有分片的全局 rank
    │   ├── local_global_slices_pairs            # 本地/全局切片映射
    │   └── unsharded_shape: Tuple[int, ...]     # 未分片的完整形状

    # 绑定信息
    tied_info: TiedInfo
    │   ├── name: str                            # 参数名
    │   ├── root_module: nn.Module               # 所属模块
    │   ├── global_ranks: Tuple[int, ...]        # 绑定的全局 rank
    │   └── reduce_op: Optional[ReduceOp]        # 梯度归约操作
```

### 6.2 分片参数（Sharded Parameters）

分片参数用于张量并行，将参数沿特定维度切分到多个设备。

```python
# 创建分片参数的流程
SplitConfig(split_dim=0, contiguous_chunks=None)
    │
    ▼ create_sharded_parameter_from_config()
    │
    ▼ mark_all_parameters_in_module_as_sharded()
    │
    ▼ NanotronParameter.mark_as_sharded()
```

**示例**：`TensorParallelColumnLinear` 的权重分片

```
原始权重: [4096, 4096]  (out_features × in_features)
TP=2 时：
  Rank 0: weight[0:2048, :]   (local_slice)
  Rank 1: weight[2048:4096, :] (local_slice)

ShardedInfo:
  global_ranks = (0, 1)
  local_global_slices_pairs = (
    SlicesPair(
      local_slices = (slice(0, 2048), slice(None)),
      global_slices = (slice(0, 2048), slice(None))   # Rank 0
    ),
  )
  unsharded_shape = (4096, 4096)
```

### 6.3 绑定参数（Tied Parameters）

绑定参数用于权重共享场景（如 Embedding 和 LM Head）。

```python
# 绑定参数流程
tie_parameters(root_module, ties, parallel_context, reduce_op)
│
├── 1. 验证所有绑定参数在同一 DP rank
├── 2. 在同一设备上：替换为同一个 Parameter 对象
├── 3. 跨设备：标记为 tied，添加 TiedInfo 元数据
└── 4. 创建专门的进程组用于梯度同步

# 梯度同步
sync_tied_weights_gradients(model, parallel_context)
│
└── 对每个 tied 参数，在绑定 rank 间 AllReduce 梯度
```

---

## 七、数据流水线

### 7.1 数据集构建

Nanoset 是 Nanotron 的核心数据集实现，位于 [nanoset.py](file:///c:/Users/Administrator/nanotron/src/nanotron/data/nanoset.py)。

```
Nanoset
├── 基于 DatatroveFolderDataset（datatrove 格式）
├── 支持多数据集加权混合
├── 预计算索引缓存
├── 支持 position_ids 直接构建
└── 支持 EOS token 处理
```

**数据预处理流程**：

```
原始文本 → tools/preprocess_data.py → Datatrove 格式（.ds 文件）
                                         │
                                         ▼
                                    Nanoset 加载
                                         │
                                         ▼
                                   CLM Collator
                                    │        │
                                    ▼        ▼
                              input_ids   labels (shifted)
                              attention_mask
                              position_ids
```

### 7.2 数据加载器

数据加载器位于 [dataloader.py](file:///c:/Users/Administrator/nanotron/src/nanotron/data/dataloader.py)，负责：

1. **数据分发**：每个 DP rank 获取不同数据
2. **TP 同步**：同一 TP 组内数据一致
3. **PP 分发**：只有输入 PP rank 获取实际数据，其他 rank 获取 `TensorPointer`
4. **健全性检查**：验证 DP 间数据不同、TP 间数据同步

```python
# 数据批次在并行组间的分布
batch = {
    "input_ids": Tensor 或 TensorPointer,    # 实际数据或指针
    "input_mask": Tensor 或 TensorPointer,
    "position_ids": Tensor 或 TensorPointer,
    "labels": Tensor 或 TensorPointer,
}

# 对于 PP：
# - input_pp_rank: 获取实际 Tensor
# - 其他 PP rank: 获取 TensorPointer（表示数据在别的 rank 上）
```

### 7.3 数据流图

```
┌──────────────────────────────────────────────────────────┐
│                    数据加载流程                            │
│                                                          │
│  Nanoset / HuggingFace Dataset                           │
│         │                                                │
│         ▼                                                │
│  Sampler (按 DP rank 分配不同数据)                        │
│         │                                                │
│         ▼                                                │
│  CLM Collator (构建 input_ids, labels, mask, positions)  │
│         │                                                │
│         ▼                                                │
│  sanity_check_dataloader()                               │
│  ├── 验证 DP 间数据不同                                   │
│  └── 验证 TP 间数据同步                                   │
│         │                                                │
│         ▼                                                │
│  micro_batch (移至 CUDA)                                  │
│  ├── DP rank 0: 实际 Tensor                              │
│  ├── DP rank 1: 不同的实际 Tensor                         │
│  └── PP rank ≠ input_rank: TensorPointer                 │
└──────────────────────────────────────────────────────────┘
```

---

## 八、优化器与梯度管理

### 8.1 优化器架构

```
BaseOptimizer (抽象基类)
├── zero_grad()
├── step()
├── state_dict() / load_state_dict()
└── InheritFromOtherOptimizer
    └── ZeroDistributedOptimizer (ZeRO Stage 1)
```

### 8.2 梯度累积器

`GradientAccumulator` 位于 [gradient_accumulator.py](file:///c:/Users/Administrator/nanotron/src/nanotron/optim/gradient_accumulator.py)。

```python
class FP32GradientAccumulator(GradientAccumulator):
    """在 FP32 中累积梯度，支持跨 micro-batch 累积"""

    def backward(self, loss):
        """反向传播，累积 FP32 梯度"""

    def sync_gradients_across_dp(self, dp_pg, reduce_op, reduce_scatter):
        """跨 DP 进程组同步梯度"""

    def get_parameter_for_optimizer(self, name):
        """返回带有累积梯度的参数（供优化器使用）"""
```

**梯度累积流程**：

```
Micro-batch 1 → Loss → Backward → 累积 FP32 梯度
Micro-batch 2 → Loss → Backward → 累积 FP32 梯度
...
Micro-batch N → Loss → Backward → 累积 FP32 梯度
                                    │
                                    ▼
                         sync_gradients_across_dp()
                                    │
                                    ▼
                              optimizer.step()
```

### 8.3 ZeRO 优化器

`ZeroDistributedOptimizer` 位于 [zero.py](file:///c:/Users/Administrator/nanotron/src/nanotron/optim/zero.py)，实现 ZeRO Stage 1。

```
ZeRO Stage 1 工作原理：
1. 将优化器状态（momentum, variance 等）分片到 DP rank
2. 每个 rank 只维护 1/DP_size 的优化器状态
3. 参数和梯度仍然完整存在于每个 rank

参数分片映射：
param_name_to_dp_rank_offsets = {
    "weight": {rank_0: (0, N/DP), rank_1: (N/DP, 2N/DP), ...},
    "bias":   {rank_0: (0, M/DP), rank_1: (M/DP, 2M/DP), ...},
}
```

---

## 九、FP8 量化支持

FP8 模块位于 [fp8/](file:///c:/Users/Administrator/nanotron/src/nanotron/fp8/)，提供 FP8 混合精度训练能力。

```
FP8 量化架构：

FP8Parameter (extends nn.Parameter)
│   └── 将权重存储为 FP8 格式

FP8Tensor (extends torch.Tensor)
│   ├── 量化/反量化逻辑
│   └── 缩放因子管理

FP8Linear (extends nn.Linear)
│   ├── weight: FP8Parameter
│   ├── fp8_meta: FP8LinearMeta
│   │   ├── input_grad: FP8Meta (E4M3)
│   │   ├── weight_grad: FP8Meta (E4M3)
│   │   └── output_grad: FP8Meta (E5M2)
│   └── _FP8Matmul (autograd Function)
│       ├── forward: FP8 量化输入 → FP8 矩阵乘法 → FP32 输出
│       └── backward: FP8 梯度计算

FP8 数据类型：
├── FP8E4M3 (E4M3FN): 前向传播（更大动态范围）
└── FP8E5M2 (E5M2): 反向传播（更高精度梯度）
```

**缩放因子更新**：

```python
scaling_factor = amax / (fp8_max * margin)
# amax: 当前 batch 的最大绝对值
# fp8_max: FP8 格式的最大可表示值
# margin: 安全余量
```

---

## 十、检查点与序列化

序列化系统位于 [serialize/](file:///c:/Users/Administrator/nanotron/src/nanotron/serialize/)，使用 Safetensors 格式。

```
检查点保存流程：
save(config, model, optimizer, lr_scheduler, parallel_context, metadata, root_folder)
│
├── 1. 保存配置 → config.yaml
├── 2. 保存权重 → Safetensors 格式
│   └── 每个 rank 保存自己的参数分片
├── 3. 保存优化器状态
│   └── ZeRO: 每个 rank 只保存自己的分片
├── 4. 保存学习率调度器状态
├── 5. 保存训练元数据
│   └── consumed_train_samples, last_train_step, data_stages
└── 6. 保存随机状态

检查点目录结构：
checkpoints/
├── config.yaml
├── metadata.json
├── model/
│   ├── model_tp_0_pp_0.safetensors
│   ├── model_tp_1_pp_0.safetensors
│   └── ...
├── optimizer/
│   ├── optimizer_tp_0_pp_0_dp_0.safetensors
│   └── ...
├── lr_scheduler/
│   └── lr_scheduler_tp_0_pp_0_dp_0.safetensors
└── random_states/
    └── ...
```

**S3 上传**：支持异步上传检查点到 S3 存储，使用 `s5cmd` 高性能工具。

---

## 十一、推理与生成

生成系统位于 [generation/](file:///c:/Users/Administrator/nanotron/src/nanotron/generation/)。

```
生成流程：
decode(parallel_context, model, tokenizer, prompts, generation_args)
│
├── 1. Tokenize 输入文本
├── 2. 自回归生成循环
│   ├── Forward pass（使用 PipelineEvalBatchState）
│   ├── 采样下一个 token
│   │   ├── GreedySampler: argmax
│   │   ├── TopKSampler: top-k 采样
│   │   ├── TopPSampler: nucleus 采样
│   │   └── BasicSampler: 温度采样
│   └── 更新 KV Cache (Store)
├── 3. Detokenize 输出
└── 4. 返回 GenerationOutput
```

---

## 十二、评估系统

评估系统与 [LightEval](https://github.com/huggingface/lighteval) 集成，位于 [eval/](file:///c:/Users/Administrator/nanotron/src/nanotron/eval/)。

```
LightEvalRunner
├── 配置：LightEvalConfig（任务、批次大小等）
├── 触发时机：
│   ├── 训练中按间隔评估
│   └── S3 上传后自动评估
└── 输出：WandB 上传评估结果
```

---

## 十三、参数初始化与缩放

参数化系统位于 [scaling/parametrization.py](file:///c:/Users/Administrator/nanotron/src/nanotron/scaling/parametrization.py)。

```
ParametrizationMethod
├── STANDARD: 标准初始化
│   └── StandardParametrizator
│       ├── Column Linear: N(0, std)
│       ├── Row Linear: N(0, std / scaling_factor)
│       ├── Embedding: N(0, std)
│       └── Layer Norm: weight=1, bias=0
│
└── SPECTRAL_MUP: Spectral μP 初始化
    └── SpectralMupParametrizator
        ├── 基于 μTransfer 理论
        ├── 缩放因子: sqrt(1/d_h) 而非 1/sqrt(d_h)
        └── 允许超参数从小模型迁移到大模型

缩放方法 (InitScalingMethod):
├── NUM_LAYERS: 缩放因子 = sqrt(2 * num_layers)
└── 其他自定义方法
```

---

## 十四、完整训练数据流

以下是 Nanotron 一次完整训练迭代的数据流：

```
┌─────────────────────────────────────────────────────────────────────┐
│                        完整训练迭代数据流                              │
└─────────────────────────────────────────────────────────────────────┘

1. 数据获取
   Nanoset → Sampler → Collator → DataLoader
       │
       ▼
   micro_batch = {
       input_ids: [seq_len, batch_size],
       input_mask: [seq_len, batch_size],
       position_ids: [seq_len, batch_size],
       labels: [seq_len, batch_size]
   }

2. 前向传播（Pipeline Engine 调度）
   ┌──────────────────────────────────────────────────┐
   │ PP Rank 0          PP Rank 1          PP Rank 2  │
   │                                                    │
   │ Embedding →        Transformer       → LM Head   │
   │ + Layer 0-5        Layer 6-11        + Loss       │
   │                                                    │
   │ TP: 权重分片       TP: 权重分片       TP: 权重分片 │
   │ AllReduce/         AllReduce/         AllReduce/  │
   │ ReduceScatter      ReduceScatter      ReduceScatter│
   └──────────────────────────────────────────────────┘
       │
       ▼
   loss = output["loss"] / nb_microbatches

3. 反向传播
   loss.backward() 或 grad_accumulator.backward(loss)
       │
       ▼
   梯度通过 Pipeline 反向传播
   ├── 激活重计算（如果启用 recompute_layer）
   └── 跨 PP rank 发送梯度

4. 梯度处理
   ├── FP32 梯度累积（跨 micro-batch）
   ├── 跨 DP 梯度同步 (AllReduce / ReduceScatter)
   ├── 绑定参数梯度同步 (sync_tied_weights_gradients)
   └── 梯度裁剪 (clip_grad_norm)

5. 优化器步进
   optimizer.step()
   ├── ZeRO: 只更新本地分片的优化器状态
   └── 参数更新后 AllGather 同步

6. 学习率调度
   lr_scheduler.step()

7. 日志与监控
   ├── WandB 日志
   ├── 训练吞吐量统计
   └── 内存使用监控
```

---

## 十五、系统学习路径

### 阶段一：基础入门

**目标**：理解项目整体结构，能够运行基础训练

| 里程碑 | 学习内容 | 关键文件 | 预期成果 |
|--------|---------|---------|---------|
| M1.1 | 项目安装与环境配置 | `pyproject.toml`、`README.md` | 成功安装并运行 tiny Llama 训练 |
| M1.2 | 理解 YAML 配置系统 | `config/config.py`、`examples/config_tiny_llama.yaml` | 能修改配置并理解每个参数含义 |
| M1.3 | 训练入口与基本流程 | `run_train.py`、`trainer.py` 前 200 行 | 理解训练启动到循环开始的流程 |
| M1.4 | 模型配置与初始化 | `config/models_config.py`、`LlamaConfig` | 理解模型配置如何映射到模型实例 |

**推荐学习顺序**：

```
1. 阅读 README.md，按步骤安装并运行 tiny Llama 训练
2. 阅读 config_tiny_llama.yaml，对照 config.py 理解每个配置项
3. 阅读 run_train.py → DistributedTrainer.__init__() 的前半部分
4. 对照 LlamaConfig 字段，理解模型配置如何影响模型结构
```

**实践练习**：
- 修改 `micro_batch_size`、`sequence_length` 等参数观察训练变化
- 尝试从已有检查点恢复训练

---

### 阶段二：核心机制

**目标**：深入理解模型构建、参数管理和数据流

| 里程碑 | 学习内容 | 关键文件 | 预期成果 |
|--------|---------|---------|---------|
| M2.1 | NanotronModel 基类 | `models/base.py` | 理解模型抽象接口和生命周期 |
| M2.2 | Llama 模型实现 | `models/llama.py` | 理解完整 Transformer 模型构建 |
| M2.3 | PipelineBlock 机制 | `parallel/pipeline_parallel/block.py` | 理解模型如何被切分到不同 PP rank |
| M2.4 | NanotronParameter | `parallel/parameters.py` | 理解参数的分片和绑定元数据 |
| M2.5 | 数据加载流程 | `data/dataloader.py`、`data/nanoset.py` | 理解数据从磁盘到 GPU 的完整路径 |

**推荐学习顺序**：

```
1. 阅读 base.py，理解 NanotronModel 的接口设计
2. 阅读 llama.py，从 LlamaForTraining.__init__() 开始
   - 注意 PipelineBlock 如何包装每个子模块
   - 注意 TensorParallelColumnLinear / RowLinear 的使用
3. 阅读 block.py，理解 PipelineBlock.forward() 的 TensorPointer 机制
4. 阅读 parameters.py，理解 ShardedInfo 和 TiedInfo 的数据结构
5. 阅读 dataloader.py，理解数据在不同并行组间的分发逻辑
```

**实践练习**：
- 在 `LlamaDecoderLayer.forward()` 中添加日志，追踪张量形状变化
- 修改 `PipelineBlock` 的 rank 分配，观察模型在不同 PP rank 上的分布

---

### 阶段三：并行策略

**目标**：掌握三维并行的实现细节和通信模式

| 里程碑 | 学习内容 | 关键文件 | 预期成果 |
|--------|---------|---------|---------|
| M3.1 | ParallelContext 与进程组 | `parallel/context.py` | 理解 5D 进程组拓扑结构 |
| M3.2 | 张量并行实现 | `parallel/tensor_parallel/nn.py`、`functional.py` | 理解 Column/Row Linear 的切分和通信 |
| M3.3 | 流水线并行调度 | `parallel/pipeline_parallel/engine.py` | 理解 AFAB 和 1F1B 调度策略 |
| M3.4 | P2P 通信 | `parallel/pipeline_parallel/p2p.py`、`state.py` | 理解跨 rank 的张量传输机制 |
| M3.5 | 数据并行与梯度同步 | `parallel/data_parallel/utils.py`、`optim/gradient_accumulator.py` | 理解 DP 梯度同步和 ZeRO |

**推荐学习顺序**：

```
1. 阅读 context.py，画出进程组拓扑图
   - 理解 tp_pg, pp_pg, dp_pg, ep_pg, cp_pg 的关系
   - 手动计算 8 GPU 下 TP=2, PP=2, DP=2 的进程组分配
2. 阅读 tensor_parallel/nn.py
   - TensorParallelColumnLinear: 权重列切分 + AllReduce/ReduceScatter
   - TensorParallelRowLinear: 权重行切分 + AllGather
   - 对照 functional.py 理解 column_linear / row_linear 函数
3. 阅读 pipeline_parallel/engine.py
   - 理解 forward() 和 backward() 的调度逻辑
   - 对比 AFAB 和 1F1B 的实现差异
4. 阅读 p2p.py，理解元数据交换和实际数据传输
5. 阅读 gradient_accumulator.py 和 zero.py
   - 理解 FP32 梯度累积的必要性
   - 理解 ZeRO Stage 1 的参数分片策略
```

**实践练习**：
- 使用不同 TP/PP/DP 配置运行训练，观察通信开销差异
- 阅读测试文件 `tests/test_tensor_parallel.py` 和 `tests/test_pipeline_parallel.py`

---

### 阶段四：高级特性

**目标**：掌握 FP8 量化、MoE、参数缩放等高级功能

| 里程碑 | 学习内容 | 关键文件 | 预期成果 |
|--------|---------|---------|---------|
| M4.1 | FP8 混合精度训练 | `fp8/parameter.py`、`tensor.py`、`linear.py` | 理解 FP8 量化和反量化流程 |
| M4.2 | MoE 架构 | `nn/moe.py`、`examples/moe/` | 理解专家路由和分组计算 |
| M4.3 | 参数初始化与 μP | `scaling/parametrization.py` | 理解标准初始化和 Spectral μP |
| M4.4 | 检查点与序列化 | `serialize/main.py`、`weights.py` | 理解检查点保存/恢复机制 |
| M4.5 | 推理与生成 | `generation/decode.py`、`sampler.py` | 理解自回归生成流程 |

**推荐学习顺序**：

```
1. 阅读 fp8/ 目录
   - parameter.py: FP8Parameter 如何替换标准 Parameter
   - tensor.py: FP8Tensor 的量化/反量化
   - linear.py: _FP8Matmul 的前向/反向传播
   - dtypes.py: FP8E4M3 和 FP8E5M2 的区别
2. 阅读 nn/moe.py
   - Router: Top-K 路由选择
   - GroupedMLP: 批量专家计算
   - 对照 examples/moe/ 理解完整 MoE 训练
3. 阅读 scaling/parametrization.py
   - StandardParametrizator: 各层初始化策略
   - SpectralMupParametrizator: μTransfer 缩放规则
4. 阅读 serialize/ 目录
   - 理解 Safetensors 格式的优势
   - 理解拓扑无关的检查点设计
5. 阅读 generation/ 目录
   - 理解 PipelineEvalBatchState 与训练状态的区别
   - 理解 KV Cache 管理
```

**实践练习**：
- 启用 FP8 训练，对比精度和速度
- 运行 MoE 示例，观察专家负载均衡
- 实现自定义参数初始化策略

---

### 阶段五：实战与扩展

**目标**：能够扩展框架，添加新模型或新功能

| 里程碑 | 学习内容 | 关键文件 | 预期成果 |
|--------|---------|---------|---------|
| M5.1 | 添加新模型架构 | `models/base.py`、`models/llama.py` | 能实现自定义 NanotronModel |
| M5.2 | 自定义数据加载 | `data/dataloader_builder.py`、`examples/custom-dataloader/` | 能集成自定义数据集 |
| M5.3 | DoReMi 训练 | `examples/doremi/` | 理解域加权训练流程 |
| M5.4 | Ring Attention 长序列 | `nn/ring_attention.py`、`nn/llama3_ring_attention.py` | 理解超长序列训练 |
| M5.5 | 分布式调试 | `sanity_checks.py`、`docs/debugging.md` | 能诊断和解决分布式训练问题 |

**推荐学习顺序**：

```
1. 以 LlamaForTraining 为模板，实现一个自定义模型
   - 继承 NanotronModel
   - 使用 PipelineBlock 包装各层
   - 使用 TP 线性层实现并行
   - 在 CONFIG_TO_MODEL_CLASS 中注册
2. 阅读 examples/custom-dataloader/
   - 理解如何替换默认数据加载器
3. 阅读 examples/doremi/
   - 理解参考模型训练 → 代理模型训练 → 域权重调整的流程
4. 阅读 Ring Attention 实现
   - 理解序列切分和环形通信模式
5. 阅读 sanity_checks.py 和 docs/debugging.md
   - 理解常见的分布式训练问题及解决方案
```

**实践练习**：
- 实现一个 GPT-2 风格的模型
- 添加自定义的注意力机制
- 贡献代码到项目（参考 `CONTRIBUTING.md`）

---

## 附录：关键概念速查表

| 概念 | 缩写 | 描述 |
|------|------|------|
| Tensor Parallelism | TP | 权重矩阵切分到多个 GPU |
| Pipeline Parallelism | PP | 模型层切分到多个 GPU |
| Data Parallelism | DP | 数据切分，模型完整复制 |
| Expert Parallelism | EP | MoE 专家切分到多个 GPU |
| Context Parallelism | CP | 序列长度切分到多个 GPU |
| AllForwardAllBackward | AFAB | 先全部前向再全部反向的 PP 调度 |
| OneForwardOneBackward | 1F1B | 交替前向反向的 PP 调度 |
| ReduceScatter | RS | 先归约再散射，用于序列并行 |
| ZeRO | Zero | 优化器状态分片（Stage 1） |
| μTransfer | μP | 超参数从小模型迁移到大模型 |
| FP8 | FP8 | 8-bit 浮点数混合精度训练 |
| MoE | MoE | 混合专家模型 |
| GQA | GQA | 分组查询注意力 |
| RoPE | RoPE | 旋转位置编码 |
| TensorPointer | TP | 跨 rank 张量引用（非实际数据） |
| PipelineBlock | PB | 流水线并行最小调度单元 |
| Safetensors | ST | 安全的张量序列化格式 |

---

## 附录：核心文件阅读清单

按优先级排序的必读文件：

```
🔴 必读（理解核心流程）：
1. src/nanotron/trainer.py              # 训练器主逻辑
2. src/nanotron/parallel/context.py     # 并行上下文
3. src/nanotron/models/base.py          # 模型基类
4. src/nanotron/models/llama.py         # 具体模型实现
5. src/nanotron/config/config.py        # 配置系统

🟡 重要（理解并行策略）：
6. src/nanotron/parallel/tensor_parallel/nn.py     # TP 线性层
7. src/nanotron/parallel/pipeline_parallel/block.py # PP 块
8. src/nanotron/parallel/pipeline_parallel/engine.py # PP 引擎
9. src/nanotron/parallel/parameters.py              # 参数管理
10. src/nanotron/optim/gradient_accumulator.py      # 梯度累积

🟢 进阶（理解高级特性）：
11. src/nanotron/fp8/linear.py          # FP8 线性层
12. src/nanotron/nn/moe.py             # MoE 组件
13. src/nanotron/optim/zero.py          # ZeRO 优化器
14. src/nanotron/serialize/main.py      # 检查点系统
15. src/nanotron/scaling/parametrization.py # 参数初始化
```
