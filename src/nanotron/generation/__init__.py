from .sampler import BasicSampler, GreedySampler, Sampler, SamplerType, TopKSampler, TopPSampler

"""
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

"""
__all__ = ["BasicSampler", "GreedySampler", "Sampler", "SamplerType", "TopKSampler", "TopPSampler"]
