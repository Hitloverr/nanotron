# flake8: noqa
from nanotron.serialize.main import *
from nanotron.serialize.optimizer import *
from nanotron.serialize.random import *
from nanotron.serialize.weights import *
from nanotron.serialize.metadata import *

"""
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
**S3 上传**：支持异步上传检查点到 S3 存储，使用 `s5cmd` 高性能工具。
"""