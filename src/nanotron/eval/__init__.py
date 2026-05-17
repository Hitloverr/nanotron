# flake8: noqa: F401
"""
LightEvalRunner
├── 配置：LightEvalConfig（任务、批次大小等）
├── 触发时机：
│   ├── 训练中按间隔评估
│   └── S3 上传后自动评估
└── 输出：WandB 上传评估结果

"""
from .one_job_runner import LightEvalRunner
