import warnings

from nanotron.fp8.dtypes import DTypes  # noqa
from nanotron.fp8.linear import FP8Linear  # noqa
from nanotron.fp8.parameter import FP8Parameter  # noqa
from nanotron.fp8.tensor import FP8Tensor  # noqa


"""
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

1. E4M3 适用场景
前向传播：激活值、特征图等分布集中，动态范围小，需更高精度维持输出质量
权重存储：多数模型权重分布在 [-1,1] 区间，E4M3 精度足够且存储效率高
推理阶段：对精度敏感，需最小化量化损失，保证生成结果流畅性
2. E5M2 适用场景
反向传播：梯度值动态范围极大（从 10⁻⁸到 10⁴），E5M2 避免梯度下溢 / 上溢
优化器状态：Adam 等优化器的动量项、二阶矩统计需要大动态范围
多尺度特征：部分模型（如 ViT、扩散模型）的特征图存在极端值，E5M2 更安全

"""
try:
    import transformer_engine as te  # noqa
    import transformer_engine_extensions as tex  # noqa
except ImportError:
    warnings.warn("Please install Transformer engine for FP8 training!")
