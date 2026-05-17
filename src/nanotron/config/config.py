"""
Nanotron 配置系统 —— 定义训练大语言模型所需的所有配置参数。

本模块使用 Python dataclass 定义了完整的配置层次结构，包括：
    - GeneralArgs: 通用训练参数（项目名、运行名、随机种子等）
    - TokensArgs: 序列长度、批大小、训练步数等 token 相关参数
    - ModelArgs: 模型架构和初始化参数
    - OptimizerArgs: 优化器和学习率调度器参数
    - ParallelismArgs: 3D 并行配置（TP/PP/DP/CP/EP）
    - DataArgs / DatasetStageArgs: 数据集配置
    - CheckpointsArgs: 检查点保存和恢复参数
    - LoggingArgs / MetricsLoggingArgs: 日志和指标记录参数
    - ProfilerArgs: 性能分析参数
    - GenerationArgs: 文本生成参数
    - Config: 顶层配置类，组合所有子配置

配置可通过 YAML 文件加载，使用 dacite 库进行类型安全的反序列化。
"""

import datetime
import glob
import os
from dataclasses import dataclass, fields
from pathlib import Path
from typing import List, Optional, Type, Union

import dacite
import torch
import yaml
from dacite import from_dict
from datasets.download.streaming_download_manager import xPath
from transformers import AutoTokenizer
from yaml.loader import SafeLoader

from nanotron.config.lighteval_config import LightEvalConfig
from nanotron.config.models_config import ExistingCheckpointInit, NanotronConfigs, RandomInit, SpectralMupInit
from nanotron.config.parallelism_config import ParallelismArgs
from nanotron.config.utils_config import (
    InitScalingMethod,
    RecomputeGranularity,
    cast_str_to_pipeline_engine,
    cast_str_to_torch_dtype,
    serialize,
)
from nanotron.generation.sampler import SamplerType
from nanotron.logging import get_logger, human_format
from nanotron.parallel.pipeline_parallel.engine import PipelineEngine
from nanotron.parallel.tensor_parallel.nn import TensorParallelLinearMode
from nanotron.config.models_config import Qwen2Config

logger = get_logger(__name__)

DEFAULT_SEED = 42


@dataclass
class BenchArgs:
    """基准测试参数。

    Attributes:
        model_name (str): 模型名称。
        sequence_length (int): 序列长度。
        micro_batch_size (int): 微批大小。
        batch_accumulation_per_replica (int): 每个副本的梯度累积步数。
        benchmark_csv_path (str): 基准测试结果 CSV 文件路径。
    """
    model_name: str
    sequence_length: int
    micro_batch_size: int
    batch_accumulation_per_replica: int
    benchmark_csv_path: str


@dataclass
class LoggingArgs:
    """日志记录参数。

    Attributes:
        log_level (Optional[str]): 全局日志级别，可选值: debug/info/warning/error/critical/passive。
        log_level_replica (Optional[str]): 副本日志级别，仅 rank 0 使用 log_level，其他 rank 使用此级别。
        iteration_step_info_interval (Optional[int]): 迭代步信息打印间隔，默认为 1。
    """

    log_level: Optional[str] = None
    log_level_replica: Optional[str] = None
    iteration_step_info_interval: Optional[int] = 1

    def __post_init__(self):
        if self.log_level is None:
            self.log_level = "info"
        if self.log_level not in [
            "debug",
            "info",
            "warning",
            "error",
            "critical",
            "passive",
        ]:
            raise ValueError(
                f"log_level should be a string selected in ['debug', 'info', 'warning', 'error', 'critical', 'passive'] and not {self.log_level}"
            )
        if self.log_level_replica is None:
            self.log_level_replica = "info"
        if self.log_level_replica not in [
            "debug",
            "info",
            "warning",
            "error",
            "critical",
            "passive",
        ]:
            raise ValueError(
                f"log_level_replica should be a string selected in ['debug', 'info', 'warning', 'error', 'critical', 'passive'] and not {self.log_level_replica}"
            )


@dataclass
class MetricsLoggingArgs:
    """指标日志记录参数。

    Attributes:
        log_level (int): 指标日志级别，0=基础，1=完整。
        log_detail_interval (int): 详细指标记录间隔（步数）。
    """

    log_level: int = 0
    log_detail_interval: int = 10

    def __post_init__(self):
        if self.log_level not in [0, 1]:
            raise ValueError(f"metrics_level should be either 0 (basic) or 1 (full) and not {self.level}")
        if self.log_detail_interval <= 0:
            raise ValueError(f"metrics_interval should be a positive integer and not {self.interval}")


@dataclass
class PretrainDatasetsArgs:
    """预训练数据集参数（基于 HuggingFace datasets）。

    Attributes:
        hf_dataset_or_datasets (Union[str, list, dict]): HuggingFace 数据集名称或配置。
        hf_dataset_splits (Optional[Union[str, list]]): 数据集划分，默认为 "train"。
        hf_dataset_config_name (Optional[str]): HuggingFace 数据集配置名称。
        dataset_processing_num_proc_per_process (Optional[int]): 每个进程的数据处理工作数，默认为 1。
        dataset_overwrite_cache (Optional[bool]): 是否覆盖数据集缓存，默认为 False。
        text_column_name (Optional[str]): 文本列名，默认为 "text"。
    """
    hf_dataset_or_datasets: Union[str, list, dict]
    hf_dataset_splits: Optional[Union[str, list]] = None
    hf_dataset_config_name: Optional[str] = None
    dataset_processing_num_proc_per_process: Optional[int] = 1
    dataset_overwrite_cache: Optional[bool] = False
    text_column_name: Optional[str] = None

    def __post_init__(self):
        if self.text_column_name is None:
            self.text_column_name = "text"
        if self.hf_dataset_splits is None:
            self.hf_dataset_splits = "train"


@dataclass
class SFTDatasetsArgs:
    """监督微调（SFT）数据集参数。

    Attributes:
        hf_dataset_or_datasets (Union[str, list, dict]): HuggingFace 数据集名称或配置。
        hf_dataset_splits (Optional[Union[str, list]]): 数据集划分，默认为 "train"。
        hf_dataset_config_name (Optional[str]): HuggingFace 数据集配置名称。
        dataset_processing_num_proc_per_process (Optional[int]): 每个进程的数据处理工作数。
        dataset_overwrite_cache (Optional[bool]): 是否覆盖数据集缓存。
        sft_dataloader (Optional[bool]): 是否使用 SFT 数据加载器，默认为 True。
        debug_max_samples (Optional[int]): 调试模式下的最大样本数。
    """
    hf_dataset_or_datasets: Union[str, list, dict]
    hf_dataset_splits: Optional[Union[str, list]] = None
    hf_dataset_config_name: Optional[str] = None
    dataset_processing_num_proc_per_process: Optional[int] = 1
    dataset_overwrite_cache: Optional[bool] = False
    sft_dataloader: Optional[bool] = True
    debug_max_samples: Optional[int] = None

    def __post_init__(self):
        if self.hf_dataset_splits is None:
            self.hf_dataset_splits = "train"


@dataclass
class S3UploadArgs:
    """S3 检查点上传参数。

    Attributes:
        upload_s3_path (xPath): S3 上传路径。
        remove_after_upload (bool): 上传后是否删除本地文件。
        s5cmd_numworkers (Optional[int]): s5cmd 并发工作数。
        s5cmd_concurrency (Optional[int]): s5cmd 并发请求数。
        s5cmd_path (Optional[xPath]): s5cmd 可执行文件路径。
    """

    upload_s3_path: xPath
    remove_after_upload: bool
    s5cmd_numworkers: Optional[int]
    s5cmd_concurrency: Optional[int]
    s5cmd_path: Optional[xPath]

    def __post_init__(self):
        if isinstance(self.upload_s3_path, str):
            self.upload_s3_path = xPath(self.upload_s3_path)
        if isinstance(self.s5cmd_path, str):
            self.s5cmd_path = xPath(self.s5cmd_path)


@dataclass
class NanosetDatasetsArgs:
    """Nanoset 数据集参数（基于预分词的二进制数据格式）。

    Nanoset 使用 datatrove 预处理的二进制格式，支持高效的分布式数据加载。

    Attributes:
        dataset_folder (Union[str, List[str]]): 数据集文件夹路径。
        dataset_weights (Optional[List[float]]): 各数据集的采样权重。
        dataset_read_path (Optional[Union[str, List[str]]]): 本地读取路径（优先于 dataset_folder）。
        tokenizer_name (Optional[str]): 分词器名称，从元数据文件自动推断。
        vocab_size (Optional[int]): 词表大小，从元数据文件自动推断。
        token_size_in_bytes (Optional[int]): 每个 token 的字节大小，从元数据文件自动推断。
        return_positions (Optional[bool]): 是否返回位置信息，默认为 True。
        skip_in_stream (Optional[bool]): 是否在流式读取中跳过，默认为 False。
        pad_samples_to_global_batch_size (Optional[bool]): 是否填充样本到全局批大小，默认为 False。
        dataset_max_tokens (Optional[List[int]]): 各数据集的最大 token 数。
        shuffle_files (Optional[bool]): 是否打乱文件顺序，默认为 False。
        use_old_brrr_dataloader (Optional[bool]): 是否使用旧版数据加载器，默认为 False。
    """
    dataset_folder: Union[str, List[str]]
    dataset_weights: Optional[List[float]] = None
    dataset_read_path: Optional[
        Union[str, List[str]]
    ] = None  # Path to local file/copy to read from. If it exists, we read from this folder instead of from dataset_folder. Useful when we offload some data to remote and only keep the needed files on disk.
    # Tokenizer config, assuming all datasets use the same tokenizer
    tokenizer_name: Optional[str] = None
    vocab_size: Optional[int] = None
    token_size_in_bytes: Optional[int] = None
    return_positions: Optional[
        bool
    ] = True  # read positions stored in disk by datatrove if eos_token_id is None, else computed on the fly

    # Tokenized bytes dataset config
    skip_in_stream: Optional[bool] = False
    pad_samples_to_global_batch_size: Optional[bool] = False
    dataset_max_tokens: Optional[List[int]] = None
    shuffle_files: Optional[bool] = False
    use_old_brrr_dataloader: Optional[bool] = False

    def __post_init__(self):
        if isinstance(self.dataset_folder, str):  # Case 1: 1 Dataset folder
            self.dataset_folder = [self.dataset_folder]
            self.dataset_weights = [1]

        # Check if dataset_weights is provided and matches the number of dataset folders
        if self.dataset_weights is not None and len(self.dataset_weights) != len(self.dataset_folder):
            raise ValueError(
                f"Number of dataset weights ({len(self.dataset_weights)}) does not match number of dataset folders ({len(self.dataset_folder)})"
            )

        # Read the first metadata file in the dataset folder to extract tokenizer name and token size.
        for folder in self.dataset_folder:
            # Find all metadata files in the folder
            metadata_files = glob.glob(os.path.join(folder, "*.metadata"))
            if metadata_files:
                # Read the first line of the first metadata file
                with open(metadata_files[0], "r") as f:
                    first_line = f.readline().strip()
                    if "|" in first_line:
                        tokenizer_name, token_size_in_bytes = first_line.split("|")
                        if self.tokenizer_name is None:
                            self.tokenizer_name = tokenizer_name
                            self.token_size_in_bytes = int(token_size_in_bytes)
                            self.vocab_size = len(AutoTokenizer.from_pretrained(tokenizer_name).get_vocab())
                        else:
                            assert (
                                self.tokenizer_name == tokenizer_name
                            ), f"Tokenizer name mismatch while reading datasets metadata file, found both {self.tokenizer_name} and {tokenizer_name}"
                            assert self.token_size_in_bytes == int(
                                token_size_in_bytes
                            ), f"Token size mismatch while reading datasets metadata file, found both {self.token_size_in_bytes} and {token_size_in_bytes}"

        # Check if dataset_read_path is provided and matches the number of dataset folders
        if self.dataset_read_path is not None and len(self.dataset_read_path) != len(self.dataset_folder):
            raise ValueError(
                f"Number of dataset read paths ({len(self.dataset_read_path)}) does not match number of dataset folders ({len(self.dataset_folder)})"
            )


@dataclass
class DataArgs:
    """数据加载参数。

    Attributes:
        dataset (Optional[Union[PretrainDatasetsArgs, NanosetDatasetsArgs, SFTDatasetsArgs]]):
            数据集配置，None 时使用虚拟无限数据生成器。
        seed (Optional[int]): 数据加载随机种子，默认为 DEFAULT_SEED。
        num_loading_workers (Optional[int]): 数据加载工作进程数，默认为 1。
    """

    dataset: Optional[
        Union[PretrainDatasetsArgs, NanosetDatasetsArgs, SFTDatasetsArgs]
    ]  # If None we use dummy_infinite_data_generator
    seed: Optional[int]
    num_loading_workers: Optional[int] = 1

    def __post_init__(self):
        if self.seed is None:
            self.seed = DEFAULT_SEED


@dataclass
class DatasetStageArgs:
    """训练阶段数据集参数，支持训练过程中切换数据集。

    Attributes:
        name (str): 阶段名称（必须唯一）。
        start_training_step (int): 阶段开始的训练步数（第一个阶段必须为 1）。
        data (DataArgs): 该阶段的数据配置。
        sequence_length (Optional[int]): 该阶段的序列长度，None 时使用全局配置。
    """

    name: str
    start_training_step: int
    data: DataArgs
    sequence_length: Optional[int] = None # if None, we use the sequence length from the config

    def __post_init__(self):
        if self.start_training_step < 0:
            raise ValueError(f"training_steps should be a positive integer and not {self.start_training_step}")


@dataclass
class CheckpointsArgs:
    """检查点保存和恢复参数。

    Attributes:
        checkpoints_path (Path): 检查点保存路径。
        checkpoint_interval (int): 检查点保存间隔（步数）。
        save_initial_state (Optional[bool]): 是否保存初始状态，默认为 False。
        save_final_state (Optional[bool]): 是否保存最终状态，默认为 True。
        resume_checkpoint_path (Optional[xPath]): 恢复训练的检查点路径。
        load_lr_scheduler (Optional[bool]): 恢复时是否加载学习率调度器状态，默认为 True。
        load_optimizer (Optional[bool]): 恢复时是否加载优化器状态，默认为 True。
        checkpoints_path_is_shared_file_system (Optional[bool]): 检查点路径是否为共享文件系统，默认为 False。
    """

    checkpoints_path: Path
    checkpoint_interval: int
    save_initial_state: Optional[bool] = False
    save_final_state: Optional[bool] = True
    resume_checkpoint_path: Optional[xPath] = None
    load_lr_scheduler: Optional[bool] = True
    load_optimizer: Optional[bool] = True
    checkpoints_path_is_shared_file_system: Optional[bool] = False

    def __post_init__(self):
        if isinstance(self.checkpoints_path, str):
            self.checkpoints_path = xPath(self.checkpoints_path)
        if isinstance(self.resume_checkpoint_path, str):
            self.resume_checkpoint_path = xPath(self.resume_checkpoint_path)


@dataclass
class GeneralArgs:
    """通用训练实验参数。

    Attributes:
        project (str): 项目名称（同一项目的多次运行共享 tensorboard/hub 目录）。
        run (Optional[str]): 运行名称，支持 %date 和 %jobid 占位符。
        seed (Optional[int]): 全局随机种子，默认为 DEFAULT_SEED。
        step (Optional[int]): 当前全局步数（从检查点恢复时更新）。
        consumed_train_samples (Optional[int]): 已消耗的训练样本数。
        benchmark_csv_path (Optional[Path]): 基准测试结果路径。
        ignore_sanity_checks (bool): 是否跳过健全性检查，默认为 True。
    """

    project: str
    run: Optional[str] = None
    seed: Optional[int] = None
    step: Optional[int] = None
    consumed_train_samples: Optional[int] = None # TODO: remove this
    benchmark_csv_path: Optional[Path] = None
    ignore_sanity_checks: bool = True

    def __post_init__(self):
        if self.seed is None:
            self.seed = DEFAULT_SEED
        if self.run is None:
            self.run = "%date_%jobid"
        self.run = self.run.replace("%date", datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
        self.run = self.run.replace("%jobid", os.environ.get("SLURM_JOB_ID", "local"))


@dataclass
class ProfilerArgs:
    """性能分析参数。

    Attributes:
        profiler_export_path (Optional[Path]): 分析结果导出路径。
        wait (int): 等待步数，默认为 1。
        warmup (int): 预热步数，默认为 1。
        active (int): 活跃记录步数，默认为 1。
        repeat (int): 重复次数，默认为 1。
        skip_first (int): 跳过前 N 步，默认为 3。
        record_shapes (bool): 是否记录张量形状，默认为 False。
        profile_memory (bool): 是否分析内存，默认为 False。
        with_stack (bool): 是否记录调用栈，默认为 True。
        export_chrome_trace (bool): 是否导出 Chrome trace 格式，默认为 False。
    """

    profiler_export_path: Optional[Path]  # e.g. ./tb_logs
    wait: int = 1
    warmup: int = 1
    active: int = 1
    repeat: int = 1
    skip_first: int = 3
    record_shapes: bool = False
    profile_memory: bool = False
    with_stack: bool = True
    export_chrome_trace: bool = False

@dataclass
class ModelArgs:
    """模型架构和初始化参数。

    Attributes:
        model_config (NanotronConfigs): 模型配置（如 LlamaConfig、Qwen2Config）。
        init_method (Union[RandomInit, SpectralMupInit, ExistingCheckpointInit]): 参数初始化方法。
        dtype (Optional[torch.dtype]): 模型计算精度，默认为 bfloat16。
        make_vocab_size_divisible_by (int): 使词表大小可被该值整除，默认为 1。
        ddp_bucket_cap_mb (int): DDP 桶大小（MB），默认为 25。
    """

    model_config: NanotronConfigs
    init_method: Union[RandomInit, SpectralMupInit, ExistingCheckpointInit]
    dtype: Optional[torch.dtype] = None
    make_vocab_size_divisible_by: int = 1
    ddp_bucket_cap_mb: int = 25

    def __post_init__(self):
        if self.dtype is None:
            self.dtype = torch.bfloat16
        if isinstance(self.dtype, str):
            self.dtype = cast_str_to_torch_dtype(self.dtype)

        if isinstance(self.model_config, dict):
            self.model_config = Qwen2Config(**self.model_config)

        self.model_config._is_using_mup = isinstance(self.init_method, SpectralMupInit)

        # if self.model_config.max_position_embeddings is None:
        #     self.model_config.max_position_embeddings = 0


@dataclass
class TokenizerArgs:
    """分词器参数。

    Attributes:
        tokenizer_name_or_path (Optional[str]): 分词器名称或路径。
        tokenizer_revision (Optional[str]): 分词器版本。
        tokenizer_max_length (Optional[int]): 分词器最大长度。
    """

    tokenizer_name_or_path: Optional[str] = None
    tokenizer_revision: Optional[str] = None
    tokenizer_max_length: Optional[int] = None


@dataclass
class TokensArgs:
    """Token、序列、批大小和训练步数参数。

    全局批大小 = micro_batch_size × batch_accumulation_per_replica × dp

    Attributes:
        sequence_length (int): 序列长度。
        train_steps (int): 总训练步数。
        micro_batch_size (int): 微批大小（每个 GPU 上的批大小）。
        batch_accumulation_per_replica (int): 每个副本的梯度累积步数。
        val_check_interval (Optional[int]): 验证检查间隔，-1 表示不验证。
        limit_val_batches (Optional[int]): 验证批次数限制，0 表示不验证。
        limit_test_batches (Optional[int]): 测试批次数限制，0 表示不测试。
    """

    sequence_length: int
    train_steps: int
    micro_batch_size: int
    batch_accumulation_per_replica: int

    val_check_interval: Optional[int] = -1
    limit_val_batches: Optional[int] = 0
    limit_test_batches: Optional[int] = 0


@dataclass
class LRSchedulerArgs:
    """学习率调度器参数。

    学习率变化过程：warmup → 稳定 → decay

    Attributes:
        learning_rate (float): 峰值学习率。
        lr_warmup_steps (int): 预热步数，默认为 0。
        lr_warmup_style (str): 预热方式，"linear" 或 "constant"。
        lr_decay_style (str): 衰减方式，"linear"、"cosine" 或 "1-sqrt"。
        lr_decay_steps (Optional[int]): 衰减步数，默认为 train_steps - warmup_steps。
        lr_decay_starting_step (Optional[int]): 衰减开始步数，默认为 warmup_steps。
        min_decay_lr (float): 衰减后的最小学习率，默认为 learning_rate。
    """

    learning_rate: float
    lr_warmup_steps: int = 0
    lr_warmup_style: str = None
    lr_decay_style: str = None
    lr_decay_steps: Optional[int] = None
    lr_decay_starting_step: Optional[int] = None
    min_decay_lr: float = None

    def __post_init__(self):
        if self.lr_warmup_style not in ["linear", "constant"]:
            raise ValueError(
                f"lr_warmup_style should be a string selected in ['linear', 'constant'] and not {self.lr_warmup_style}"
            )
        if self.lr_warmup_style is None:
            self.lr_warmup_style = "linear"
        if self.lr_decay_style is None:
            self.lr_decay_style = "linear"
        if self.lr_decay_style not in ["linear", "cosine", "1-sqrt"]:
            raise ValueError(
                f"lr_decay_style should be a string selected in ['linear', 'cosine', '1-sqrt'] and not {self.lr_decay_style}"
            )
        if self.min_decay_lr is None:
            self.min_decay_lr = self.learning_rate


@dataclass
class SGDOptimizerArgs:
    """SGD 优化器参数。

    Attributes:
        name (str): 优化器名称，默认为 "sgd"。
    """
    name: str = "sgd"


@dataclass
class AdamWOptimizerArgs:
    """AdamW 优化器参数。

    Attributes:
        adam_eps (float): Adam 的 epsilon 值，防止除零。
        adam_beta1 (float): Adam 的一阶矩衰减系数。
        adam_beta2 (float): Adam 的二阶矩衰减系数。
        torch_adam_is_fused (bool): 是否使用 PyTorch 的融合 Adam 实现。
        name (str): 优化器名称，默认为 "adamW"。
    """
    adam_eps: float
    adam_beta1: float
    adam_beta2: float
    torch_adam_is_fused: bool
    name: str = "adamW"


@dataclass
class OptimizerArgs:
    """优化器和学习率参数。

    Attributes:
        optimizer_factory (Union[SGDOptimizerArgs, AdamWOptimizerArgs]): 优化器工厂配置。
        zero_stage (int): ZeRO 优化阶段（0/1）。
        weight_decay (float): 权重衰减系数。
        clip_grad (Optional[float]): 梯度裁剪阈值。
        accumulate_grad_in_fp32 (bool): 是否在 FP32 精度下累积梯度。
        learning_rate_scheduler (LRSchedulerArgs): 学习率调度器配置。
        weight_decay_exclude_named_params (Optional[List[str]]): 排除权重衰减的参数名正则模式列表。
    """

    optimizer_factory: Union[SGDOptimizerArgs, AdamWOptimizerArgs]
    zero_stage: int
    weight_decay: float
    clip_grad: Optional[float]
    accumulate_grad_in_fp32: bool
    learning_rate_scheduler: LRSchedulerArgs
    weight_decay_exclude_named_params: Optional[
        List[str]
    ] = None  # List of regex patterns to exclude parameters from weight decay

    def __post_init__(self):
        if self.weight_decay_exclude_named_params is None:
            self.weight_decay_exclude_named_params: List[str] = []


@dataclass
class GenerationArgs:
    """文本生成参数。

    Attributes:
        sampler (Optional[Union[str, SamplerType]]): 采样策略（greedy/top_k/top_p/multinomial）。
        temperature (Optional[float]): 采样温度。
        top_k (Optional[int]): Top-K 采样的 K 值。
        top_p (Optional[float]): Top-P（核）采样的 P 值。
        n_samples (Optional[int]): 生成样本数。
        eos (Optional[str]): 结束标记。
        seed (Optional[int]): 生成随机种子，默认为 DEFAULT_SEED。
        use_cache (Optional[bool]): 是否使用 KV 缓存，默认为 False。
    """
    sampler: Optional[Union[str, SamplerType]] = None
    temperature: Optional[float] = None
    top_k: Optional[int] = None
    top_p: Optional[float] = None
    n_samples: Optional[int] = None
    eos: Optional[str] = None
    seed: Optional[int] = None
    use_cache: Optional[bool] = False

    def __post_init__(self):
        if isinstance(self.sampler, str):
            self.sampler = SamplerType[self.sampler.upper()]
        if self.seed is None:
            self.seed = DEFAULT_SEED


@dataclass
class Config:
    """Nanotron 顶层配置类，组合所有子配置。

    该类是整个训练配置的入口点，包含模型、并行、优化器、数据等所有配置。
    支持从 YAML 文件加载和保存，并进行跨字段的健全性检查。

    Attributes:
        general (GeneralArgs): 通用训练参数。
        parallelism (ParallelismArgs): 并行配置。
        model (ModelArgs): 模型配置。
        tokenizer (Optional[TokenizerArgs]): 分词器配置。
        checkpoints (Optional[CheckpointsArgs]): 检查点配置。
        logging (Optional[LoggingArgs]): 日志配置。
        metrics_logging (Optional[MetricsLoggingArgs]): 指标日志配置。
        tokens (Optional[TokensArgs]): Token 和训练步数配置。
        optimizer (Optional[OptimizerArgs]): 优化器配置。
        data_stages (Optional[List[DatasetStageArgs]]): 训练阶段数据配置。
        profiler (Optional[ProfilerArgs]): 性能分析配置。
        lighteval (Optional[LightEvalConfig]): LightEval 评估配置。
        s3_upload (Optional[S3UploadArgs]): S3 上传配置。
    """

    general: GeneralArgs
    parallelism: ParallelismArgs
    model: ModelArgs
    tokenizer: Optional[TokenizerArgs] = None
    checkpoints: Optional[CheckpointsArgs] = None
    logging: Optional[LoggingArgs] = None
    metrics_logging: Optional[MetricsLoggingArgs] = None
    tokens: Optional[TokensArgs] = None
    optimizer: Optional[OptimizerArgs] = None
    data_stages: Optional[List[DatasetStageArgs]] = None
    profiler: Optional[ProfilerArgs] = None
    lighteval: Optional[LightEvalConfig] = None
    s3_upload: Optional[S3UploadArgs] = None

    @classmethod
    def create_empty(cls):
        """创建一个所有字段为 None 的空配置对象。

        Returns:
            Config: 空配置对象。
        """
        cls_fields = fields(cls)
        return cls(**{f.name: None for f in cls_fields})

    def __post_init__(self):
        """配置初始化后的跨字段验证和默认值设置。

        执行以下检查：
            - S3 上传和 LightEval 配置的一致性
            - Profiler 步数不超过训练步数
            - 学习率衰减步数默认值
            - 数据阶段的排序和唯一性
            - 模型并行与注意力头的兼容性
        """

        if self.s3_upload is not None:
            self.s3_upload.__post_init__()
            if self.lighteval is not None:
                if self.lighteval.eval_interval is None:
                    self.lighteval.eval_interval = self.checkpoints.checkpoint_interval
                else:
                    assert (
                        self.lighteval.eval_interval % self.checkpoints.checkpoint_interval == 0
                    ), f"eval_interval={self.lighteval.eval_interval} must be a multiple of checkpoint_interval={self.checkpoints.checkpoint_interval}"

        # Some final sanity checks across separate arguments sections:
        if self.profiler is not None and self.profiler.profiler_export_path is not None:
            total_profiling_steps = self.profiler.skip_first + self.profiler.repeat * (
                self.profiler.wait + self.profiler.warmup + self.profiler.active
            )
            assert (
                self.tokens.train_steps >= total_profiling_steps
            ), f"Profiling steps ({total_profiling_steps}) must be less than or equal to train steps ({self.tokens.train_steps})"

        if self.optimizer is not None and self.optimizer.learning_rate_scheduler.lr_decay_steps is None:
            self.optimizer.learning_rate_scheduler.lr_decay_steps = (
                self.tokens.train_steps - self.optimizer.learning_rate_scheduler.lr_warmup_steps
            )

        if self.data_stages is not None:
            self.data_stages = sorted(self.data_stages, key=lambda stage: stage.start_training_step)
            names = [stage.name for stage in self.data_stages]
            training_steps = [stage.start_training_step for stage in self.data_stages]
            assert any(
                stage.start_training_step == 1 for stage in self.data_stages
            ), "You must have a training stage starting at 1 in the config's data_stages"

            for stage in self.data_stages:
                if names.count(stage.name) > 1:
                    raise ValueError(f"Each stage should have unique names and not {names}")

                if training_steps.count(stage.start_training_step) > 1:
                    raise ValueError(
                        f"Each stage should have unique starting training step, please change the starting training step for stage {stage.name}"
                    )

                if isinstance(stage.data.dataset, NanosetDatasetsArgs):
                    if self.model.model_config.vocab_size == -1:
                        self.model.model_config.vocab_size = stage.data.dataset.vocab_size
                        logger.warning(
                            f"Setting model's vocab_size to {self.model.model_config.vocab_size} from dataset's vocab_size ({stage.data.dataset.vocab_size})"
                        )
                    assert (
                        self.model.model_config.vocab_size == stage.data.dataset.vocab_size
                    ), f"Model's vocab_size ({self.model.model_config.vocab_size}) does not match dataset's ({stage.data.dataset.dataset_folder}) vocab_size ({stage.data.dataset.vocab_size})"
                    if self.tokenizer is None:
                        self.tokenizer = TokenizerArgs(tokenizer_name_or_path=stage.data.dataset.tokenizer_name)
                        logger.warning(
                            f"Setting tokenizer to {self.tokenizer.tokenizer_name_or_path} from dataset's tokenizer ({stage.data.dataset.tokenizer_name})"
                        )
                    assert (
                        self.tokenizer.tokenizer_name_or_path == stage.data.dataset.tokenizer_name
                    ), f"Tokenizer passed in config ({self.tokenizer.tokenizer_name_or_path}) does not match dataset's ({stage.data.dataset.dataset_folder}) tokenizer ({stage.data.dataset.tokenizer_name})"

            # NOTE: must order the stages by start_training_step from lowest to highest
            assert all(
                self.data_stages[i].start_training_step < self.data_stages[i + 1].start_training_step
                for i in range(len(self.data_stages) - 1)
            ), "The stages are not sorted by start_training_step in increasing order"

        # # if lighteval, we need tokenizer to be defined
        # if self.checkpoints.lighteval is not None:
        #     assert self.tokenizer.tokenizer_name_or_path is not None

        # Model verifications
        assert (
            self.model.model_config.num_attention_heads % self.parallelism.tp == 0
        ), f"num_attention_heads ({self.model.model_config.num_attention_heads}) must be divisible by tp ({self.parallelism.tp})"
        assert (
            self.model.model_config.num_attention_heads >= self.model.model_config.num_key_value_heads
        ), f"num_attention_heads ({self.model.model_config.num_attention_heads}) must be >= num_key_value_heads ({self.model.model_config.num_key_value_heads})"
        assert (
            self.model.model_config.num_key_value_heads >= self.parallelism.tp
        ), f"num_key_value_heads ({self.model.model_config.num_key_value_heads}) must be >= tp ({self.parallelism.tp})"  # TODO: remove this once we ensure KV heads get duplicated correctly
        assert (
            self.model.model_config.num_attention_heads % self.model.model_config.num_key_value_heads == 0
        ), f"num_attention_heads ({self.model.model_config.num_attention_heads}) must be divisible by num_key_value_heads ({self.model.model_config.num_key_value_heads})"

        # data_stages
        if self.data_stages is not None:
            for stage in self.data_stages:
                if stage.sequence_length is None:
                    stage.sequence_length = self.tokens.sequence_length

    @property
    def global_batch_size(self):
        """计算全局批大小。

        global_batch_size = micro_batch_size × batch_accumulation_per_replica × dp

        Returns:
            int: 全局批大小。
        """
        return self.tokens.micro_batch_size * self.tokens.batch_accumulation_per_replica * self.parallelism.dp

    @property
    def global_batch_size_in_tokens(self):
        """计算全局批大小的 token 数。

        Returns:
            int: 全局批大小（以 token 为单位）。
        """
        return self.global_batch_size * self.tokens.sequence_length

    def save_as_yaml(self, file_path: str, sanity_checks: bool = True):
        """将配置保存为 YAML 文件。

        Args:
            file_path (str): 保存路径。
            sanity_checks (bool): 是否验证保存后可以重新加载，默认为 True。
        """
        config_dict = serialize(self)
        file_path = str(file_path)
        with open(file_path, "w") as f:
            yaml.dump(config_dict, f)

        # Sanity test config can be reloaded
        if sanity_checks:
            _ = get_config_from_file(file_path, config_class=self.__class__)

    def get_yaml(self):
        """获取配置的 YAML 字符串表示。

        Returns:
            str: YAML 格式的配置字符串。
        """
        config_dict = serialize(self)
        return yaml.dump(config_dict)

    @classmethod
    def load_from_yaml(cls, file_path: str):
        """从 YAML 文件加载配置。

        Args:
            file_path (str): YAML 文件路径。

        Returns:
            Config: 加载的配置对象。
        """
        config_dict = yaml.load(open(file_path), Loader=SafeLoader)
        return get_config_from_dict(config_dict, config_class=cls)

    def as_dict(self) -> dict:
        """将配置转换为字典。

        Returns:
            dict: 配置字典。
        """
        return serialize(self)

    def print_config_details(self):
        """打印配置的详细信息，包括模型架构、训练配置和并行设置。"""
        print("\n=== Model Architecture ===")
        print(f"hidden_size: {self.model.model_config.hidden_size}")
        print(f"num_layers: {self.model.model_config.num_hidden_layers}")
        print(f"intermediate_size: {self.model.model_config.intermediate_size}")
        print(f"num_attention_heads: {self.model.model_config.num_attention_heads}")
        print(f"num_key_value_heads: {self.model.model_config.num_key_value_heads}")
        print(f"tie_word_embeddings: {self.model.model_config.tie_word_embeddings}")
        print(f"vocab_size: {self.model.model_config.vocab_size}")
        print(f"num_params: {_calculate_model_params(self)}")

        print("\n=== Training Configuration ===")
        print(
            f"seq_len: {self.model.model_config.max_position_embeddings} | mbs: {self.tokens.micro_batch_size} | batch_accum: {self.tokens.batch_accumulation_per_replica} | gbs: {human_format(self.global_batch_size_in_tokens)} | train_steps: {self.tokens.train_steps} | total_tokens: {human_format(self.tokens.train_steps * self.global_batch_size_in_tokens)}"
        )

        print("\n=== Parallelism ===")
        print(
            f"tp: {self.parallelism.tp} | pp: {self.parallelism.pp} | dp: {self.parallelism.dp} | cp: {self.parallelism.context_parallel_size} | ep: {self.parallelism.expert_parallel_size}"
        )
        print(f"zero_stage: {self.optimizer.zero_stage} | full_checkpointing: {self.parallelism.recompute_layer}")
        print("=" * 20 + "\n")



def get_config_from_dict(
    config_dict: dict, config_class: Type = Config, skip_unused_config_keys: bool = False, skip_null_keys: bool = False
):
    """从字典创建配置对象。

    使用 dacite 库进行类型安全的反序列化，支持自动类型转换
    （如字符串到 torch.dtype、枚举类型等）。

    Args:
        config_dict (dict): 配置字典。
        config_class (Type): 配置类类型，默认为 Config。
        skip_unused_config_keys (bool): 是否跳过未使用的顶层键，默认为 False。
        skip_null_keys (bool): 是否跳过值为 None 的键，默认为 False。

    Returns:
        Config: 配置对象。
    """
    if skip_unused_config_keys:
        logger.warning("skip_unused_config_keys set")
        config_dict = {
            field.name: config_dict[field.name] for field in fields(config_class) if field.name in config_dict
        }
    if skip_null_keys:
        logger.warning("Skip_null_keys set")
        config_dict = {
            k: {kk: vv for kk, vv in v.items() if vv is not None} if isinstance(v, dict) else v
            for k, v in config_dict.items()
            if v is not None
        }
    return from_dict(
        data_class=config_class,
        data=config_dict,
        config=dacite.Config(
            cast=[Path],
            type_hooks={
                torch.dtype: cast_str_to_torch_dtype,
                PipelineEngine: cast_str_to_pipeline_engine,
                TensorParallelLinearMode: lambda x: TensorParallelLinearMode[x.upper()],
                RecomputeGranularity: lambda x: RecomputeGranularity[x.upper()],
                InitScalingMethod: lambda x: InitScalingMethod[x.upper()],
                SamplerType: lambda x: SamplerType[x.upper()],
            },
            # strict_unions_match=True,
            strict=True,
        ),
    )

# xx 配置加载
def get_config_from_file(
    config_path: str,
    config_class: Type = Config,
    model_config_class: Optional[Type] = None,
    skip_unused_config_keys: bool = False,
    skip_null_keys: bool = False,
) -> Config:
    """从 YAML 文件加载配置对象。

    Args:
        config_path (str): YAML 配置文件路径。
        config_class (Type): 配置类类型，默认为 Config。
        model_config_class (Optional[Type]): 模型配置类，用于覆盖默认的模型配置类型。
        skip_unused_config_keys (bool): 是否跳过未使用的顶层键。
        skip_null_keys (bool): 是否跳过值为 None 的键。

    Returns:
        Config: 加载的配置对象。
    """
    # Open the file and load the file
    with open(config_path) as f:
        config_dict = yaml.load(f, Loader=SafeLoader)

    config = get_config_from_dict(
        config_dict,
        config_class=config_class,
        skip_unused_config_keys=skip_unused_config_keys,
        skip_null_keys=skip_null_keys,
    )
    if model_config_class is not None:
        if not isinstance(config.model.model_config, (dict, model_config_class)):
            raise ValueError(
                f"model_config should be a dictionary or a {model_config_class} and not {config.model.model_config}"
            )
        config.model.model_config = model_config_class(**config.model.model_config)
    return config


def _calculate_model_params(config: Config):
    """估算模型的参数数量。

    使用简化公式估算 LLaMA 类模型的参数量：
    N = vocab × h × (1 + tie_word_embeddings) + num_layers × (3 × h × inter + 4 × h²)

    其中 h = hidden_size, inter = intermediate_size

    Args:
        config (Config): 配置对象。

    Returns:
        str: 人类可读格式的参数数量（如 "7B"、"70B"）。
    """
    num_params = human_format(
        config.model.model_config.vocab_size
        * config.model.model_config.hidden_size
        * (2 if config.model.model_config.tie_word_embeddings else 1)
        + config.model.model_config.num_hidden_layers
        * (
            3 * config.model.model_config.hidden_size * config.model.model_config.intermediate_size
            + 4 * config.model.model_config.hidden_size * config.model.model_config.hidden_size
        )
    )
    return num_params
