import typing as tp
from dataclasses import dataclass

from xlstm_jax.components.util import ShardingRule


@dataclass(slots=True)
class FeedForwardShardingConfig:
    up_proj: ShardingRule
    down_proj: ShardingRule
    bias: ShardingRule

    def __post_init__(self):
        self.up_proj = tuple(self.up_proj)
        self.down_proj = tuple(self.down_proj)
        self.bias = tuple(self.bias)

    @staticmethod
    def get_default_sharding():
        shd = FeedForwardShardingConfig(
            up_proj=(None, "tp"),
            down_proj=(None, "tp"),
            bias=("tp",),
        )

        shd.__post_init__()
        return shd


@dataclass(slots=True)
class CausalConv1dShardingConfig:
    kernel: ShardingRule
    bias: ShardingRule

    def __post_init__(self):
        self.kernel = tuple(self.kernel)
        self.bias = tuple(self.bias)

    @staticmethod
    def get_default_sharding():
        shd = CausalConv1dShardingConfig(
            kernel=(None, None, "tp"),
            bias=("tp",),
        )

        shd.__post_init__()
        return shd


@dataclass(slots=True)
class mLSTMCellShardingConfig:
    gate_kernel: ShardingRule
    gate_bias: ShardingRule
    norm: ShardingRule

    def __post_init__(self):
        self.gate_kernel = tuple(self.gate_kernel)
        self.gate_bias = tuple(self.gate_bias)
        self.norm = tuple(self.norm)

    @staticmethod
    def get_default_sharding():
        shd = mLSTMCellShardingConfig(
            gate_kernel=(None, "tp"),
            gate_bias=("tp",),
            norm=("tp",),
        )

        shd.__post_init__()
        return shd


@dataclass(slots=True)
class mLSTMLayerShardingConfig:
    causal_conv: CausalConv1dShardingConfig
    mlstm_cell: mLSTMCellShardingConfig
    bias: ShardingRule
    up_proj: ShardingRule
    qkv_proj: ShardingRule
    learnable_skip: ShardingRule
    down_proj: ShardingRule

    def __post_init__(self):
        self.causal_conv.__post_init__()
        self.mlstm_cell.__post_init__()

        self.bias = tuple(self.bias)
        self.up_proj = tuple(self.up_proj)
        self.qkv_proj = tuple(self.qkv_proj)
        self.learnable_skip = tuple(self.learnable_skip)
        self.down_proj = tuple(self.down_proj)

    @staticmethod
    def get_default_sharding():
        shd = mLSTMLayerShardingConfig(
            causal_conv=CausalConv1dShardingConfig.get_default_sharding(),
            mlstm_cell=mLSTMCellShardingConfig.get_default_sharding(),
            bias=("tp",),
            up_proj=(None, "tp"),
            qkv_proj=(None, None, "tp"),
            learnable_skip=("tp",),
            down_proj=("tp", None),
        )

        shd.__post_init__()
        return shd


@dataclass(slots=True)
class sLSTMCellShardingConfig:
    recurrent_kernel: ShardingRule
    recurrent_bias: ShardingRule

    def __post_init__(self):
        self.recurrent_bias = tuple(self.recurrent_bias)
        self.recurrent_kernel = tuple(self.recurrent_kernel)

    @staticmethod
    def get_default_sharding():
        shd = sLSTMCellShardingConfig(
            recurrent_kernel=(None, None, None, "tp"),
            recurrent_bias=(None, None, "tp"),
        )

        shd.__post_init__()
        return shd


@dataclass(slots=True)
class sLSTMLayerShardingConfig:
    slstm_cell: sLSTMCellShardingConfig
    causal_conv: CausalConv1dShardingConfig
    gate_kernel: ShardingRule
    norm: ShardingRule

    def __post_init__(self):
        self.slstm_cell.__post_init__()
        self.causal_conv.__post_init__()
        self.gate_kernel = tuple(self.gate_kernel)
        self.norm = tuple(self.norm)

    @staticmethod
    def get_default_sharding():
        shd = sLSTMLayerShardingConfig(
            slstm_cell=sLSTMCellShardingConfig.get_default_sharding(),
            causal_conv=CausalConv1dShardingConfig.get_default_sharding(),
            gate_kernel=(None, None, "tp"),
            norm=("tp",),
        )

        shd.__post_init__()
        return shd


@dataclass(slots=True)
class xLSTMBlockShardingConfig:
    xlstm_norm: ShardingRule
    xlstm: tp.Optional[tp.Union[mLSTMLayerShardingConfig, sLSTMLayerShardingConfig]] = (
        None
    )
    ffn: tp.Optional[FeedForwardShardingConfig] = None
    ffn_norm: tp.Optional[ShardingRule] = None

    def __post_init__(self):
        if self.xlstm is not None:
            self.xlstm.__post_init__()

        if self.ffn is not None:
            self.ffn.__post_init__()

        self.ffn_norm = tuple(self.ffn_norm)
        self.xlstm_norm = tuple(self.xlstm_norm)

    @staticmethod
    def get_default_sharding():
        shd = xLSTMBlockShardingConfig(
            xlstm=None,
            xlstm_norm=("tp",),
            ffn=FeedForwardShardingConfig.get_default_sharding(),
            ffn_norm=("tp",),
        )

        shd.__post_init__()
        return shd


@dataclass(slots=True)
class mLSTMBlockShardingConfig(xLSTMBlockShardingConfig):
    @staticmethod
    def get_default_sharding():
        shd = mLSTMBlockShardingConfig(
            xlstm=mLSTMLayerShardingConfig.get_default_sharding(),
            ffn=FeedForwardShardingConfig.get_default_sharding(),
            ffn_norm=("tp",),
            xlstm_norm=("tp",),
        )

        shd.__post_init__()
        return shd


class sLSTMBlockShardingConfig(xLSTMBlockShardingConfig):
    @staticmethod
    def get_default_sharding():
        shd = mLSTMBlockShardingConfig(
            xlstm=sLSTMLayerShardingConfig.get_default_sharding(),
            ffn=FeedForwardShardingConfig.get_default_sharding(),
            ffn_norm=("tp",),
            xlstm_norm=("tp",),
        )

        shd.__post_init__()
        return shd


@dataclass(slots=True)
class xLSTMBlockStackShardingConfig:
    mlstm: tp.Optional[mLSTMBlockShardingConfig] = None
    slstm: tp.Optional[sLSTMBlockShardingConfig] = None
    post_blocks_norm: tp.Optional[ShardingRule] = None

    def __post_init__(self):
        if self.mlstm is not None:
            self.mlstm.__post_init__()

        if self.slstm is not None:
            self.slstm.__post_init__()

        self.post_blocks_norm = tuple(self.post_blocks_norm)

    @staticmethod
    def get_default_sharding():
        shd = xLSTMBlockStackShardingConfig(
            mlstm=mLSTMBlockShardingConfig.get_default_sharding(),
            slstm=sLSTMBlockShardingConfig.get_default_sharding(),
            post_blocks_norm=("tp",),
        )

        shd.__post_init__()
        return shd


@dataclass(slots=True)
class xLSTMLMModelShardingConfig:
    blocks: xLSTMBlockStackShardingConfig
    embedding: ShardingRule
    lm_head: ShardingRule

    def __post_init__(self):
        self.lm_head = tuple(self.lm_head)
        self.embedding = tuple(self.embedding)

        self.blocks.__post_init__()

    @staticmethod
    def get_default_sharding():
        shd = xLSTMLMModelShardingConfig(
            blocks=xLSTMBlockStackShardingConfig.get_default_sharding(),
            embedding=(None, "tp"),
            lm_head=(None, "tp"),
        )

        shd.__post_init__()
        return shd
