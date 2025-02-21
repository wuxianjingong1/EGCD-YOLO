# Ultralytics YOLO 🚀, AGPL-3.0 license
"""
Ultralytics modules.

Example:
    Visualize a module with Netron.
    ```python
    from ultralytics.nn.modules import *
    import torchcd
    import os

    x = torch.ones(1, 128, 40, 40)
    m = Conv(128, 128)
    f = f"{m._get_name()}.onnx"
    torch.onnx.export(m, x, f)
    os.system(f"onnxslim {f} {f} && open {f}")  # pip install onnxslim
    ```
"""
from .SimpleSAFM import SAFMNPP
from .fastnet import C2f_FasterBlock
from .MSBlock import C2f_MSBlock
from .CSPPC import CSPPC
from .conv import SimAM
from .attention import *
from .content_block import *
from .OREPA import OREPA
from .dynamicconv import DynamicConv
from .fastnet import BasicStage,PatchEmbed_FasterNet,PatchMerging_FasterNet
from .DWR import C2f_DWRSeg
from .block import (
    MV2Block,
    MobileViTBlock,
    C1,
    C2,
    C2PSA,
    C3,
    C3TR,
    CIB,
    DFL,
    ELAN1,
    PSA,
    SPP,
    SPPELAN,
    SPPF,
    AConv,
    ADown,
    Attention,
    BNContrastiveHead,
    Bottleneck,
    BottleneckCSP,
    C2f,
    C2fAttn,
    C2fCIB,
    C2fPSA,
    C3Ghost,
    C3k2,
    C3x,
    CBFuse,
    CBLinear,
    ContrastiveHead,
    GhostBottleneck,
    HGBlock,
    HGStem,
    ImagePoolingAttn,
    Proto,
    RepC3,
    RepNCSPELAN4,
    RepVGGDW,
    ResNetLayer,
    SCDown,
    ODConv_3rd,
    C2f_Faster,
    ShuffleV2Block,
    conv_bn_relu_maxpool,
    stem,
    FusedMBConv,
    MBConv,
    SEAttention,
    C3STR,
    C2f_DCN,
    C2f_DySnakeConv,
    C2f_Dual,
    RFAConv,
    SPDConv,
    ADown,
    DiverseBranchBlock,
    C2f_DBB,
)
from .msaa import MSAA
from .rcsosa import RCSOSA
from .DLKA import deformable_LKA_Attention
# from .hat import HAT
from .lfa import FocusedLinearAttention
from .lska import LSKA
from .dat import DAttentionBaseline
from .conv import (
    CBAM,
    ChannelAttention,
    Concat,
    Conv,
    Conv2,
    ConvTranspose,
    DWConv,
    DWConvTranspose2d,
    Focus,
    GhostConv,
    LightConv,
    RepConv,
    SpatialAttention,
    SimAM,
    VoVGSCSPC,
    VoVGSCSP,
    GSConv
)
from .MSDA import MultiDilatelocalAttention
from .ackconv import AKConv
from .haar import Down_wt
from .head import OBB, Classify, Detect, Pose, RTDETRDecoder, Segment, WorldDetect, v10Detect
from .transformer import (
    AIFI,
    MLP,
    DeformableTransformerDecoder,
    DeformableTransformerDecoderLayer,
    LayerNorm2d,
    MLPBlock,
    MSDeformAttn,
    TransformerBlock,
    TransformerEncoderLayer,
    TransformerLayer,
)


__all__ = (
    "MV2Block",
    "MobileViTBlock",
    "Conv",
    "Conv2",
    "LightConv",
    "RepConv",
    "DWConv",
    "DWConvTranspose2d",
    "ConvTranspose",
    "Focus",
    "GhostConv",
    "ChannelAttention",
    "SpatialAttention",
    "CBAM",
    "Concat",
    "TransformerLayer",
    "TransformerBlock",
    "MLPBlock",
    "LayerNorm2d",
    "DFL",
    "HGBlock",
    "HGStem",
    "SPP",
    "SPPF",
    "C1",
    "C2",
    "C3",
    "C2f",
    "C3k2",
    "SCDown",
    "C2fPSA",
    "C2PSA",
    "C2fAttn",
    "C3x",
    "C3TR",
    "C3Ghost",
    "GhostBottleneck",
    "Bottleneck",
    "BottleneckCSP",
    "Proto",
    "Detect",
    "Segment",
    "Pose",
    "Classify",
    "TransformerEncoderLayer",
    "RepC3",
    "RTDETRDecoder",
    "AIFI",
    "DeformableTransformerDecoder",
    "DeformableTransformerDecoderLayer",
    "MSDeformAttn",
    "MLP",
    "ResNetLayer",
    "OBB",
    "WorldDetect",
    "v10Detect",
    "ImagePoolingAttn",
    "ContrastiveHead",
    "BNContrastiveHead",
    "RepNCSPELAN4",
    "ADown",
    "SPPELAN",
    "CBFuse",
    "CBLinear",
    "AConv",
    "ELAN1",
    "RepVGGDW",
    "CIB",
    "C2fCIB",
    "Attention",
    "PSA",
    "SimAM",
    "C2f_Faster",
    "ShuffleV2Block",
    "conv_bn_relu_maxpool",
    "stem",
    "FusedMBConv",
    "MBConv",
    "SEAttention",
    "C3STR",
    "C2f_DCN",
    "C2f_DySnakeConv",
    "C2f_Dual",
    "RFAConv",
    "SPDConv",
    "ADown",
    "DiverseBranchBlock",
    "ContextGuidedBlock_Down",
    "Down_wt",
    "AKConv",
    "OREPA",
    "DynamicConv",
    "MultiDilatelocalAttention",
    "DAttentionBaseline",
    "LSKA",
    "deformable_LKA_Attention",
    "FocusedLinearAttention",
    "RCSOSA",
    "C2f_MSBlock",
    "CSPPC",
    "C2f_DWRSeg",
    "C2f_DBB",
    "MSAA",
    "SAFMNPP",
    "C2f_FasterBlock",
    "BasicStage",
    "PatchEmbed_FasterNet",
    "PatchMerging_FasterNet"
)
