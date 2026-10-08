"""Swin-T scene backbone used by OVGOP."""

from typing import List

from torch import nn

from util.misc import NestedTensor

from .position_encoding import build_position_encoding
from .swin_transformer import build_swin_transformer


class Joiner(nn.Sequential):
    def __init__(self, backbone, position_embedding):
        super().__init__(backbone, position_embedding)

    def forward(self, tensor_list: NestedTensor):
        features = self[0](tensor_list)
        outputs: List[NestedTensor] = []
        positions = []
        for feature in features.values():
            outputs.append(feature)
            positions.append(self[1](feature).to(feature.tensors.dtype))
        return outputs, positions


def build_backbone(args):
    if args.scene_backbone != "swin_T_224_1k":
        raise ValueError(
            f"This release only supports the Swin-T scene backbone, got {args.scene_backbone!r}"
        )

    return_indices = args.return_interm_indices
    if return_indices != [1, 2, 3]:
        raise ValueError(f"Expected Swin-T stages [1, 2, 3], got {return_indices}")

    backbone = build_swin_transformer(
        args.scene_backbone,
        pretrain_img_size=224,
        out_indices=tuple(return_indices),
        dilation=False,
        use_checkpoint=getattr(args, "use_checkpoint", False),
    )
    model = Joiner(backbone, build_position_encoding(args))
    model.num_channels = backbone.num_features[1:]
    return model
