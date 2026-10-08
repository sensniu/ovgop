# ------------------------------------------------------------------------
# Mofified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# ------------------------------------------------------------------------

"""
Backbone modules.
"""
import os
from unicodedata import category
import torch
import torch.nn.functional as F
import torchvision
from torch import nn
from torchvision.models._utils import IntermediateLayerGetter
from typing import Dict, List
from models.clip import clip
from util.misc import NestedTensor, is_main_process
from .position_encoding import build_position_encoding
from models.clip.clip import _MODELS, _download, available_models, tokenize
from models.clip.model import Transformer
from models.clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
from models.clip.prompts import imagenet_templates

_tokenizer = _Tokenizer()
class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class Classifier(torch.nn.Module):
    def __init__(self, args, classnames, name: str, TEMPLATES,
                 token_len=77,
                 classifier_cache=''):
        super().__init__()
        if classifier_cache == '':
            self.cache = {}
        else:
            self.cache = torch.load(classifier_cache)

        self.template = TEMPLATES

        if "clip" in name:
            name = name.replace('clip_', '')
            model_path = args.model_path
            if model_path:
                if not os.path.isfile(model_path):
                    raise FileNotFoundError(f"CLIP model not found: {model_path}")
            elif name in _MODELS:
                model_path = _download(_MODELS[name], os.path.expanduser("~/.cache/clip"))
            elif os.path.isfile(name):
                model_path = name
            else:
                raise RuntimeError(f"Model {name} not found; available models = {available_models()}")

            with open(model_path, 'rb') as opened_file:
                model = torch.jit.load(opened_file, map_location="cpu")
                state_dict = model.state_dict()

            embed_dim = state_dict["text_projection"].shape[1]

            self.context_length = 77
            self.vocab_size = state_dict["token_embedding.weight"].shape[0]
            transformer_width = state_dict["ln_final.weight"].shape[0]
            transformer_heads = transformer_width // 64
            transformer_layers = len(set(k.split(".")[2] for k in state_dict if k.startswith("transformer.resblocks")))

            self.transformer = Transformer(
                width=transformer_width,
                layers=transformer_layers,
                heads=transformer_heads,
                attn_mask=self.build_attention_mask()
            )

            self.token_embedding = nn.Embedding(self.vocab_size, transformer_width)
            self.object_positional_embedding = nn.Parameter(torch.empty(self.context_length, transformer_width))
            self.direction_positional_embedding = nn.Parameter(torch.empty(self.context_length, transformer_width))

            self.ln_final = LayerNorm(transformer_width)
            self.text_projection = nn.Parameter(torch.empty(transformer_width, embed_dim))

            # load
            self.transformer.load_state_dict(
                {k.replace('transformer.', ''): v for k, v in state_dict.items() if k.startswith('transformer.')})
            self.token_embedding.load_state_dict(
                {k.replace('token_embedding.', ''): v for k, v in state_dict.items() if 'token_embedding' in k})
            self.ln_final.load_state_dict(
                {k.replace('ln_final.', ''): v for k, v in state_dict.items() if 'ln_final' in k})
            self.object_positional_embedding.data = state_dict['positional_embedding']
            self.direction_positional_embedding.data = state_dict['positional_embedding']
            self.text_projection.data = state_dict['text_projection']

            for v in self.parameters():
                v.requires_grad_(False)

            object_classnames = classnames['object']
            direction_classnames = classnames['direction']

            n_cls_object = len(object_classnames)
            n_cls_direction = len(direction_classnames)
            n_ctx = args.N_CTX
            ctx_dim = state_dict["ln_final.weight"].shape[0]

            print("Initializing a generic context")
            ctx_vectors = torch.empty(n_ctx, ctx_dim)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)

            print(f'Initial context: "{prompt_prefix}"')
            print(f"Number of context words (tokens): {n_ctx}")

            self.ctx_object = nn.Parameter(ctx_vectors)
            self.ctx_direction = nn.Parameter(ctx_vectors)

            object_classnames = [name.replace("_", " ") for name in object_classnames]
            object_name_lens = [len(_tokenizer.encode(name)) for name in object_classnames]
            object_prompts = [prompt_prefix + " " + name + "." for name in object_classnames]

            direction_classnames = [name.replace("_", " ") for name in direction_classnames]
            direction_name_lens = [len(_tokenizer.encode(name)) for name in direction_classnames]
            direction_prompts = [prompt_prefix + " " + name + "." for name in direction_classnames]

            tokenized_object_prompts = torch.cat([tokenize(p) for p in object_prompts])
            tokenized_direction_prompts = torch.cat([tokenize(p) for p in direction_prompts])

            with torch.no_grad():
                object_embedding = self.token_embedding(tokenized_object_prompts)
                direction_embedding = self.token_embedding(tokenized_direction_prompts)
            # These token vectors will be saved when in save_model(),
            # but they should be ignored in load_model() as we want to use
            # those computed using the current class names
            self.register_buffer("object_token_prefix", object_embedding[:, :1, :])  # SOS
            self.register_buffer("object_token_suffix", object_embedding[:, 1 + n_ctx:, :])  # CLS, EOS
            self.register_buffer("direction_token_prefix", direction_embedding[:, :1, :])  # SOS
            self.register_buffer("direction_token_suffix", direction_embedding[:, 1 + n_ctx:, :])  # CLS, EOS

            self.n_cls_object = n_cls_object
            self.n_cls_direction = n_cls_direction
            self.n_ctx = n_ctx
            self.tokenized_object_prompts = tokenized_object_prompts  # torch.Tensor
            self.tokenized_direction_prompts = tokenized_direction_prompts  # torch.Tensor

            self.object_name_lens = object_name_lens
            self.direction_name_lens = direction_name_lens
            self.class_token_position = args.CLASS_TOKEN_POSITION

        else:
            raise NotImplementedError

    def build_attention_mask(self):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask

    def encode_text(self, text, tokenized_prompts):

        x = text
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x)

        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection.to(x.dtype)

        return x

    def forward_feature(self, text_category):

        if text_category == 'object':
            ctx = self.ctx_object
            if ctx.dim() == 2:
                ctx = ctx.unsqueeze(0).expand(self.n_cls_object, -1, -1)
            prefix = self.object_token_prefix
            suffix = self.object_token_suffix
            if self.class_token_position == "end":
                prompts = torch.cat(
                    [
                        prefix,  # (n_cls, 1, dim)
                        ctx,  # (n_cls, n_ctx, dim)
                        suffix,  # (n_cls, *, dim)
                    ],
                    dim=1,
                )
            elif self.class_token_position == "middle":
                half_n_ctx = self.n_ctx // 2
                prompts = []
                for i in range(self.n_cls):
                    name_len = self.object_name_lens[i]
                    prefix_i = prefix[i: i + 1, :, :]
                    class_i = suffix[i: i + 1, :name_len, :]
                    suffix_i = suffix[i: i + 1, name_len:, :]
                    ctx_i_half1 = ctx[i: i + 1, :half_n_ctx, :]
                    ctx_i_half2 = ctx[i: i + 1, half_n_ctx:, :]
                    prompt = torch.cat(
                        [
                            prefix_i,  # (1, 1, dim)
                            ctx_i_half1,  # (1, n_ctx//2, dim)
                            class_i,  # (1, name_len, dim)
                            ctx_i_half2,  # (1, n_ctx//2, dim)
                            suffix_i,  # (1, *, dim)
                        ],
                        dim=1,
                    )
                    prompts.append(prompt)
                prompts = torch.cat(prompts, dim=0)
            elif self.class_token_position == "front":
                prompts = []
                for i in range(self.n_cls):
                    name_len = self.object_name_lens[i]
                    prefix_i = prefix[i: i + 1, :, :]
                    class_i = suffix[i: i + 1, :name_len, :]
                    suffix_i = suffix[i: i + 1, name_len:, :]
                    ctx_i = ctx[i: i + 1, :, :]
                    prompt = torch.cat(
                        [
                            prefix_i,  # (1, 1, dim)
                            class_i,  # (1, name_len, dim)
                            ctx_i,  # (1, n_ctx, dim)
                            suffix_i,  # (1, *, dim)
                        ],
                        dim=1,
                    )
                    prompts.append(prompt)
                prompts = torch.cat(prompts, dim=0)
            tokenized_prompts = self.tokenized_object_prompts
            texts = prompts + self.object_positional_embedding
        else:
            ctx = self.ctx_direction
            if ctx.dim() == 2:
                ctx = ctx.unsqueeze(0).expand(self.n_cls_direction, -1, -1)
            prefix = self.direction_token_prefix
            suffix = self.direction_token_suffix
            if self.class_token_position == "end":
                prompts = torch.cat(
                    [
                        prefix,  # (n_cls, 1, dim)
                        ctx,  # (n_cls, n_ctx, dim)
                        suffix,  # (n_cls, *, dim)
                    ],
                    dim=1,
                )  # (8, 77, 512)
            elif self.class_token_position == "middle":
                half_n_ctx = self.n_ctx // 2
                prompts = []
                for i in range(self.n_cls):
                    name_len = self.direction_name_lens[i]
                    prefix_i = prefix[i: i + 1, :, :]
                    class_i = suffix[i: i + 1, :name_len, :]
                    suffix_i = suffix[i: i + 1, name_len:, :]
                    ctx_i_half1 = ctx[i: i + 1, :half_n_ctx, :]
                    ctx_i_half2 = ctx[i: i + 1, half_n_ctx:, :]
                    prompt = torch.cat(
                        [
                            prefix_i,  # (1, 1, dim)
                            ctx_i_half1,  # (1, n_ctx//2, dim)
                            class_i,  # (1, name_len, dim)
                            ctx_i_half2,  # (1, n_ctx//2, dim)
                            suffix_i,  # (1, *, dim)
                        ],
                        dim=1,
                    )
                    prompts.append(prompt)
                prompts = torch.cat(prompts, dim=0)
            elif self.class_token_position == "front":
                prompts = []
                for i in range(self.n_cls):
                    name_len = self.direction_name_lens[i]
                    prefix_i = prefix[i: i + 1, :, :]
                    class_i = suffix[i: i + 1, :name_len, :]
                    suffix_i = suffix[i: i + 1, name_len:, :]
                    ctx_i = ctx[i: i + 1, :, :]
                    prompt = torch.cat(
                        [
                            prefix_i,  # (1, 1, dim)
                            class_i,  # (1, name_len, dim)
                            ctx_i,  # (1, n_ctx, dim)
                            suffix_i,  # (1, *, dim)
                        ],
                        dim=1,
                    )
                    prompts.append(prompt)
                prompts = torch.cat(prompts, dim=0)
            tokenized_prompts = self.tokenized_direction_prompts  # (8, 77)
            texts = prompts + self.direction_positional_embedding

        class_embeddings = self.encode_text(texts, tokenized_prompts)

        class_embeddings = class_embeddings / class_embeddings.norm(dim=-1, keepdim=True)

        return class_embeddings

    def forward(self, text_category):
        class_embedding = self.forward_feature(text_category)  # (25, 1024)

        return class_embedding


def build_classifier(args, classnames, TEMPLATES):
    classifier = Classifier(args, classnames, args.clip_backbone, TEMPLATES, token_len=args.text_len, classifier_cache=args.classifier_cache)
    return classifier
