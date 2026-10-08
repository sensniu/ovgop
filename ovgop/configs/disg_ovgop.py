"""OVGOP configuration for the DiSG open-vocabulary split."""

# Model components
modelname = "dino"
num_classes = 87
scene_backbone = "swin_T_224_1k"
clip_backbone = "clip_RN50"
head_backbone = "clip_RN50"
position_embedding = "sine"
pe_temperatureH = 20
pe_temperatureW = 20
return_interm_indices = [1, 2, 3]
use_checkpoint = True

# Transformer
hidden_dim = 256
dim_feedforward = 2048
dropout = 0.0
nheads = 8
enc_layers = 6
dec_layers = 6
pre_norm = False
num_queries = 900
query_dim = 4
num_patterns = 0
num_feature_levels = 4
enc_n_points = 4
dec_n_points = 4
two_stage_type = "standard"
two_stage_bbox_embed_share = False
two_stage_class_embed_share = False
dec_pred_bbox_embed_share = True
dec_pred_class_embed_share = True
embed_init_tgt = True
transformer_activation = "relu"
use_transformer_ckpt = True
use_text_cross_attention = True

# Text and visual alignment
text_len = 15
classifier_cache = ""
head_loc_conv_num = 5
N_CTX = 16
CLASS_TOKEN_POSITION = "end"
sub_sentence_present = True
max_text_len = 256
use_text_enhancer = True
use_fusion_layer = True
text_dropout = 0.0
fusion_dropout = 0.0
fusion_droppath = 0.1
gaze_hidden_size = 256

# Optimization
param_dict_type = "tuning"
lr = 1e-4
lr_backbone = 1e-5
lr_text_encoder = 1e-4
lr_linear_proj_mult = 1e-5
lr_backbone_names = ["backbone.0", "bert"]
lr_linear_proj_names = ["reference_points", "sampling_offsets"]
lr_frozen_od = 1e-6
lr_frozen_od_backbone = 1e-7
lr_frozen_od_linear_proj_mult = 1e-7
batch_size = 16
param_select_batch_size = 10
gist_selection_ratio = 0.4
weight_decay = 1e-4
epochs = 60
lr_drop = 11
save_checkpoint_interval = 1
clip_max_norm = 0.1
onecyclelr = False

# Detection loss and postprocessing
aux_loss = True
matcher_type = "HungarianMatcher"
set_cost_class = 2.0
set_cost_bbox = 5.0
set_cost_giou = 2.0
cls_loss_coef = 1.0
bbox_loss_coef = 5.0
giou_loss_coef = 2.0
interm_loss_coef = 1.0
no_interm_box_loss = False
focal_alpha = 0.25
focal_gamma = 2.0
num_select = 300
nms_iou_threshold = -1

# Evaluation and EMA
use_coco_eval = True
use_ema = False
ema_decay = 0.9997
ema_epoch = 0
