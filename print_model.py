from transformers import AutoModel,AutoTokenizer,AutoConfig,AutoModelForSeq2SeqLM

# Load model
model_checkpoint = "/home/maxime/Documents/LUMIA/Project/transformers-translation/opus-mt-fr-en"
model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)
config = AutoConfig.from_pretrained(model_checkpoint)

print("Model Config:")
print(config)
print("------------------------------------------------")
print("Model architecture:")
print(model)
print("------------------------------------------------")
# Print number of parameters
print("Number of parameters:")
print(str(model.num_parameters()/1000000)+"M")

# Model Config:
# MarianConfig {
#   "_name_or_path": "/home/maxime/Documents/LUMIA/Project/transformers-translation/opus-mt-fr-en",
#   "_num_labels": 3,
#   "activation_dropout": 0.0,
#   "activation_function": "swish",
#   "add_bias_logits": false,
#   "add_final_layer_norm": false,
#   "architectures": [
#     "MarianMTModel"
#   ],
#   "attention_dropout": 0.0,
#   "bad_words_ids": [
#     [
#       59513
#     ]
#   ],
#   "bos_token_id": 0,
#   "classif_dropout": 0.0,
#   "classifier_dropout": 0.0,
#   "d_model": 512,
#   "decoder_attention_heads": 8,
#   "decoder_ffn_dim": 2048,
#   "decoder_layerdrop": 0.0,
#   "decoder_layers": 6,
#   "decoder_start_token_id": 59513,
#   "decoder_vocab_size": 59514,
#   "dropout": 0.1,
#   "encoder_attention_heads": 8,
#   "encoder_ffn_dim": 2048,
#   "encoder_layerdrop": 0.0,
#   "encoder_layers": 6,
#   "eos_token_id": 0,
#   "forced_eos_token_id": 0,
#   "gradient_checkpointing": false,
#   "id2label": {
#     "0": "LABEL_0",
#     "1": "LABEL_1",
#     "2": "LABEL_2"
#   },
#   "init_std": 0.02,
#   "is_encoder_decoder": true,
#   "label2id": {
#     "LABEL_0": 0,
#     "LABEL_1": 1,
#     "LABEL_2": 2
#   },
#   "max_length": 512,
#   "max_position_embeddings": 512,
#   "model_type": "marian",
#   "normalize_before": false,
#   "normalize_embedding": false,
#   "num_beams": 4,
#   "num_hidden_layers": 6,
#   "pad_token_id": 59513,
#   "scale_embedding": true,
#   "share_encoder_decoder_embeddings": true,
#   "static_position_embeddings": true,
#   "transformers_version": "4.28.1",
#   "use_cache": true,
#   "vocab_size": 59514
# }

# ------------------------------------------------
# Model architecture:
# MarianMTModel(
#   (model): MarianModel(
#     (shared): Embedding(59514, 512, padding_idx=59513)
#     (encoder): MarianEncoder(
#       (embed_tokens): Embedding(59514, 512, padding_idx=59513)
#       (embed_positions): MarianSinusoidalPositionalEmbedding(512, 512)
#       (layers): ModuleList(
#         (0): MarianEncoderLayer(
#           (self_attn): MarianAttention(
#             (k_proj): Linear(in_features=512, out_features=512, bias=True)
#             (v_proj): Linear(in_features=512, out_features=512, bias=True)
#             (q_proj): Linear(in_features=512, out_features=512, bias=True)
#             (out_proj): Linear(in_features=512, out_features=512, bias=True)
#           )
#           (self_attn_layer_norm): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
#           (activation_fn): SiLUActivation()
#           (fc1): Linear(in_features=512, out_features=2048, bias=True)
#           (fc2): Linear(in_features=2048, out_features=512, bias=True)
#           (final_layer_norm): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
#         )
#         (1): MarianEncoderLayer(
#           (self_attn): MarianAttention(
#             (k_proj): Linear(in_features=512, out_features=512, bias=True)
#             (v_proj): Linear(in_features=512, out_features=512, bias=True)
#             (q_proj): Linear(in_features=512, out_features=512, bias=True)
#             (out_proj): Linear(in_features=512, out_features=512, bias=True)
#           )
#           (self_attn_layer_norm): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
#           (activation_fn): SiLUActivation()
#           (fc1): Linear(in_features=512, out_features=2048, bias=True)
#           (fc2): Linear(in_features=2048, out_features=512, bias=True)
#           (final_layer_norm): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
#         )
#         (2): MarianEncoderLayer(
#           (self_attn): MarianAttention(
#             (k_proj): Linear(in_features=512, out_features=512, bias=True)
#             (v_proj): Linear(in_features=512, out_features=512, bias=True)
#             (q_proj): Linear(in_features=512, out_features=512, bias=True)
#             (out_proj): Linear(in_features=512, out_features=512, bias=True)
#           )
#           (self_attn_layer_norm): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
#           (activation_fn): SiLUActivation()
#           (fc1): Linear(in_features=512, out_features=2048, bias=True)
#           (fc2): Linear(in_features=2048, out_features=512, bias=True)
#           (final_layer_norm): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
#         )
#         (3): MarianEncoderLayer(
#           (self_attn): MarianAttention(
#             (k_proj): Linear(in_features=512, out_features=512, bias=True)
#             (v_proj): Linear(in_features=512, out_features=512, bias=True)
#             (q_proj): Linear(in_features=512, out_features=512, bias=True)
#             (out_proj): Linear(in_features=512, out_features=512, bias=True)
#           )
#           (self_attn_layer_norm): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
#           (activation_fn): SiLUActivation()
#           (fc1): Linear(in_features=512, out_features=2048, bias=True)
#           (fc2): Linear(in_features=2048, out_features=512, bias=True)
#           (final_layer_norm): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
#         )
#         (4): MarianEncoderLayer(
#           (self_attn): MarianAttention(
#             (k_proj): Linear(in_features=512, out_features=512, bias=True)
#             (v_proj): Linear(in_features=512, out_features=512, bias=True)
#             (q_proj): Linear(in_features=512, out_features=512, bias=True)
#             (out_proj): Linear(in_features=512, out_features=512, bias=True)
#           )
#           (self_attn_layer_norm): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
#           (activation_fn): SiLUActivation()
#           (fc1): Linear(in_features=512, out_features=2048, bias=True)
#           (fc2): Linear(in_features=2048, out_features=512, bias=True)
#           (final_layer_norm): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
#         )
#         (5): MarianEncoderLayer(
#           (self_attn): MarianAttention(
#             (k_proj): Linear(in_features=512, out_features=512, bias=True)
#             (v_proj): Linear(in_features=512, out_features=512, bias=True)
#             (q_proj): Linear(in_features=512, out_features=512, bias=True)
#             (out_proj): Linear(in_features=512, out_features=512, bias=True)
#           )
#           (self_attn_layer_norm): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
#           (activation_fn): SiLUActivation()
#           (fc1): Linear(in_features=512, out_features=2048, bias=True)
#           (fc2): Linear(in_features=2048, out_features=512, bias=True)
#           (final_layer_norm): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
#         )
#       )
#     )
#     (decoder): MarianDecoder(
#       (embed_tokens): Embedding(59514, 512, padding_idx=59513)
#       (embed_positions): MarianSinusoidalPositionalEmbedding(512, 512)
#       (layers): ModuleList(
#         (0): MarianDecoderLayer(
#           (self_attn): MarianAttention(
#             (k_proj): Linear(in_features=512, out_features=512, bias=True)
#             (v_proj): Linear(in_features=512, out_features=512, bias=True)
#             (q_proj): Linear(in_features=512, out_features=512, bias=True)
#             (out_proj): Linear(in_features=512, out_features=512, bias=True)
#           )
#           (activation_fn): SiLUActivation()
#           (self_attn_layer_norm): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
#           (encoder_attn): MarianAttention(
#             (k_proj): Linear(in_features=512, out_features=512, bias=True)
#             (v_proj): Linear(in_features=512, out_features=512, bias=True)
#             (q_proj): Linear(in_features=512, out_features=512, bias=True)
#             (out_proj): Linear(in_features=512, out_features=512, bias=True)
#           )
#           (encoder_attn_layer_norm): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
#           (fc1): Linear(in_features=512, out_features=2048, bias=True)
#           (fc2): Linear(in_features=2048, out_features=512, bias=True)
#           (final_layer_norm): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
#         )
#         (1): MarianDecoderLayer(
#           (self_attn): MarianAttention(
#             (k_proj): Linear(in_features=512, out_features=512, bias=True)
#             (v_proj): Linear(in_features=512, out_features=512, bias=True)
#             (q_proj): Linear(in_features=512, out_features=512, bias=True)
#             (out_proj): Linear(in_features=512, out_features=512, bias=True)
#           )
#           (activation_fn): SiLUActivation()
#           (self_attn_layer_norm): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
#           (encoder_attn): MarianAttention(
#             (k_proj): Linear(in_features=512, out_features=512, bias=True)
#             (v_proj): Linear(in_features=512, out_features=512, bias=True)
#             (q_proj): Linear(in_features=512, out_features=512, bias=True)
#             (out_proj): Linear(in_features=512, out_features=512, bias=True)
#           )
#           (encoder_attn_layer_norm): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
#           (fc1): Linear(in_features=512, out_features=2048, bias=True)
#           (fc2): Linear(in_features=2048, out_features=512, bias=True)
#           (final_layer_norm): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
#         )
#         (2): MarianDecoderLayer(
#           (self_attn): MarianAttention(
#             (k_proj): Linear(in_features=512, out_features=512, bias=True)
#             (v_proj): Linear(in_features=512, out_features=512, bias=True)
#             (q_proj): Linear(in_features=512, out_features=512, bias=True)
#             (out_proj): Linear(in_features=512, out_features=512, bias=True)
#           )
#           (activation_fn): SiLUActivation()
#           (self_attn_layer_norm): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
#           (encoder_attn): MarianAttention(
#             (k_proj): Linear(in_features=512, out_features=512, bias=True)
#             (v_proj): Linear(in_features=512, out_features=512, bias=True)
#             (q_proj): Linear(in_features=512, out_features=512, bias=True)
#             (out_proj): Linear(in_features=512, out_features=512, bias=True)
#           )
#           (encoder_attn_layer_norm): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
#           (fc1): Linear(in_features=512, out_features=2048, bias=True)
#           (fc2): Linear(in_features=2048, out_features=512, bias=True)
#           (final_layer_norm): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
#         )
#         (3): MarianDecoderLayer(
#           (self_attn): MarianAttention(
#             (k_proj): Linear(in_features=512, out_features=512, bias=True)
#             (v_proj): Linear(in_features=512, out_features=512, bias=True)
#             (q_proj): Linear(in_features=512, out_features=512, bias=True)
#             (out_proj): Linear(in_features=512, out_features=512, bias=True)
#           )
#           (activation_fn): SiLUActivation()
#           (self_attn_layer_norm): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
#           (encoder_attn): MarianAttention(
#             (k_proj): Linear(in_features=512, out_features=512, bias=True)
#             (v_proj): Linear(in_features=512, out_features=512, bias=True)
#             (q_proj): Linear(in_features=512, out_features=512, bias=True)
#             (out_proj): Linear(in_features=512, out_features=512, bias=True)
#           )
#           (encoder_attn_layer_norm): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
#           (fc1): Linear(in_features=512, out_features=2048, bias=True)
#           (fc2): Linear(in_features=2048, out_features=512, bias=True)
#           (final_layer_norm): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
#         )
#         (4): MarianDecoderLayer(
#           (self_attn): MarianAttention(
#             (k_proj): Linear(in_features=512, out_features=512, bias=True)
#             (v_proj): Linear(in_features=512, out_features=512, bias=True)
#             (q_proj): Linear(in_features=512, out_features=512, bias=True)
#             (out_proj): Linear(in_features=512, out_features=512, bias=True)
#           )
#           (activation_fn): SiLUActivation()
#           (self_attn_layer_norm): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
#           (encoder_attn): MarianAttention(
#             (k_proj): Linear(in_features=512, out_features=512, bias=True)
#             (v_proj): Linear(in_features=512, out_features=512, bias=True)
#             (q_proj): Linear(in_features=512, out_features=512, bias=True)
#             (out_proj): Linear(in_features=512, out_features=512, bias=True)
#           )
#           (encoder_attn_layer_norm): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
#           (fc1): Linear(in_features=512, out_features=2048, bias=True)
#           (fc2): Linear(in_features=2048, out_features=512, bias=True)
#           (final_layer_norm): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
#         )
#         (5): MarianDecoderLayer(
#           (self_attn): MarianAttention(
#             (k_proj): Linear(in_features=512, out_features=512, bias=True)
#             (v_proj): Linear(in_features=512, out_features=512, bias=True)
#             (q_proj): Linear(in_features=512, out_features=512, bias=True)
#             (out_proj): Linear(in_features=512, out_features=512, bias=True)
#           )
#           (activation_fn): SiLUActivation()
#           (self_attn_layer_norm): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
#           (encoder_attn): MarianAttention(
#             (k_proj): Linear(in_features=512, out_features=512, bias=True)
#             (v_proj): Linear(in_features=512, out_features=512, bias=True)
#             (q_proj): Linear(in_features=512, out_features=512, bias=True)
#             (out_proj): Linear(in_features=512, out_features=512, bias=True)
#           )
#           (encoder_attn_layer_norm): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
#           (fc1): Linear(in_features=512, out_features=2048, bias=True)
#           (fc2): Linear(in_features=2048, out_features=512, bias=True)
#           (final_layer_norm): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
#         )
#       )
#     )
#   )
#   (lm_head): Linear(in_features=512, out_features=59514, bias=False)
# )
# ------------------------------------------------
# Number of parameters:
# 75.133952M