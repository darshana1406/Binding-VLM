import gc
from typing import Optional
from PIL import Image


import torch
import transformer_lens.HookedTransformer as HookedTransformer
from transformers import AutoProcessor, PaliGemmaForConditionalGeneration


class HookedPaliGemma():

    def __init__(self, cache_dir, device='cuda',num_devices=1):
        global model

        model = PaliGemmaForConditionalGeneration.from_pretrained("google/paligemma-3b-mix-448", cache_dir=cache_dir, device_map='cpu', torch_dtype=torch.float32,
                                                attn_implementation="eager")
        self.processor = AutoProcessor.from_pretrained("google/paligemma-3b-mix-448")
        self.config = model.config
        self.lm_config = model.language_model.config

        self.vision_tower = model.vision_tower
        self.multi_modal_projector = model.multi_modal_projector
        self.vocab_size = model.config.text_config.vocab_size
        self._attn_implementation = model.config._attn_implementation

        self.pad_token_id = model.config.pad_token_id if model.config.pad_token_id is not None else -1
        self.embed_tokens = model.language_model.model.embed_tokens

        self.hooked_language_model = HookedTransformer.from_pretrained(
            'gemma-2b',
            hf_model=model.language_model,
            center_unembed=False,
            center_writing_weights=False,
            fold_ln=False,
            refactor_factored_attn_matrices=False,
            fold_value_biases=False,
            device=device,
            n_devices=num_devices,
            dtype=torch.float32
        )

        self.hooked_language_model.cfg.attention_dir = 'bidirectional'
        self.hooked_language_model.set_tokenizer(self.processor.tokenizer, 
                                                 default_padding_side=self.processor.tokenizer.padding_side)

        del model
        gc.collect()

    @torch.no_grad
    def _update_causal_mask(
        self, attention_mask, token_type_ids, inputs_embeds, past_key_values, cache_position, is_training: bool = False
    ):
        # using_static_cache = isinstance(past_key_values, StaticCache)
        using_static_cache = False
        dtype, device = inputs_embeds.dtype, inputs_embeds.device
        min_dtype = torch.finfo(dtype).min
        sequence_length = inputs_embeds.shape[1]
        if using_static_cache:
            target_length = past_key_values.get_max_length()
        else:
            target_length = (
                attention_mask.shape[-1]
                if isinstance(attention_mask, torch.Tensor)
                else cache_position[0] + sequence_length + 1
            )

        if attention_mask is not None and attention_mask.dim() == 4:
            # In this case we assume that the mask comes already in inverted form and requires no inversion or slicing.
            causal_mask = attention_mask
        else:
            causal_mask = torch.full(
                (sequence_length, target_length), fill_value=min_dtype, dtype=dtype, device=device
            )
            # Causal diagonal mask only if training, otherwise attend to the whole prefix. Training-specific attn for prefix is handled below
            if sequence_length != 1:
                if is_training:
                    causal_mask = torch.triu(causal_mask, diagonal=1)
                else:
                    causal_mask = torch.zeros_like(causal_mask)

        causal_mask *= torch.arange(target_length, device=device) > cache_position.reshape(-1, 1)
        causal_mask = causal_mask[None, None, :, :].expand(inputs_embeds.shape[0], 1, -1, -1)
        if attention_mask is not None:
            causal_mask = causal_mask.clone()  # copy to contiguous memory for in-place edit
            mask_length = attention_mask.shape[-1]
            padding_mask = causal_mask[:, :, :, :mask_length] + attention_mask[:, None, None, :].to(causal_mask.device)
            padding_mask = padding_mask == 0
            causal_mask[:, :, :, :mask_length] = causal_mask[:, :, :, :mask_length].masked_fill(
                padding_mask, min_dtype
            )
            # we are training thus we need to create a full mask on the image + prefix but causal on suffix
            if is_training:
                causal_mask[:, :, :, :mask_length] = causal_mask[:, :, :, :mask_length].masked_fill(
                    token_type_ids[:, None, None, :].to(causal_mask.device) == 0, 0
                )
        return causal_mask


    @torch.no_grad
    def pre_forward(
        self,
        input_ids: torch.LongTensor = None,
        pixel_values: torch.FloatTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        no_norm = False
    ):
        # output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        # output_hidden_states = (
        #     output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        # )
        # return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # is_training = token_type_ids is not None and labels is not None

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)


        if cache_position is None:
            past_seen_tokens = 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0) + 1  # Paligemma positions are 1-indexed

        # Merge text and images
        if pixel_values is not None:
            image_outputs = self.vision_tower(pixel_values.to(inputs_embeds.dtype))
            selected_image_feature = image_outputs.last_hidden_state
            image_features = self.multi_modal_projector(selected_image_feature)
            image_features = image_features / (self.config.hidden_size**0.5)

            special_image_mask = (input_ids == self.config.image_token_index).unsqueeze(-1).expand_as(inputs_embeds)
            if inputs_embeds[special_image_mask].numel() != image_features.numel():
                image_tokens_in_text = torch.sum(input_ids == self.config.image_token_index)
                raise ValueError(
                    f"Number of images does not match number of special image tokens in the input text. "
                    f"Got {image_tokens_in_text} image tokens in the text but {image_features.shape[0] * image_features.shape[1]} "
                    "tokens from image embeddings."
                )
            image_features = image_features.to(inputs_embeds.device, inputs_embeds.dtype)
            inputs_embeds = inputs_embeds.masked_scatter(special_image_mask, image_features)

        # mask out pad-token-ids in labels for BC
        if labels is not None and self.pad_token_id in labels:
            print(
                "`labels` contains `pad_token_id` which will be masked with `config.ignore_index`. ",
                "You have to mask out `pad_token_id` when preparing `labels`, this behavior will be removed in v.4.46.",
            )
            labels = torch.where(input_ids == self.pad_token_id, self.config.ignore_index, labels)

        causal_mask = self._update_causal_mask(
            attention_mask, token_type_ids, inputs_embeds, None, cache_position, False
        )
        if not no_norm:
            inputs_embeds = inputs_embeds * torch.tensor(self.lm_config.hidden_size**0.5, dtype=inputs_embeds.dtype)
        return causal_mask, position_ids, inputs_embeds, cache_position
    

    @torch.no_grad
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        pixel_values: torch.FloatTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
    ):

        causal_mask, position_ids, inputs_embeds, cache_position = self.pre_forward(
                                                    input_ids=input_ids, pixel_values=pixel_values, attention_mask=attention_mask)
        outputs = self.hooked_language_model(
                inputs_embeds, start_at_layer=0, shortformer_pos_embed=None#, attention_mask=causal_mask
        )

        return outputs
    
    @torch.no_grad
    def run_with_cache(
        self,
        input_ids: torch.LongTensor = None,
        pixel_values: torch.FloatTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        names_filter = None
    ):
        causal_mask, position_ids, inputs_embeds, cache_position = self.pre_forward(
                                                    input_ids=input_ids, pixel_values=pixel_values, attention_mask=attention_mask)
        logits, cache = self.hooked_language_model.run_with_cache(
            inputs_embeds, names_filter=names_filter, start_at_layer=0, shortformer_pos_embed=None
        )

        return logits, cache
    
    @torch.no_grad  
    def run_with_hooks(self, input_ids, pixel_values, attention_mask, fwd_hooks = []):
        
        causal_mask, position_ids, inputs_embeds, cache_position = self.pre_forward(
                                                    input_ids=input_ids, pixel_values=pixel_values, attention_mask=attention_mask)
        logits = self.hooked_language_model.run_with_hooks(
            inputs_embeds, start_at_layer=0, shortformer_pos_embed=None, fwd_hooks=fwd_hooks
        )

        return logits
    
    def to(self, device):
        self.vision_tower.to(device)
        self.multi_modal_projector.to(device)
        self.embed_tokens.to(device)
        # self.hooked_language_model.to(device)
    

def fetch_gemma_model(cache_dir='/mnt/Shared-Storage/darshana/hf/hub', device='cpu',num_devices=1):
    
    model = HookedPaliGemma(cache_dir=cache_dir,device=device,num_devices=num_devices)
    return model
