from utils.data import *
from utils.helper import *
from hookedpaligemma import fetch_gemma_model


from PIL import Image
from functools import partial
import torch
from jaxtyping import Float, Int, Bool
import einops
from collections import defaultdict
from tqdm import tqdm

from transformer_lens.utils import get_act_name

def patch_masked_residue(
    target_residual_component: Float[torch.Tensor, "batch pos d_model"],
    hook,
    mask: Bool[torch.Tensor, "batch pos"],
    source_cache,
):
    device = target_residual_component.device
    target_residual_component[mask.to(device), :] = (
        source_cache[hook.name].to(device)[mask.to(device), :].to(device)
    )
    return target_residual_component



def rotary_deltas(
    x: Float[torch.Tensor, "batch pos head_index d_head"],
    pos_deltas: Int[torch.Tensor, "batch pos"],
    attn,
) -> Float[torch.Tensor, "batch pos head_index d_head"]:
    # adapted from components.py -> Attention -> apply_rotary
    x_pos = x.size(1)
    x_rot = x[..., : attn.cfg.rotary_dim]  # batch pos head_index d_head
    x_pass = x[..., attn.cfg.rotary_dim :]
    x_flip = attn.rotate_every_two(x_rot)
    abs_pos = torch.abs(pos_deltas)
    coses = attn.rotary_cos[abs_pos]  # batch pos d_head
    sines = attn.rotary_sin[abs_pos] * torch.sign(pos_deltas)[..., None]
    x_rotated = x_rot * einops.rearrange(
        coses, "b p d -> b p 1 d"
    ) + x_flip * einops.rearrange(sines, "b p d -> b p 1 d")
    return torch.cat([x_rotated, x_pass], dim=-1)


def patch_rotary_k(
    target_k: Float[torch.Tensor, "batch pos head_index d_head"],
    hook,
    pos_deltas: Int[torch.Tensor, "batch pos"],
    rotate_function,
):
    # consistency tests:
    # y = rotate_function(target_k, pos_deltas)
    # x = rotate_function(y, -pos_deltas)
    # assert torch.allclose(target_k, x, rtol=1e-3, atol=1e-4)
    return rotate_function(target_k, pos_deltas.to(target_k.device))


def evaluate(logits, ctx1, tokenizer):
    score = torch.zeros(2)
    items = [ctx1['items']['item1'], ctx1['items']['item2']]
    
    item_tokens = [ tokenizer.encode(item, add_special_tokens=False)[0] for item in items] 
    logits = logits.squeeze()[-1]
    item_logits = logits[item_tokens]
    item_log_probs = item_logits - logits.logsumexp(dim=-1, keepdim=False)

    val, ind = item_log_probs.max(-1)
    score[ind.item()] = 1.


    return item_log_probs, score


def position_intervene_object(*, model, vocab, num_samples, num_layers, device):
    position_scores = defaultdict(lambda: torch.zeros((2,2)))
    position_log_probs = defaultdict(lambda: torch.zeros(2, num_samples, 2))

  
    token_diff = 144
    lower_delta, upper_delta = -24, token_diff+24


    for idx in tqdm(range(num_samples)):

        ctx1, _ = vocab.get_sample_dicts(idx)
        ctx1_text, ctx1_indices, _, _ = vocab.get_samples(idx, 0)

        _, ctx1_aligned_indices = align_token_indices(model.processor.tokenizer, ctx1_text, ctx1_indices)

        ctx1_image = Image.open(ctx1['image_path']).convert('RGB')
        ctx1_inputs = model.processor(images=ctx1_image, text=ctx1_text, return_tensors="pt").to(device)

        _, ctx1_cache = model.run_with_cache(**ctx1_inputs, names_filter=lambda x: any(y in x for y in ['resid_pre']))
       
        target_dependent_mask = torch.zeros(
            (1, num_layers, ctx1_inputs.input_ids.shape[1]), dtype=bool
        )
        dependent_start = ctx1_aligned_indices['context'][1]
        target_dependent_mask[:, :, dependent_start:] = True
        source_mask = ~target_dependent_mask

        for delta in range(lower_delta, upper_delta+1, 24):

            pos_deltas = torch.zeros((1, ctx1_inputs.input_ids.shape[1]), dtype=int)
            for pos1, pos2 in zip(vocab.image_patches['shape1'], vocab.image_patches['shape2']):
                pos_deltas[:,pos1.to_slice()] = delta
                pos_deltas[:,pos2.to_slice()] = -delta

            fwd_hooks = []
            for layer in range(num_layers):
                fwd_hooks.append((
                    get_act_name('resid_pre', layer),
                    partial(patch_masked_residue,mask=source_mask[:,layer,:],source_cache=ctx1_cache)
                ))
                fwd_hooks.append((
                    get_act_name('rot_k', layer),
                    partial(patch_rotary_k, 
                            pos_deltas=pos_deltas,
                            rotate_function=partial(rotary_deltas, attn=model.hooked_language_model.blocks[layer].attn)
                    )
                ))

            for shape_id in range(2):
                ctx1_qn,_,_,_ = vocab.get_samples(idx, shape_id)
                inputs = model.processor(images=ctx1_image, text=ctx1_qn, return_tensors='pt').to(device)
                logits = model.run_with_hooks(**inputs, fwd_hooks=fwd_hooks)
                item_log_probs, score = evaluate(logits, ctx1, model.processor.tokenizer)
                
                position_scores[delta][shape_id] += score
                position_log_probs[delta][shape_id, idx] = item_log_probs


        print(idx,'____________Scores________________')
        print(*position_scores.keys())
        print(*(position_scores.values()),sep='\n\n')

    for key in position_log_probs:
        log_probs = position_log_probs[key].mean(dim=-2)
        position_log_probs[key] = log_probs

    
    print('____________Scores________________')
    print(*position_scores.keys())
    print(*(position_scores.values()),sep='\n\n')

    print('____________Log_Probs______________')
    print(*position_log_probs.keys())
    print(*(position_log_probs.values()), sep='\n\n')
            




def position_intervene_item(*, model, vocab, num_samples, num_layers, device):

    position_scores = defaultdict(lambda: torch.zeros((2,2)))
    position_log_probs = defaultdict(lambda: torch.zeros(2, num_samples, 2))

    for idx in tqdm(range(num_samples)):

        ctx1, _ = vocab.get_sample_dicts(idx)
        ctx1_text, ctx1_indices, _, _ = vocab.get_samples(idx, 0)

        _, ctx1_aligned_indices = align_token_indices(model.processor.tokenizer, ctx1_text, ctx1_indices, num_image_patches=1024+2)

        ctx1_image = Image.open(ctx1['image_path']).convert('RGB')
        ctx1_inputs = model.processor(images=ctx1_image, text=ctx1_text, return_tensors="pt").to(device)

        _, ctx1_cache = model.run_with_cache(**ctx1_inputs, names_filter=lambda x: any(y in x for y in ['resid_pre']))
        # target_dependent_mask = torch.zeros(
        #     (1, num_layers, ctx1_inputs.input_ids.shape[1]+(576-1)), dtype=bool
        # )
        target_dependent_mask = torch.zeros(
            (1, num_layers, ctx1_inputs.input_ids.shape[1]), dtype=bool
        )
        dependent_start = ctx1_aligned_indices['context'][1]
        target_dependent_mask[:, :, dependent_start:] = True
        source_mask = ~target_dependent_mask

        item1_pos, item2_pos = ctx1_aligned_indices['item1'], ctx1_aligned_indices['item2']
        token_diff = item2_pos[0] - item1_pos[0]
        if idx == 0:
            print('TOKEN_DIFF:', token_diff)
        lower_delta, upper_delta = -4, token_diff+4

        # for delta in range(lower_delta, upper_delta+1, 4):
        for delta in [-3, 0, 3, 7, 10]:

            # pos_deltas = torch.zeros((1, ctx1_inputs.input_ids.shape[1]+(576-1)), dtype=int)
            pos_deltas = torch.zeros((1, ctx1_inputs.input_ids.shape[1]), dtype=int)

            pos_deltas[:,item1_pos.to_slice()] = delta
            pos_deltas[:,item2_pos.to_slice()] = -delta

            fwd_hooks = []
            for layer in range(num_layers):
                fwd_hooks.append((
                    get_act_name('resid_pre', layer),
                    partial(patch_masked_residue,mask=source_mask[:,layer,:],source_cache=ctx1_cache)
                ))
                fwd_hooks.append((
                    get_act_name('rot_k', layer),
                    partial(patch_rotary_k, 
                            pos_deltas=pos_deltas,
                            rotate_function=partial(rotary_deltas, attn=model.hooked_language_model.blocks[layer].attn)
                    )
                ))


            for shape_id in range(2):
                ctx1_qn,_,_,_ = vocab.get_samples(idx, shape_id)
                inputs = model.processor(images=ctx1_image, text=ctx1_qn, return_tensors='pt').to(device)
                logits = model.run_with_hooks(**inputs, fwd_hooks=fwd_hooks)
                item_log_probs, score = evaluate(logits, ctx1, model.processor.tokenizer)
                
                position_scores[delta][shape_id] += score
                position_log_probs[delta][shape_id, idx] = item_log_probs

                # print('idx:{} shaped_id:{} delta:{}'.format(idx,shape_id,delta),position_scores)

        print(idx,'____________Scores________________')
        print(*position_scores.keys())
        print(*(position_scores.values()),sep='\n\n')

    for key in position_log_probs:
        log_probs = position_log_probs[key].mean(dim=-2)
        position_log_probs[key] = log_probs

    
    print('____________Scores________________')
    print(*position_scores.keys())
    print(*(position_scores.values()),sep='\n\n')

    print('____________Log_Probs______________')
    print(*position_log_probs.keys())
    print(*(position_log_probs.values()), sep='\n\n')
            


def get_gemma_details():
    template = """Answer the question based on the provided image and the context below. Keep the answer short.

Context:{context}

Question: Which item does the {qn_shape} contain?

Answer: The {ans_shape} contains item  """

    context_template = 'The {color} object contains item {letter}.'

    image_offset = 0
    shape1_patches = [Substring(*tup)+image_offset for tup in [
        (103,108+1),(135,140+1),(167,172+1),(199,204+1),(231,236+1),(263,268+1)
    ]]
    shape2_patches = [Substring(*tup)+image_offset for tup in [
        (247,252+1),(279,284+1),(311,316+1),(343,348+1),(373,380+1),(407,412+1)
    ]]

    return dict(
        template = template,
        context_template = context_template,
        image_offset = image_offset,
        shape1_patches = shape1_patches,
        shape2_patches = shape2_patches
    )
    


def run_pos_intervene_paligemma():

    dataset_name = 'shapes'
    images_root = '/home/darshana/research/vlm-bind/dataset/images_large_448/'
    save_path_temp = './paligemma_results/paligemma_pind_{}_{}.pth'

    vocab = ShapesItems(csv_path='/home/darshana/research/vlm-bind/dataset/factorizability.csv', 
                        img_path=images_root, mapping=False,flip=False, order_flip=False)
    gemma_deets = get_gemma_details()
    vocab.set_details(**gemma_deets, flip=False)

    num_samples = len(vocab.rows)

    device = torch.device('cuda')
    # device = torch.device('cpu')
    model = fetch_gemma_model(device='cuda',num_devices=1)
    model.to(device=device)
    num_layers = model.hooked_language_model.cfg.n_layers    

    position_intervene_item(model=model, vocab=vocab, num_samples=num_samples, num_layers=num_layers, 
                    device=device)   
    # position_intervene_object(model=model, vocab=vocab, num_samples=num_samples, num_layers=num_layers, 
    #                 device=device)



run_pos_intervene_paligemma()
