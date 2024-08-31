from utils.data import *
from utils.helper import *
from hookedpaligemma import fetch_gemma_model

from PIL import Image
from functools import partial
import torch
from jaxtyping import Float, Int, Bool
import gc
from tqdm import tqdm

from transformer_lens.utils import get_act_name



def median_calib_acc(logits: torch.Tensor):
    num_shapes = logits.shape[0]
    num_attr = logits.shape[-1]
    scores = torch.zeros((num_shapes, num_attr))
    mean_logits = logits.reshape((-1,4)).quantile(0.5, dim=-2)

    logits -= mean_logits

    for i in range(num_shapes):
        val, ind = logits[i].max(dim=-1)
        for j in range(num_attr):
            scores[i][j] = (ind == j).sum()

    return scores


def replace_cache(src_cache, tgt_cache, src_pos, tgt_pos, num_layers, hook_name):

    for layer in range(num_layers):
        key = get_act_name(hook_name, layer)
        src_cache[key][:,src_pos.to_slice(),:] = tgt_cache[key][:,tgt_pos.to_slice(),:]

    return src_cache


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


def evaluate(logits, ctx1, ctx2, tokenizer):
    score = torch.zeros(4)
    items = [ctx1['items']['item1'], ctx1['items']['item2'],
             ctx2['items']['item1'], ctx2['items']['item2'],
            ]
    item_tokens = [ tokenizer.encode(item, add_special_tokens=False)[0] for item in items] 
    logits = logits.squeeze()[-1]
    item_logits = logits[item_tokens]
    item_log_probs = item_logits - logits.logsumexp(dim=-1, keepdim=False)

    val, ind = item_log_probs.max(-1)
    score[ind.item()] = 1.

    return item_log_probs, score


def intervene(*, model, vocab, num_samples, device, num_layers, intervene_list, batch_size, num_image_patches):
    counts = torch.zeros((4,4))
    log_probs_matrix = torch.zeros((4,num_samples, 4))

    for batch_idx, start_idx in enumerate(tqdm(range(0, num_samples, batch_size))):

        # start_idx = batch_idx * batch_size
        end_idx = min(start_idx+batch_size, num_samples)
        batch_size = end_idx - start_idx

        ctx1_list, ctx2_list = zip(*vocab.get_sample_dicts(list(range(start_idx,end_idx))))
        ctx1_text_list, ctx1_indices_list, ctx2_text_list, ctx2_indices_list = zip(*vocab.get_samples(list(range(start_idx,end_idx)),0))

        ctx1_aligned_indices = [align_token_indices(
            model.processor.tokenizer,
            ctx1_text_list[i],
            ctx1_indices_list[i],
            num_image_patches=num_image_patches
        )[1] for i in range(batch_size)]
        ctx2_aligned_indices = [align_token_indices(
            model.processor.tokenizer,
            ctx2_text_list[i],
            ctx2_indices_list[i],
            num_image_patches=num_image_patches
        )[1] for i in range(batch_size)]
        
        ctx1_img_list = [Image.open(ctx1['image_path']).convert('RGB') for ctx1 in ctx1_list]
        ctx2_img_list = [Image.open(ctx2['image_path']).convert('RGB') for ctx2 in ctx2_list]

        ctx1_inputs = model.processor(images=ctx1_img_list, text=ctx1_text_list, return_tensors="pt").to(device)
        ctx2_inputs = model.processor(images=ctx2_img_list, text=ctx2_text_list, return_tensors="pt").to(device)

        # import pdb; pdb.set_trace()

        _, ctx2_cache = model.run_with_cache(**ctx2_inputs, names_filter=lambda x: any(y in x for y in ['resid_pre']))
        _, ctx1_cache = model.run_with_cache(**ctx1_inputs, names_filter=lambda x: any(y in x for y in ['resid_pre']))

        alter_cache_i1 = ctx1_cache
        for intervene in intervene_list:
            if intervene.startswith('item') or intervene.startswith('color'):
                alter_cache_i1 = replace_cache(alter_cache_i1, ctx2_cache, 
                                        ctx1_aligned_indices[0][intervene], ctx2_aligned_indices[0][intervene], num_layers, 'resid_pre')
            elif intervene.startswith('shape'):
                for img_pos in vocab.image_patches[intervene]:
                    alter_cache_i1 = replace_cache(alter_cache_i1, ctx2_cache,
                                         img_pos, img_pos, num_layers, 'resid_pre')
                

        # target_dependent_mask = torch.zeros(
        #     (batch_size, num_layers, ctx1_inputs.input_ids.shape[1]+(num_image_patches-1)), dtype=bool
        # )
        target_dependent_mask = torch.zeros(
            (batch_size, num_layers, ctx1_inputs.input_ids.shape[1]), dtype=bool
        )
        dependent_start = ctx1_aligned_indices[0]['context'][1]
        target_dependent_mask[:, :, dependent_start:] = True
        source_mask = ~target_dependent_mask

        fwd_hooks=[(
                get_act_name('resid_pre', layer),
                partial(patch_masked_residue,mask=source_mask[:,layer,:],source_cache=alter_cache_i1)
            ) for layer in range(num_layers)]
        
        for shape_id in range(4):
            ctx1_qn,_,ctx2_qn,_ = zip(*vocab.get_samples(list(range(start_idx,end_idx)),shape_id))
            inputs = model.processor(images=ctx1_img_list, text=ctx1_qn, return_tensors='pt').to(device)
            logits = model.run_with_hooks(**inputs, fwd_hooks=fwd_hooks)

            for i, (ctx1, ctx2) in enumerate(zip(ctx1_list, ctx2_list)):
                idx = start_idx+i
                item_logits, score = evaluate(logits[i], ctx1, ctx2, model.processor.tokenizer)
                log_probs_matrix[shape_id][idx] = item_logits
                counts[shape_id] += score
    
            print(idx,counts)

        del ctx1_img_list, ctx2_img_list, ctx1_inputs, ctx2_inputs, inputs
        del ctx1_cache, ctx2_cache, alter_cache_i1, logits
        gc.collect()

    log_probs = log_probs_matrix.mean(dim=-2)

    print('__________________Accuracy_________________')
    print(counts)

    print('__________________Mean Log Probs_________________')
    print(log_probs)

    return dict(accuracy=counts, log_probs=log_probs_matrix)



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



def run_factorizability():

    dataset_name = 'shapes'
    images_root = '/mnt/Shared-Storage/darshana/vlm-bind-data/'
    # images_root = '/home/darshana/research/vlm-bind/dataset/images_large'
    save_path = './results/factorizability_{}_{}.pth'

    vocab = ShapesItems(csv_path='/home/darshana/research/vlm-bind/dataset/factorizability.csv', img_path=images_root, flip=False)
    num_samples = len(vocab.rows)

    device = torch.device('cuda')
    # device = torch.device('cpu')
    model = fetch_model(device='cuda',num_devices=2)
    model.to(device=device)
    num_layers = model.hooked_language_model.cfg.n_layers

#[], ['color1'],['item1'],
    no_intervene = []
    item_intervenes = [['item1'], ['item2'], ['item1', 'item2']]
    patch_intervenes = []
    intervenes = [[]]

    for intervention in intervenes:
        print(intervention)
        save_path = save_path.format(dataset_name, '_'.join(intervention))
        result = intervene(model=model, vocab=vocab, num_samples=num_samples, num_layers=num_layers, 
                       device=device,  intervene_list=intervention, batch_size=4)
        torch.save(result, save_path)



def run_factorizability_gemma():

    dataset_name = 'shapes'
    images_root = '/home/darshana/research/vlm-bind/dataset/images_large_448/'
    save_path_temp = './paligemma_results/paligemma_factorizability_{}_{}.pth'

    vocab = ShapesItems(csv_path='/home/darshana/research/vlm-bind/dataset/factorizability.csv', 
                        img_path=images_root, mapping=False,flip=False, order_flip=True)
    gemma_deets = get_gemma_details()
    vocab.set_details(**gemma_deets, flip=False)

    num_samples = len(vocab.rows)

    device = torch.device('cuda')
    # device = torch.device('cpu')
    model = fetch_gemma_model(device='cuda',num_devices=1)
    model.to(device=device)
    num_layers = model.hooked_language_model.cfg.n_layers

    no_intervene = [[]]
    item_intervenes = [['item1'], ['item2'], ['item1','item2']]
    patch_intervenes = [['shape1'], ['shape2'], ['shape1','shape2']]
    color_intervenes = [['color1'], ['color2'], ['color1','color2']]
    patch_color_intervenes = [['shape1','color1'], ['shape2','color2'], ['shape1','shape2','color1','color2']]
    patch_color_item_intervenes = [
        ['shape1','color1','item1'], ['shape2','color2','item2'], ['shape1','shape2','color1','color2','item1','item2']
    ]
    all_intervenes = [
          *no_intervene, *item_intervenes, *patch_intervenes, *color_intervenes, *patch_color_intervenes, *patch_color_item_intervenes
    ]

    flipped = [['shape1','color2'],['shape2','color1'], ['shape1','color2','item2'],['shape2','color1','item1']]

    for intervention in flipped:
        print(intervention)
        save_path = save_path_temp.format(dataset_name, '_'.join(intervention))
        print(save_path)
        result = intervene(model=model, vocab=vocab, num_samples=num_samples, num_layers=num_layers, 
                       device=device,  intervene_list=intervention, batch_size=8, num_image_patches=1024+2)
        torch.save(result, save_path)


run_factorizability_gemma()





