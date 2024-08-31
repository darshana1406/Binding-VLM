import pandas as pd
from string import Formatter
import bisect
import ast
from os.path import join

from utils.helper import Substring, recursify


class ShapesItems():

    def __init__(self, csv_path, img_path, flip=False, order_flip=False, mapping=True) -> None:

        self.flip = flip
        self.order_flip = order_flip
        self.mapping = mapping
        self.data = pd.read_csv(csv_path)
        self.data.image1 = self.data.image1.apply(ast.literal_eval)
        self.data.image2 = self.data.image2.apply(ast.literal_eval)
        self.data.img1_items = self.data.img1_items.apply(ast.literal_eval)
        self.data.img2_items = self.data.img2_items.apply(ast.literal_eval)
        self.rows = [row for i, row in self.data.iterrows()]

        self.images_path = img_path

        self.context_template = 'The {color} object contains item {letter}.'
        # self.context_template_bad = 'The small boy pocesses item {letter}.'
        # self.context_template_bad = 'item {letter}.'
        self.context_template_bad = 'item {letter} is contained in {color} object.'

        self.template = """USER: <image>\nAnswer the question based on the provided image and the context below. Keep the answer short.

Context:{context}

Question: Which item does the {qn_shape} contain? ASSISTANT: The {ans_shape} contains item """

        self.image_offset = 5
        # [79, 80, 81, 103, 104, 105, 127, 128, 129] 1 index
        shape1_patches = [Substring(*tup)+self.image_offset for tup in [(78,81),(102,105),(126,129)]]
        # [163, 164, 165, 187, 188, 189, 211, 212, 213]
        shape2_patches = [Substring(*tup)+self.image_offset for tup in [(162,165),(186,189),(210,213)]]
        # obj2_patches_bad = [Substring(*tup)+(-4)+self.image_offset for tup in [(162,165),(186,189),(210,213)]]
        self.image_patches = dict(
            shape1 = shape1_patches,
            shape2 = shape2_patches
    )
        if self.flip:
            self.image_patches = dict(
            shape1 = shape2_patches,
            shape2 = shape1_patches
    )
            

    def set_details(self, context_template, template, image_offset, shape1_patches, shape2_patches, flip):
        self.context_template = context_template
        self.template = template
        self.image_offset = image_offset
        self.flip = flip
        if self.flip:
            self.image_patches = dict(
            shape1 = shape2_patches,
            shape2 = shape1_patches)
        else:
           self.image_patches = dict(
            shape1 = shape1_patches,
            shape2 = shape2_patches
        ) 
    

        
    def recursify(func, dtype=Substring, pred=None):
        if pred is None:
            pred = lambda x: isinstance(x, Substring) or isinstance(x, int)

        def wrapper(self, indices, *args, **kwargs):
            if pred(indices):
                return func(self, indices, *args, **kwargs)
            elif isinstance(indices, dict):
                return {
                    key: wrapper(self, value, *args, **kwargs) for key, value in indices.items()
                }
            elif isinstance(indices, list):
                return [wrapper(self, value, *args, **kwargs) for value in indices]
            else:
                raise Exception(f"Unexpected type {type(indices)}")

        return wrapper


    @recursify
    def get_sample_dicts(self, index):

        if self.mapping:
            c_map = lambda c: 'bright '+c if not (c in ['purple','cyan']) else c
            s_map = lambda s: 'can' if s == 'cylinder' else s
        else:
            c_map = lambda c: c
            s_map = lambda s: s

        row = self.rows[index]
        context_s1_image = dict(
            color1 = c_map(row['image1'][0]),shape1 = s_map(row['image1'][1]),
            color2 = c_map(row['image1'][2]),shape2 = s_map(row['image1'][3])
        )
        context_s1_items = dict(item1 = row['img1_items'][0],item2 = row['img1_items'][1])
        context_s2_image = dict(
            color1 = c_map(row['image2'][0]),shape1 = s_map(row['image2'][1]),
            color2 = c_map(row['image2'][2]),shape2 = s_map(row['image2'][3])
        )
        context_s2_items = dict(item1 = row['img2_items'][0],item2 = row['img2_items'][1])

        context_s1 = dict(
            image_path = join(self.images_path,row['img1_path']),
            objects = context_s1_image,
            items = context_s1_items
        )
        context_s2 = dict(

            image_path = join(self.images_path,row['img2_path']),
            objects = context_s2_image,
            items = context_s2_items
        )

        if self.flip:
            return context_s2, context_s1
        return context_s1, context_s2
    

    def template_format(self, template, format_dict, offset=0):
        ind_dict = {}
        cur_offset = 0
        for literal_text, field_name, format_spec, conversion in Formatter().parse(template):
            if field_name is not None:
                value = format_dict[field_name]
                start = len(literal_text) + cur_offset
                end = start + len(value)
                ind_dict[field_name] = Substring(start+offset,end+offset)
                cur_offset = end

        format_str = template.format(**format_dict)

        return format_str,ind_dict
    

    def get_rand_context(self, context, shape):
        context_p1, ind_dict_p1 = self.template_format(
            self.context_template_bad,
            dict(color=context['objects']['color1'], letter=context['items']['item1'])
        )
        # context_p2, ind_dict_p2 = self.template_format(
        #     self.context_template,
        #     dict(color=context['objects']['color2'], letter=context['items']['item2'])
        # )
        context_p2, ind_dict_p2 = self.template_format(
            self.context_template_bad,
            dict(color=context['objects']['color2'], letter=context['items']['item2'])
        )
        # ind_dict_p1['color']=Substring(-100,-100)
        # ind_dict_p2['color']=Substring(-100,-100)

        joined_context = ' '.join([context_p1, context_p2])
        for key in ind_dict_p2:
            ind_dict_p2[key] += len(context_p1) + 1

        text_prompt, text_ind_dict = self.template_format(
            self.template, 
            dict(context=joined_context, qn_shape=shape, ans_shape=shape)
            )

        for key in ind_dict_p1:
            ind_dict_p1[key] += text_ind_dict['context'][0]
        for key in ind_dict_p2:
            ind_dict_p2[key] += text_ind_dict['context'][0]

        full_ind_dict = dict(
            color1 = ind_dict_p1['color'],
            item1 = ind_dict_p1['letter'],
            color2 = ind_dict_p2['color'],
            item2 = ind_dict_p2['letter'],
            context = text_ind_dict['context'],
            qn_shape = text_ind_dict['qn_shape'],
            ans_shape = text_ind_dict['ans_shape']
        )

        return text_prompt, full_ind_dict


    def get_context(self, context, shape, order_flip=False):
        context_p1, ind_dict_p1 = self.template_format(
            self.context_template,
            dict(color=context['objects']['color1'], letter=context['items']['item1'])
        )
        context_p2, ind_dict_p2 = self.template_format(
            self.context_template,
            dict(color=context['objects']['color2'], letter=context['items']['item2'])
        )

        if order_flip:
            context_p1, context_p2 = context_p2, context_p1
            ind_dict_p1, ind_dict_p2 = ind_dict_p2, ind_dict_p1

        joined_context = ' '.join([context_p1, context_p2])
        for key in ind_dict_p2:
            ind_dict_p2[key] += len(context_p1) + 1

        text_prompt, text_ind_dict = self.template_format(
            self.template, 
            dict(context=joined_context, qn_shape=shape, ans_shape=shape)
            )

        for key in ind_dict_p1:
            ind_dict_p1[key] += text_ind_dict['context'][0]
        for key in ind_dict_p2:
            ind_dict_p2[key] += text_ind_dict['context'][0]

        full_ind_dict = dict(
            color1 = ind_dict_p1['color'],
            item1 = ind_dict_p1['letter'],
            color2 = ind_dict_p2['color'],
            item2 = ind_dict_p2['letter'],
            context = text_ind_dict['context'],
            qn_shape = text_ind_dict['qn_shape'],
            ans_shape = text_ind_dict['ans_shape']
        )

        return text_prompt, full_ind_dict

    @recursify
    def get_samples(self, index, shape_id):

        context_s1, context_s2 = self.get_sample_dicts(index)
        if shape_id == 0:
            shape = context_s1['objects']['shape1']
        elif shape_id == 1:
            shape = context_s1['objects']['shape2']
        elif shape_id == 2:
            shape = context_s2['objects']['shape1']
        else:
            shape = context_s2['objects']['shape2']

        s1_text, s1_index = self.get_context(context_s1, shape, order_flip=self.order_flip)
        s2_text, s2_index = self.get_context(context_s2, shape, order_flip=self.order_flip)
        # s2_text, s2_index = self.get_rand_context(context_s2, shape)


        return s1_text, s1_index, s2_text, s2_index
    

def align_token_indices(tokenizer, prompt, index, num_image_patches=576):
    tokenized = tokenizer(prompt, return_offsets_mapping=True)
    inputs, offset_mapping = tokenized['input_ids'], tokenized['offset_mapping']

    @recursify
    def align(pos, num_image_patches):
        start, end = pos
        start = bisect.bisect_right([x for x, _ in offset_mapping], start) - 1
        end = bisect.bisect_right([x for x, _ in offset_mapping], end-1) - 1
        start += num_image_patches-1
        end += num_image_patches-1
        return Substring(start, end+1)
    
    aligned_index = align(index, num_image_patches=num_image_patches)
    return inputs, aligned_index
