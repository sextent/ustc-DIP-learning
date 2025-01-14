import os
import os.path as osp
from argparse import ArgumentParser
from functools import partial

import gradio as gr
import numpy as np
import torch
from PIL import Image
import face_alignment

import dnnlib
from gradio_utils import (ImageMask, draw_mask_on_image, draw_points_on_image,
                          get_latest_points_pair, get_valid_mask,
                          on_change_single_global_state)
from viz.renderer import Renderer, add_watermark_np

parser = ArgumentParser()
parser.add_argument('--share', action='store_true',default='True')
parser.add_argument('--cache-dir', type=str, default='./checkpoints')
parser.add_argument(
    "--listen",
    action="store_true",
    help="launch gradio with 0.0.0.0 as server name, allowing to respond to network requests",
)
args = parser.parse_args()

cache_dir = args.cache_dir

device = 'cuda'


def reverse_point_pairs(points):
    new_points = []
    for p in points:
        new_points.append([p[1], p[0]])
    return new_points


def clear_state(global_state, target=None):
    """Clear target history state from global_state
    If target is not defined, points and mask will be both removed.
    1. set global_state['points'] as empty dict
    2. set global_state['mask'] as full-one mask.
    """
    if target is None:
        target = ['point', 'mask']
    if not isinstance(target, list):
        target = [target]
    if 'point' in target:
        global_state['points'] = dict()
        print('Clear Points State!')
    if 'mask' in target:
        image_raw = global_state["images"]["image_raw"]
        global_state['mask'] = np.ones((image_raw.size[1], image_raw.size[0]),
                                       dtype=np.uint8)
        print('Clear mask State!')

    return global_state


def init_images(global_state, custom_image=None, reset_generator=True):
    """This function is called only ones with Gradio App is started.
    0. pre-process global_state, unpack value from global_state of need
    1. Re-init renderer
    2. run `renderer._render_drag_impl` with `is_drag=False` to generate
       new image
    3. Assign images to global state and re-generate mask
    """

    if isinstance(global_state, gr.State):
        state = global_state.value
    else:
        state = global_state
        
    if reset_generator:
        state['renderer'].init_network(
            state['generator_params'],  # res
            valid_checkpoints_dict[state['pretrained_weight']],  # pkl
            state['params']['seed'],  # w0_seed,
            None,  # w_load
            state['params']['latent_space'] == 'w+',  # w_plus
            'const',
            state['params']['trunc_psi'],  # trunc_psi,
            state['params']['trunc_cutoff'],  # trunc_cutoff,
            None,  # input_transform
            state['params']['lr']  # lr,
        )

    if custom_image is None:
        # 使用生成器生成图片
        state['renderer']._render_drag_impl(state['generator_params'],
                                          is_drag=False,
                                          to_pil=True)
        init_image = state['generator_params'].image
    else:
        # 使用上传的图片
        init_image = custom_image.resize((512, 512)) if custom_image is not None else state['images']['image_raw']

    state['images']['image_orig'] = init_image
    state['images']['image_raw'] = init_image
    state['images']['image_show'] = Image.fromarray(
        add_watermark_np(np.array(init_image)))
    state['mask'] = np.ones((init_image.size[1], init_image.size[0]),
                            dtype=np.uint8)
    return global_state

# 添加处理上传图片的函数
def on_upload_image(image, global_state):
    """处理上传图片的函数
    1. 使用上传的图片初始化
    2. 清除所有状态
    """
    if image is None:
        return global_state, global_state['images']['image_show']
    
    # 设置标志表示使用的是上传的图片    
    global_state['is_uploaded'] = True
    init_images(global_state, image, reset_generator=False)
    clear_state(global_state)
    
    return global_state, global_state['images']['image_show']

def update_image_draw(image, points, mask, show_mask, global_state=None):

    image_draw = draw_points_on_image(image, points)
    if show_mask and mask is not None and not (mask == 0).all() and not (
            mask == 1).all():
        image_draw = draw_mask_on_image(image_draw, mask)

    image_draw = Image.fromarray(add_watermark_np(np.array(image_draw)))
    if global_state is not None:
        global_state['images']['image_show'] = image_draw
    return image_draw


def preprocess_mask_info(global_state, image):
    """Function to handle mask information.
    1. last_mask is None: Do not need to change mask, return mask
    2. last_mask is not None:
        2.1 global_state is remove_mask:
        2.2 global_state is add_mask:
    """
    if isinstance(image, dict):
        last_mask = get_valid_mask(image['mask'])
    else:
        last_mask = None
    mask = global_state['mask']

    # mask in global state is a placeholder with all 1.
    if (mask == 1).all():
        mask = last_mask

    # last_mask = global_state['last_mask']
    editing_mode = global_state['editing_state']

    if last_mask is None:
        return global_state

    if editing_mode == 'remove_mask':
        updated_mask = np.clip(mask - last_mask, 0, 1)
        print(f'Last editing_state is {editing_mode}, do remove.')
    elif editing_mode == 'add_mask':
        updated_mask = np.clip(mask + last_mask, 0, 1)
        print(f'Last editing_state is {editing_mode}, do add.')
    else:
        updated_mask = mask
        print(f'Last editing_state is {editing_mode}, '
              'do nothing to mask.')

    global_state['mask'] = updated_mask
    # global_state['last_mask'] = None  # clear buffer
    return global_state


valid_checkpoints_dict = {
    f.split('/')[-1].split('.')[0]: osp.join(cache_dir, f)
    for f in os.listdir(cache_dir)
    if (f.endswith('pkl') and osp.exists(osp.join(cache_dir, f)))
}
print(f'File under cache_dir ({cache_dir}):')
print(os.listdir(cache_dir))
print('Valid checkpoint file:')
print(valid_checkpoints_dict)

init_pkl = 'stylegan2_lions_512_pytorch'

with gr.Blocks() as app:

    # renderer = Renderer()
    global_state = gr.State({
        "images": {
            # image_orig: the original image, change with seed/model is changed
            # image_raw: image with mask and points, change durning optimization
            # image_show: image showed on screen
        },
        "temporal_params": {
            # stop
        },
        'mask':
        None,  # mask for visualization, 1 for editing and 0 for unchange
        'last_mask': None,  # last edited mask
        'show_mask': True,  # add button
        "generator_params": dnnlib.EasyDict(),
        "params": {
            "seed": 0,
            "motion_lambda": 20,
            "r1_in_pixels": 3,
            "r2_in_pixels": 12,
            "magnitude_direction_in_pixels": 1.0,
            "latent_space": "w+",
            "trunc_psi": 0.7,
            "trunc_cutoff": None,
            "lr": 0.001,
        },
        "device": device,
        "draw_interval": 1,
        "renderer": Renderer(disable_timing=True),
        "points": {},
        "curr_point": None,
        "curr_type_point": "start",
        'editing_state': 'add_points',
        'pretrained_weight': init_pkl
    })

    # init image
    global_state = init_images(global_state)

    with gr.Row():

        with gr.Row():

            # Left --> tools
            with gr.Column(scale=3):

                # Pickle
                with gr.Row():

                    with gr.Column(scale=1, min_width=10):
                        gr.Markdown(value='Pickle', show_label=False)

                    with gr.Column(scale=4, min_width=10):
                        form_pretrained_dropdown = gr.Dropdown(
                            choices=list(valid_checkpoints_dict.keys()),
                            label="Pretrained Model",
                            value=init_pkl,
                        )

                # Latent
                with gr.Row():
                    with gr.Column(scale=1, min_width=10):
                        gr.Markdown(value='Latent', show_label=False)

                    with gr.Column(scale=4, min_width=10):
                        form_seed_number = gr.Number(
                            value=global_state.value['params']['seed'],
                            interactive=True,
                            label="Seed",
                        )
                        form_lr_number = gr.Number(
                            value=global_state.value["params"]["lr"],
                            interactive=True,
                            label="Step Size")
                        # 添加上传按钮
                        form_upload_image = gr.Image(
                            label="Upload Image",
                            type="pil",
                            visible=True
                        )

                        with gr.Row():
                            with gr.Column(scale=2, min_width=10):
                                form_reset_image = gr.Button("Reset Image")
                            with gr.Column(scale=3, min_width=10):
                                form_latent_space = gr.Radio(
                                    ['w', 'w+'],
                                    value=global_state.value['params']
                                    ['latent_space'],
                                    interactive=True,
                                    label='Latent space to optimize',
                                    show_label=False,
                                )
                    
                    

                # Drag
                with gr.Row():
                    with gr.Column(scale=1, min_width=10):
                        gr.Markdown(value='Drag', show_label=False)
                    with gr.Column(scale=4, min_width=10):
                        with gr.Row():
                            with gr.Column(scale=1, min_width=10):
                                enable_add_points = gr.Button('Add Points')
                            with gr.Column(scale=1, min_width=10):
                                undo_points = gr.Button('Reset Points')
                            with gr.Column(scale=1, min_width=10):
                                slim_face_btn = gr.Button('一键瘦脸')
                            with gr.Column(scale=1, min_width=10):
                                big_eyes_btn = gr.Button('一键大眼')
                            with gr.Column(scale=1, min_width=10):
                                smile_btn = gr.Button('一键微笑')
                        with gr.Row():
                            with gr.Column(scale=1, min_width=10):
                                form_start_btn = gr.Button("Start")
                            with gr.Column(scale=1, min_width=10):
                                form_stop_btn = gr.Button("Stop")

                        form_steps_number = gr.Number(value=0,
                                                      label="Steps",
                                                      interactive=False)

                # Mask
                with gr.Row():
                    with gr.Column(scale=1, min_width=10):
                        gr.Markdown(value='Mask', show_label=False)
                    with gr.Column(scale=4, min_width=10):
                        enable_add_mask = gr.Button('Edit Flexible Area')
                        with gr.Row():
                            with gr.Column(scale=1, min_width=10):
                                form_reset_mask_btn = gr.Button("Reset mask")
                            with gr.Column(scale=1, min_width=10):
                                show_mask = gr.Checkbox(
                                    label='Show Mask',
                                    value=global_state.value['show_mask'],
                                    show_label=False)

                        with gr.Row():
                            form_lambda_number = gr.Number(
                                value=global_state.value["params"]
                                ["motion_lambda"],
                                interactive=True,
                                label="Lambda",
                            )

                form_draw_interval_number = gr.Number(
                    value=global_state.value["draw_interval"],
                    label="Draw Interval (steps)",
                    interactive=True,
                    visible=False)

            # Right --> Image
            with gr.Column(scale=8):
                form_image = ImageMask(
                    value=global_state.value['images']['image_show'],
                    brush_radius=20).style(
                        width=768,
                        height=768)  # NOTE: hard image size code here.
    gr.Markdown("""
        ## Quick Start

        1. Select desired `Pretrained Model` and adjust `Seed` to generate an
           initial image.
        2. Click on image to add control points.
        3. Click `Start` and enjoy it!

        ## Advance Usage

        1. Change `Step Size` to adjust learning rate in drag optimization.
        2. Select `w` or `w+` to change latent space to optimize:
        * Optimize on `w` space may cause greater influence to the image.
        * Optimize on `w+` space may work slower than `w`, but usually achieve
          better results.
        * Note that changing the latent space will reset the image, points and
          mask (this has the same effect as `Reset Image` button).
        3. Click `Edit Flexible Area` to create a mask and constrain the
           unmasked region to remain unchanged.
        """)
    gr.HTML("""
        <style>
            .container {
                position: absolute;
                height: 50px;
                text-align: center;
                line-height: 50px;
                width: 100%;
            }
        </style>
        <div class="container">
        Gradio demo supported by
        <img src="https://avatars.githubusercontent.com/u/10245193?s=200&v=4" height="20" width="20" style="display:inline;">
        <a href="https://github.com/open-mmlab/mmagic">OpenMMLab MMagic</a>
        </div>
        """)

    # Network & latents tab listeners
    def on_change_pretrained_dropdown(pretrained_value, global_state):
        """Function to handle model change.
        1. Set pretrained value to global_state
        2. Re-init images and clear all states
        """

        global_state['pretrained_weight'] = pretrained_value
        global_state['is_uploaded'] = False  # 重置标志
        init_images(global_state)
        clear_state(global_state)

        return global_state, global_state["images"]['image_show']

    form_pretrained_dropdown.change(
        on_change_pretrained_dropdown,
        inputs=[form_pretrained_dropdown, global_state],
        outputs=[global_state, form_image],
    )

    def on_click_reset_image(global_state):
        """Reset image to the original one and clear all states
        1. Re-init images
        2. Clear all states
        """

        init_images(global_state)
        clear_state(global_state)

        return global_state, global_state['images']['image_show']

    form_reset_image.click(
        on_click_reset_image,
        inputs=[global_state],
        outputs=[global_state, form_image],
    )

    # Update parameters
    def on_change_update_image_seed(seed, global_state):
        """Function to handle generation seed change.
        1. Set seed to global_state
        2. Re-init images and clear all states
        """

        global_state["params"]["seed"] = int(seed)
        global_state['is_uploaded'] = False  # 重置标志
        init_images(global_state)
        clear_state(global_state)

        return global_state, global_state['images']['image_show']

    form_seed_number.change(
        on_change_update_image_seed,
        inputs=[form_seed_number, global_state],
        outputs=[global_state, form_image],
    )

    def on_click_latent_space(latent_space, global_state):
        """Function to reset latent space to optimize.
        NOTE: this function we reset the image and all controls
        1. Set latent-space to global_state
        2. Re-init images and clear all state
        """

        global_state['params']['latent_space'] = latent_space
        init_images(global_state)
        clear_state(global_state)

        return global_state, global_state['images']['image_show']

    form_latent_space.change(on_click_latent_space,
                             inputs=[form_latent_space, global_state],
                             outputs=[global_state, form_image])

    # ==== Params
    form_lambda_number.change(
        partial(on_change_single_global_state, ["params", "motion_lambda"]),
        inputs=[form_lambda_number, global_state],
        outputs=[global_state],
    )
    # 添加上传图片的事件处理
    form_upload_image.change(
        on_upload_image,
        inputs=[form_upload_image, global_state],
        outputs=[global_state, form_image]
    )


    def on_change_lr(lr, global_state):
        if lr == 0:
            print('lr is 0, do nothing.')
            return global_state
        else:
            global_state["params"]["lr"] = lr
            renderer = global_state['renderer']
            renderer.update_lr(lr)
            print('New optimizer: ')
            print(renderer.w_optim)
        return global_state

    form_lr_number.change(
        on_change_lr,
        inputs=[form_lr_number, global_state],
        outputs=[global_state],
    )

    def on_click_start(global_state, image):
        p_in_pixels = []
        t_in_pixels = []
        valid_points = []

        # handle of start drag in mask editing mode
        global_state = preprocess_mask_info(global_state, image)

        # Prepare the points for the inference
        if len(global_state["points"]) == 0:
            image_raw = global_state['images']['image_raw']
            update_image_draw(
                image_raw,
                global_state['points'],
                global_state['mask'],
                global_state['show_mask'],
                global_state,
            )

            yield (
                global_state,
                0,
                global_state['images']['image_show'],
                # gr.File.update(visible=False),
                gr.Button.update(interactive=True),
                gr.Button.update(interactive=True),
                gr.Button.update(interactive=True),
                gr.Button.update(interactive=True),
                gr.Button.update(interactive=True),
                # latent space
                gr.Radio.update(interactive=True),
                gr.Button.update(interactive=True),
                # NOTE: disable stop button
                gr.Button.update(interactive=False),

                # update other comps
                gr.Dropdown.update(interactive=True),
                gr.Number.update(interactive=True),
                gr.Number.update(interactive=True),
                gr.Button.update(interactive=True),
                gr.Button.update(interactive=True),
                gr.Checkbox.update(interactive=True),
                # gr.Number.update(interactive=True),
                gr.Number.update(interactive=True),
            )
        else:

            # Transform the points into torch tensors
            for key_point, point in global_state["points"].items():
                try:
                    p_start = point.get("start_temp", point["start"])
                    p_end = point["target"]

                    if p_start is None or p_end is None:
                        continue

                except KeyError:
                    continue

                p_in_pixels.append(p_start)
                t_in_pixels.append(p_end)
                valid_points.append(key_point)

            mask = torch.tensor(global_state['mask']).float()
            drag_mask = 1 - mask

            renderer: Renderer = global_state["renderer"]
            global_state['temporal_params']['stop'] = False
            global_state['editing_state'] = 'running'

            # reverse points order
            p_to_opt = reverse_point_pairs(p_in_pixels)
            t_to_opt = reverse_point_pairs(t_in_pixels)
            print('Running with:')
            print(f'    Source: {p_in_pixels}')
            print(f'    Target: {t_in_pixels}')
            step_idx = 0
            while True:
                if global_state["temporal_params"]["stop"]:
                    break

                # 修改这里: 如果是上传的图片,跳过生成器的渲染
                if 'is_uploaded' in global_state and global_state['is_uploaded']:
                    # 使用上传的图片
                    image_result = global_state['images']['image_raw']
                else:
                    # 使用生成器生成的图片
                    renderer._render_drag_impl(
                        global_state['generator_params'],
                        p_to_opt,  # point
                        t_to_opt,  # target
                        drag_mask,  # mask,
                        global_state['params']['motion_lambda'],  # lambda_mask
                        reg=0,
                        feature_idx=5,  # NOTE: do not support change for now
                        r1=global_state['params']['r1_in_pixels'],  # r1
                        r2=global_state['params']['r2_in_pixels'],  # r2
                        trunc_psi=global_state['params']['trunc_psi'],
                        is_drag=True,
                        to_pil=True)
                    image_result = global_state['generator_params']['image']

                if step_idx % global_state['draw_interval'] == 0:
                    print('Current Source:')
                    for key_point, p_i, t_i in zip(valid_points, p_to_opt,
                                                   t_to_opt):
                        global_state["points"][key_point]["start_temp"] = [
                            p_i[1],
                            p_i[0],
                        ]
                        global_state["points"][key_point]["target"] = [
                            t_i[1],
                            t_i[0],
                        ]
                        start_temp = global_state["points"][key_point][
                            "start_temp"]
                        print(f'    {start_temp}')

                    image_result = global_state['generator_params']['image']
                    image_draw = update_image_draw(
                        image_result,
                        global_state['points'],
                        global_state['mask'],
                        global_state['show_mask'],
                        global_state,
                    )
                    global_state['images']['image_raw'] = image_result

                yield (
                    global_state,
                    step_idx,
                    global_state['images']['image_show'],
                    # gr.File.update(visible=False),
                    gr.Button.update(interactive=False),
                    gr.Button.update(interactive=False),
                    gr.Button.update(interactive=False),
                    gr.Button.update(interactive=False),
                    gr.Button.update(interactive=False),
                    # latent space
                    gr.Radio.update(interactive=False),
                    gr.Button.update(interactive=False),
                    # enable stop button in loop
                    gr.Button.update(interactive=True),

                    # update other comps
                    gr.Dropdown.update(interactive=False),
                    gr.Number.update(interactive=False),
                    gr.Number.update(interactive=False),
                    gr.Button.update(interactive=False),
                    gr.Button.update(interactive=False),
                    gr.Checkbox.update(interactive=False),
                    # gr.Number.update(interactive=False),
                    gr.Number.update(interactive=False),
                )

                # increate step
                step_idx += 1

            # 修改这里: 同样根据是否是上传图片来决定使用哪个图像
            if 'is_uploaded' in global_state and global_state['is_uploaded']:
                image_result = global_state['images']['image_raw']
            else:
                image_result = global_state['generator_params']['image']
            
            global_state['images']['image_raw'] = image_result
            image_draw = update_image_draw(image_result,
                                           global_state['points'],
                                           global_state['mask'],
                                           global_state['show_mask'],
                                           global_state)

            # fp = NamedTemporaryFile(suffix=".png", delete=False)
            # image_result.save(fp, "PNG")

            global_state['editing_state'] = 'add_points'

            yield (
                global_state,
                0,  # reset step to 0 after stop.
                global_state['images']['image_show'],
                # gr.File.update(visible=True, value=fp.name),
                gr.Button.update(interactive=True),
                gr.Button.update(interactive=True),
                gr.Button.update(interactive=True),
                gr.Button.update(interactive=True),
                gr.Button.update(interactive=True),
                # latent space
                gr.Radio.update(interactive=True),
                gr.Button.update(interactive=True),
                # NOTE: disable stop button with loop finish
                gr.Button.update(interactive=False),

                # update other comps
                gr.Dropdown.update(interactive=True),
                gr.Number.update(interactive=True),
                gr.Number.update(interactive=True),
                gr.Checkbox.update(interactive=True),
                gr.Number.update(interactive=True),
            )

    form_start_btn.click(
        on_click_start,
        inputs=[global_state, form_image],
        outputs=[
            global_state,
            form_steps_number,
            form_image,
            # form_download_result_file,
            # >>> buttons
            form_reset_image,
            enable_add_points,
            enable_add_mask,
            undo_points,
            form_reset_mask_btn,
            form_latent_space,
            form_start_btn,
            form_stop_btn,
            # <<< buttonm
            # >>> inputs comps
            form_pretrained_dropdown,
            form_seed_number,
            form_lr_number,
            show_mask,
            form_lambda_number,
        ],
    )

    def on_click_stop(global_state):
        """Function to handle stop button is clicked.
        1. send a stop signal by set global_state["temporal_params"]["stop"] as True
        2. Disable Stop button
        """
        global_state["temporal_params"]["stop"] = True

        return global_state, gr.Button.update(interactive=False)

    form_stop_btn.click(on_click_stop,
                        inputs=[global_state],
                        outputs=[global_state, form_stop_btn])

    form_draw_interval_number.change(
        partial(
            on_change_single_global_state,
            "draw_interval",
            map_transform=lambda x: int(x),
        ),
        inputs=[form_draw_interval_number, global_state],
        outputs=[global_state],
    )

    def on_click_remove_point(global_state):
        choice = global_state["curr_point"]
        del global_state["points"][choice]

        choices = list(global_state["points"].keys())

        if len(choices) > 0:
            global_state["curr_point"] = choices[0]

        return (
            gr.Dropdown.update(choices=choices, value=choices[0]),
            global_state,
        )

    # Mask
    def on_click_reset_mask(global_state):
        global_state['mask'] = np.ones(
            (
                global_state["images"]["image_raw"].size[1],
                global_state["images"]["image_raw"].size[0],
            ),
            dtype=np.uint8,
        )
        image_draw = update_image_draw(global_state['images']['image_raw'],
                                       global_state['points'],
                                       global_state['mask'],
                                       global_state['show_mask'], global_state)
        return global_state, image_draw

    form_reset_mask_btn.click(
        on_click_reset_mask,
        inputs=[global_state],
        outputs=[global_state, form_image],
    )

    # Image
    def on_click_enable_draw(global_state, image):
        """Function to start add mask mode.
        1. Preprocess mask info from last state
        2. Change editing state to add_mask
        3. Set curr image with points and mask
        """
        global_state = preprocess_mask_info(global_state, image)
        global_state['editing_state'] = 'add_mask'
        image_raw = global_state['images']['image_raw']
        image_draw = update_image_draw(image_raw, global_state['points'],
                                       global_state['mask'], True,
                                       global_state)
        return (global_state,
                gr.Image.update(value=image_draw, interactive=True))

    def on_click_remove_draw(global_state, image):
        """Function to start remove mask mode.
        1. Preprocess mask info from last state
        2. Change editing state to remove_mask
        3. Set curr image with points and mask
        """
        global_state = preprocess_mask_info(global_state, image)
        global_state['edinting_state'] = 'remove_mask'
        image_raw = global_state['images']['image_raw']
        image_draw = update_image_draw(image_raw, global_state['points'],
                                       global_state['mask'], True,
                                       global_state)
        return (global_state,
                gr.Image.update(value=image_draw, interactive=True))

    enable_add_mask.click(on_click_enable_draw,
                          inputs=[global_state, form_image],
                          outputs=[
                              global_state,
                              form_image,
                          ])

    def on_click_add_point(global_state, image: dict):
        """Function switch from add mask mode to add points mode.
        1. Updaste mask buffer if need
        2. Change global_state['editing_state'] to 'add_points'
        3. Set current image with mask
        """

        global_state = preprocess_mask_info(global_state, image)
        global_state['editing_state'] = 'add_points'
        mask = global_state['mask']
        image_raw = global_state['images']['image_raw']
        image_draw = update_image_draw(image_raw, global_state['points'], mask,
                                       global_state['show_mask'], global_state)

        return (global_state,
                gr.Image.update(value=image_draw, interactive=False))

    enable_add_points.click(on_click_add_point,
                            inputs=[global_state, form_image],
                            outputs=[global_state, form_image])

    def on_click_image(global_state, evt: gr.SelectData):
        """This function only support click for point selection
        """
        xy = evt.index
        if global_state['editing_state'] != 'add_points':
            print(f'In {global_state["editing_state"]} state. '
                  'Do not add points.')

            return global_state, global_state['images']['image_show']

        points = global_state["points"]

        point_idx = get_latest_points_pair(points)
        if point_idx is None:
            points[0] = {'start': xy, 'target': None}
            print(f'Click Image - Start - {xy}')
        elif points[point_idx].get('target', None) is None:
            points[point_idx]['target'] = xy
            print(f'Click Image - Target - {xy}')
        else:
            points[point_idx + 1] = {'start': xy, 'target': None}
            print(f'Click Image - Start - {xy}')

        image_raw = global_state['images']['image_raw']
        image_draw = update_image_draw(
            image_raw,
            global_state['points'],
            global_state['mask'],
            global_state['show_mask'],
            global_state,
        )

        return global_state, image_draw

    form_image.select(
        on_click_image,
        inputs=[global_state],
        outputs=[global_state, form_image],
    )

    def on_click_clear_points(global_state):
        """Function to handle clear all control points
        1. clear global_state['points'] (clear_state)
        2. re-init network
        2. re-draw image
        """
        clear_state(global_state, target='point')

        renderer: Renderer = global_state["renderer"]
        renderer.feat_refs = None

        image_raw = global_state['images']['image_raw']
        image_draw = update_image_draw(image_raw, {}, global_state['mask'],
                                       global_state['show_mask'], global_state)
        return global_state, image_draw

    undo_points.click(on_click_clear_points,
                      inputs=[global_state],
                      outputs=[global_state, form_image])

    def on_click_show_mask(global_state, show_mask):
        """Function to control whether show mask on image."""
        global_state['show_mask'] = show_mask

        image_raw = global_state['images']['image_raw']
        image_draw = update_image_draw(
            image_raw,
            global_state['points'],
            global_state['mask'],
            global_state['show_mask'],
            global_state,
        )
        return global_state, image_draw

    show_mask.change(
        on_click_show_mask,
        inputs=[global_state, show_mask],
        outputs=[global_state, form_image],
    )

    def get_face_landmarks(image):
        """获取人脸关键点"""
        fa = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, flip_input=False, device='cuda')
        if isinstance(image, Image.Image):
            image = np.array(image)
        landmarks = fa.get_landmarks(image)
        if landmarks is None or len(landmarks) == 0:
            return None
        return landmarks[0]  # 返回第一个检测到的人脸关键点

    def setup_face_slimming_points(landmarks):
        """设置瘦脸控制点"""
        # 获取脸部轮廓关键点
        jaw_points = landmarks[0:17]  # 下巴到额头的轮廓点
        
        # 选择左右脸颊的控制点
        left_face_idx = 2  # 左脸位置
        right_face_idx = 14  # 右脸位置
        
        # 设置控制点和目标点
        points = {}
        
        # 左脸控制点
        target_left = jaw_points[left_face_idx]
        start_left = target_left + np.array([20, 0])  
        # 确保坐标为整数
        points[0] = {
            'target': [int(start_left[0]), int(start_left[1])],
            'start': [int(target_left[0]), int(target_left[1])]
        }
        
        # 右脸控制点  
        start_right = jaw_points[right_face_idx]
        target_right = start_right + np.array([-20, 0])
        # 确保坐标为整数
        points[1] = {
            'start': [int(start_right[0]), int(start_right[1])],
            'target': [int(target_right[0]), int(target_right[1])]
        }
        
        return points

    def on_click_slim_face(global_state):
        """一键瘦脸功能"""
        # 获取当前图像
        image = global_state['images']['image_raw']
        
        # 获取人脸关键点
        landmarks = get_face_landmarks(image)
        if landmarks is None:
            raise gr.Error('未检测到人脸!')
            
        # 设置瘦脸控制点
        points = setup_face_slimming_points(landmarks)
        global_state['points'] = points
        
        # 更新图像显示
        image_draw = update_image_draw(
            image,
            global_state['points'],
            global_state['mask'],
            global_state['show_mask'],
            global_state,
        )
        
        return global_state, image_draw

    # 绑定瘦脸按钮事件
    slim_face_btn.click(
        on_click_slim_face,
        inputs=[global_state],
        outputs=[global_state, form_image]
    )

    # 添加设置大眼控制点的函数
    def setup_big_eyes_points(landmarks):
        """设置大眼控制点"""
        points = {}
        
        # 获取左眼关键点 (36-41)
        left_eye = landmarks[36:42]
        # 获取右眼关键点 (42-47) 
        right_eye = landmarks[42:48]
        
        # 左眼上下点
        left_eye_top = left_eye[1]  # 37号点
        left_eye_bottom = left_eye[4]  # 40号点
        
        # 右眼上下点
        right_eye_top = right_eye[1]  # 43号点
        right_eye_bottom = right_eye[4]  # 46号点
        
        # 设置左眼控制点
        points[0] = {
            'start': [int(left_eye_top[0]), int(left_eye_top[1])],
            'target': [int(left_eye_top[0]), int(left_eye_top[1] - 10)]  # 向上移动
        }
        points[1] = {
            'start': [int(left_eye_bottom[0]), int(left_eye_bottom[1])],
            'target': [int(left_eye_bottom[0]), int(left_eye_bottom[1] + 10)]  # 向下移动
        }
        
        # 设置右眼控制点
        points[2] = {
            'start': [int(right_eye_top[0]), int(right_eye_top[1])],
            'target': [int(right_eye_top[0]), int(right_eye_top[1] - 10)]  # 向上移动
        }
        points[3] = {
            'start': [int(right_eye_bottom[0]), int(right_eye_bottom[1])],
            'target': [int(right_eye_bottom[0]), int(right_eye_bottom[1] + 10)]  # 向下移动
        }
        
        return points

    # 添加设置微笑控制点的函数
    def setup_smile_points(landmarks):
        """设置微笑控制点"""
        points = {}
        
        # 获取嘴角关键点
        mouth_left = landmarks[48]  # 左嘴角
        mouth_right = landmarks[54]  # 右嘴角
        
        # 设置左嘴角控制点
        points[0] = {
            'start': [int(mouth_left[0]), int(mouth_left[1])],
            'target': [int(mouth_left[0] - 10), int(mouth_left[1] - 10)]  # 向左上移动
        }
        
        # 设置右嘴角控制点
        points[1] = {
            'start': [int(mouth_right[0]), int(mouth_right[1])],
            'target': [int(mouth_right[0] + 10), int(mouth_right[1] - 10)]  # 向右上移动
        }
        
        return points

    # 添加一键大眼处理函数
    def on_click_big_eyes(global_state):
        """一键大���功能"""
        image = global_state['images']['image_raw']
        
        landmarks = get_face_landmarks(image)
        if landmarks is None:
            raise gr.Error('未检测到人脸!')
            
        points = setup_big_eyes_points(landmarks)
        global_state['points'] = points
        
        image_draw = update_image_draw(
            image,
            global_state['points'],
            global_state['mask'],
            global_state['show_mask'],
            global_state,
        )
        
        return global_state, image_draw

    # 添加一键微笑处理函数
    def on_click_smile(global_state):
        """一键微笑功能"""
        image = global_state['images']['image_raw']
        
        landmarks = get_face_landmarks(image)
        if landmarks is None:
            raise gr.Error('未检测到人脸!')
            
        points = setup_smile_points(landmarks)
        global_state['points'] = points
        
        image_draw = update_image_draw(
            image,
            global_state['points'],
            global_state['mask'],
            global_state['show_mask'],
            global_state,
        )
        
        return global_state, image_draw

    # 绑定新按钮事件
    big_eyes_btn.click(
        on_click_big_eyes,
        inputs=[global_state],
        outputs=[global_state, form_image]
    )

    smile_btn.click(
        on_click_smile,
        inputs=[global_state],
        outputs=[global_state, form_image]
    )


gr.close_all()
app.queue(concurrency_count=3, max_size=20)
app.launch(share=args.share, server_name="0.0.0.0" if args.listen else "127.0.0.1")
