"""
Adapted from CogVideoX-5B: https://github.com/THUDM/CogVideo by Xiao Fu (CUHK)
"""

import argparse
from typing import Literal
import copy

import torch
from diffusers import (
    CogVideoXDDIMScheduler,
    CogVideoXDPMScheduler,
    CogVideoXImageToVideoPipeline,
    CogVideoXVideoToVideoPipeline,
)
import os
import json

import datetime
import sys
sys.path.append('../finetune')

from models.pipeline_cogvideox import CogVideoXPipeline
from models.cogvideox_transformer_3d import CogVideoXTransformer3DModel

from diffusers.utils import export_to_video, load_image, load_video
import json
import numpy as np
import random
from einops import rearrange

def parse_matrix(matrix_str):
    rows = matrix_str.strip().split('] [')
    matrix = []
    for row in rows:
        row = row.replace('[', '').replace(']', '')
        matrix.append(list(map(float, row.split())))
    return np.array(matrix)

def load_sceneposes(objs_file, obj_idx, obj_transl):
    ext_poses = []
    for i, key in enumerate(objs_file.keys()):
        ext_poses.append(parse_matrix(objs_file[key][obj_idx]['matrix']))
    ext_poses = np.stack(ext_poses)
    ext_poses = np.transpose(ext_poses, (0,2,1))
    ext_poses[:,:3,3] -= obj_transl
    ext_poses[:,:3,3] /= 100.
    ext_poses = ext_poses[:, :, [1,2,0,3]]
    return ext_poses

def get_pose_embeds(scene, video_name, instance_data_root, locations_info, cam_poses):

    with open(os.path.join(instance_data_root, "480_720", scene, video_name, video_name+'.json'), 'r') as f: objs_file = json.load(f)
    objs_num = len(objs_file['0'])
    video_index = 12

    location_name = video_name.split('_')[1]
    location_info = locations_info[location_name]
    cam_pose = cam_poses[video_index-1]
    obj_transl = location_info['coordinates']['CameraTarget']['position']

    obj_poses_list = []
    for obj_idx in range(objs_num):
        obj_poses = load_sceneposes(objs_file, obj_idx, obj_transl)
        obj_poses = np.linalg.inv(cam_pose) @ obj_poses
        obj_poses_list.append(obj_poses)

    obj_poses_all = torch.from_numpy(np.array(obj_poses_list))
    
    total_frames = 99
    sample_n_frames = 49
    current_sample_stride = 1.75
    start_frame_ind = 10

    cropped_length = int(sample_n_frames * current_sample_stride)
    end_frame_ind = min(start_frame_ind + cropped_length, total_frames)
    frame_indices = np.linspace(start_frame_ind, end_frame_ind - 1, sample_n_frames, dtype=int)

    # interpolation
    trunc_frame_indices = np.zeros_like(frame_indices[::4])
    trunc_frame_indices[0] = frame_indices[0]
    trunc_frame_indices[1:] = ((frame_indices[1:][::4] + frame_indices[4:][::4])/2).astype(np.int64)

    obj_poses_all = obj_poses_all[:, trunc_frame_indices]
    pose_embeds = rearrange(obj_poses_all[:, :, :3, :], "n f p q -> n f (q p)").contiguous().to(torch.bfloat16)

    return pose_embeds
    

def init_cam_poses(instance_data_root):
    cam_num = 12
    cams_path = os.path.join(instance_data_root, "Hemi12_transforms.json")
    with open(cams_path, 'r') as f: cams_info = json.load(f)
    cam_poses = []
    for i, key in enumerate(cams_info.keys()):
        if "C_" in key:
            cam_poses.append(parse_matrix(cams_info[key]))
    cam_poses = np.stack(cam_poses)
    cam_poses = np.transpose(cam_poses, (0,2,1))
    cam_poses = cam_poses[:,:,[1,2,0,3]]
    cam_poses[:,:3,3] /= 100.
    return cam_poses

def generate_video(
    model_path: str,
    ckpt_path: str,
    lora_path: str = None,
    lora_scale: float = 1.0,
    output_path: str = "./output.mp4",
    image_or_video_path: str = "",
    annealed_sample_step: int = 15,
    num_inference_steps: int = 50,
    guidance_scale: float = 6.0,
    num_videos_per_prompt: int = 1,
    dtype: torch.dtype = torch.bfloat16,
    generate_type: str = Literal["t2v", "i2v", "v2v"],  # i2v: image to video, v2v: video to video
    seed: int = 42,
):
    """
    Generates a video based on the given prompt and saves it to the specified path.

    Parameters:
    - prompt (str): The description of the video to be generated.
    - model_path (str): The path of the pre-trained model to be used.
    - lora_path (str): The path of the LoRA weights to be used.
    - lora_scale (float): 
    - output_path (str): The path where the generated video will be saved.
    - num_inference_steps (int): Number of steps for the inference process. More steps can result in better quality.
    - guidance_scale (float): The scale for classifier-free guidance. Higher values can lead to better alignment with the prompt.
    - num_videos_per_prompt (int): Number of videos to generate per prompt.
    - dtype (torch.dtype): The data type for computation (default is torch.bfloat16).
    - generate_type (str): The type of video generation (e.g., 't2v', 'i2v', 'v2v').·
    - seed (int): The seed for reproducibility.
    """
    
    # 1.  Load the pre-trained CogVideoX pipeline with the specified precision (bfloat16).
    # add device_map="balanced" in the from_pretrained function and remove the enable_model_cpu_offload()
    # function to use Multi GPUs.

    transformer = CogVideoXTransformer3DModel.from_pretrained(ckpt_path, torch_dtype=dtype)
    pipe = CogVideoXPipeline.from_pretrained(model_path, 
        transformer=transformer,
        torch_dtype=dtype
    )
    pipe.transformer_ori = copy.deepcopy(pipe.transformer).to("cuda")

    # If you're using with lora, add this code
    if lora_path:
        pipe.load_lora_weights(lora_path, weight_name="pytorch_lora_weights.safetensors", adapter_name="default")
        pipe.fuse_lora(components=['transformer'] ,lora_scale=lora_scale)
    
    # 2. Set Scheduler.
    pipe.scheduler = CogVideoXDPMScheduler.from_config(pipe.scheduler.config, timestep_spacing="trailing")

    # 3. Enable CPU offload for the model.
    pipe.to("cuda")
    pipe.enable_sequential_cpu_offload()
    pipe.vae.enable_slicing()
    pipe.vae.enable_tiling()
    
    # 4. Load object poses, scene and object prompts
    instance_data_root = "/m2v_intern/fuxiao/360Motion-Dataset"
    scene = "Desert"
    locations_path = os.path.join(instance_data_root, "480_720", scene, "location_data.json")
    with open(locations_path, 'r') as f: locations = json.load(f)
    locations_info = {locations[idx]['name']:locations[idx] for idx in range(len(locations))}
    cam_poses = init_cam_poses(instance_data_root)

    video_names = os.listdir(os.path.join(instance_data_root, "480_720", scene))
    video_names.remove('location_data.json')
    
    with open('./test_sets.json', 'r') as f: test_sets = json.load(f)

    for idx in range(len(test_sets)):
        
        eval_set = test_sets[str(idx)]
        video_caption_list = eval_set['entity_prompts']
        objs_num = len(video_caption_list) 
        location = eval_set['loc_prompt']
        video_name = eval_set['video_name']

        pose_embeds = get_pose_embeds(scene, video_name, instance_data_root, locations_info, cam_poses)

        prompt = ""
        for obj_idx in range(objs_num):
            video_caption = video_caption_list[obj_idx]
            if obj_idx == objs_num - 1:
                if objs_num == 1:
                    prompt += video_caption + ' is moving in the ' + location
                else:
                    prompt += video_caption + ' are moving in the ' + location
            else:
                prompt += video_caption + ' and '

        # 5. Generate the video frames based on the prompt.
        video_generate = pipe(
            prompt=prompt,
            prompts_list=video_caption_list,
            pose_embeds=pose_embeds[None],
            num_videos_per_prompt=num_videos_per_prompt,
            annealed_sample_step=annealed_sample_step,
            num_inference_steps=num_inference_steps,
            num_frames=49,
            use_dynamic_cfg=True,
            guidance_scale=guidance_scale,
            generator=torch.Generator().manual_seed(seed),
        ).frames[0]

        # 6. Export the generated frames to a video file. fps must be 8 for original video.
        save_video_name = ''
        save_video_name += str(objs_num) + '_' + video_name + '_' + location + '_'
        for obj_idx in range(objs_num):
            video_caption = video_caption_list[obj_idx][:30]
            video_caption = video_caption.replace(' ', '_')
            save_video_name += video_caption + '_'
        save_video_name += '.mp4'
        save_video_name = save_video_name.replace('_.mp4', '.mp4')
        save_video_path = os.path.join(output_path, save_video_name)
        export_to_video(video_generate, save_video_path, fps=8)

        with open(save_video_path.replace('.mp4', '.txt'), 'a+') as f:
            f.write(video_name)
            f.write('\n')
            for obj_idx in range(objs_num):
                f.write(video_caption_list[obj_idx])
                f.write('\n')
            f.write(location)
            f.write('\n')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a video from a text prompt using CogVideoX")
    parser.add_argument(
        "--model_path", type=str, default="THUDM/CogVideoX-5b", help="The path of the pre-trained model to be used"
    )
    parser.add_argument(
        "--ckpt_path", type=str, default="THUDM/CogVideoX-5b", help="The path of the pre-trained transformer to be used"
    )
    parser.add_argument("--lora_path", type=str, default=None, help="The path of the LoRA weights to be used")
    parser.add_argument("--lora_scale", type=float, default=1.0)
    parser.add_argument(
        "--output_path", type=str, default="./output.mp4", help="The path where the generated video will be saved"
    )
    parser.add_argument("--guidance_scale", type=float, default=6.0, help="The scale for classifier-free guidance")
    parser.add_argument(
        "--num_inference_steps", type=int, default=50, help="Number of steps for the inference process"
    )
    parser.add_argument("--annealed_sample_step", type=int, default=15, help="Number of videos to generate per prompt")
    parser.add_argument("--num_videos_per_prompt", type=int, default=1, help="Number of videos to generate per prompt")
    parser.add_argument(
        "--generate_type", type=str, default="t2v", help="The type of video generation (e.g., 't2v', 'i2v', 'v2v')"
    )
    parser.add_argument(
        "--dtype", type=str, default="bfloat16", help="The data type for computation (e.g., 'float16' or 'bfloat16')"
    )
    parser.add_argument("--seed", type=int, default=42, help="The seed for reproducibility")

    args = parser.parse_args()
    dtype = torch.float16 if args.dtype == "float16" else torch.bfloat16
    os.makedirs(args.output_path, exist_ok=True)
    generate_video(
        model_path=args.model_path,
        ckpt_path=args.ckpt_path,
        lora_path=args.lora_path,
        lora_scale=args.lora_scale,
        output_path=args.output_path,
        annealed_sample_step=args.annealed_sample_step,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        num_videos_per_prompt=args.num_videos_per_prompt,
        dtype=dtype,
        generate_type=args.generate_type,
        seed=args.seed,
    )
