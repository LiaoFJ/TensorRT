#
# SPDX-FileCopyrightText: Copyright (c) 1993-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import argparse

from cuda import cudart

from stable_diffusion_pipeline import StableDiffusionPipeline
from utilities import PIPELINE_TYPE, TRT_LOGGER, add_arguments, process_pipeline_args, save_image
from transformers import CLIPTokenizer


def convert_list_to_string(input_value):
    # 检查 input_value 是否为字符串
    if isinstance(input_value, str):
        return input_value
    # 检查 input_value 是否为列表
    elif isinstance(input_value, list):
        # 将列表转换为字符串，元素之间用逗号分隔
        return ', '.join(map(str, input_value))
    else:
        # 如果 input_value 既不是字符串也不是列表，返回原值或抛出异常
        raise ValueError("wrong type of prompt, should be str or list")


def split_prompts(prompts, target_group_size=75, num_groups=4):
    prompts = convert_list_to_string(prompts)
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    prompts_list = tokenizer.tokenize(prompts)
    # 定义组列表
    groups = [[] for _ in range(num_groups)]
    current_group = 0
    index = 0
    while index < len(prompts_list) and current_group < num_groups:
        current_count = 0
        # 从当前索引开始，试图找到最近的逗号结尾的单词，同时不超过75个单词的限制
        while current_count < target_group_size and index < len(prompts_list):
            word = prompts_list[index]
            groups[current_group].append(word)
            current_count += 1
            index += 1
            # 如果当前单词以逗号结尾，检查是否已达到或接近组容量
            if (',' in word or '.' in word) and current_count >= target_group_size:
                break
            # 如果达到75个单词但未以逗号结尾，需要回溯到最近的逗号
            if current_count == target_group_size:
                if not word.endswith(','):
                    # 回溯到最近的逗号结尾
                    while not ('.' in groups[current_group][-1] or ',' in groups[current_group][-1]):
                        groups[current_group].pop()
                        index -= 1
                break
        # 为下一个组做准备
        if current_count > 0 and current_group < num_groups - 1:
            current_group += 1
    # 清除尾部可能的空组
    groups = [group for group in groups if group]
    # 重新搞成字符串
    groups = [tokenizer.convert_tokens_to_string(group) for group in groups]
    return groups


def parseArgs():
    parser = argparse.ArgumentParser(description="Options for Stable Diffusion XL Txt2Img Demo",
                                     conflict_handler='resolve')
    parser = add_arguments(parser)
    parser.add_argument('--input-image', type=str, default="", help="Path to the input image")
    parser.add_argument('--input-image-strength', type=float, default=0.75, help="strength of the input image")
    parser.add_argument('--version', type=str, default="xl-1.0", choices=["xl-1.0", "xl-turbo"],
                        help="Version of Stable Diffusion XL")
    parser.add_argument('--height', type=int, default=1024, help="Height of image to generate (must be multiple of 8)")
    parser.add_argument('--width', type=int, default=1024, help="Height of image to generate (must be multiple of 8)")
    parser.add_argument('--num-warmup-runs', type=int, default=1,
                        help="Number of warmup runs before benchmarking performance")

    parser.add_argument('--guidance-scale', type=float, default=0.0,
                        help="Value of classifier-free guidance scale (must be greater than 1)")

    parser.add_argument('--enable-refiner', action='store_true', help="Enable SDXL-Refiner model")
    parser.add_argument('--image-strength', type=float, default=0.3,
                        help="Strength of transformation applied to input_image (must be between 0 and 1)")
    parser.add_argument('--onnx-refiner-dir', default='onnx_xl_refiner', help="Directory for SDXL-Refiner ONNX models")
    parser.add_argument('--engine-refiner-dir', default='engine_xl_refiner',
                        help="Directory for SDXL-Refiner TensorRT engines")

    return parser.parse_args()


class StableDiffusionXLPipeline(StableDiffusionPipeline):
    def __init__(self, vae_scaling_factor=0.13025, enable_refiner=False, **kwargs):
        self.enable_refiner = enable_refiner
        self.nvtx_profile = kwargs['nvtx_profile']
        self.base = StableDiffusionPipeline(
            pipeline_type=PIPELINE_TYPE.XL_EXTENSION,
            vae_scaling_factor=vae_scaling_factor,
            return_latents=self.enable_refiner,
            **kwargs)
        if self.enable_refiner:
            self.refiner = StableDiffusionPipeline(
                pipeline_type=PIPELINE_TYPE.XL_REFINER,
                vae_scaling_factor=vae_scaling_factor,
                return_latents=False,
                **kwargs)

    def loadEngines(self, framework_model_dir, onnx_dir, engine_dir, onnx_refiner_dir='onnx_xl_refiner',
                    engine_refiner_dir='engine_xl_refiner', **kwargs):
        self.base.loadEngines(engine_dir, framework_model_dir, onnx_dir, **kwargs)
        if self.enable_refiner:
            self.refiner.loadEngines(engine_refiner_dir, framework_model_dir, onnx_refiner_dir, **kwargs)

    def activateEngines(self, shared_device_memory=None):
        self.base.activateEngines(shared_device_memory)
        if self.enable_refiner:
            self.refiner.activateEngines(shared_device_memory)

    def loadResources(self, image_height, image_width, batch_size, len_prompt, seed):
        self.base.loadResources(image_height, image_width, batch_size, len_prompt, seed)
        if self.enable_refiner:
            # Use a different seed for refiner - we arbitrarily use base seed+1, if specified.
            self.refiner.loadResources(image_height, image_width, batch_size, len_prompt,
                                       ((seed + 1) if seed is not None else None))

    def get_max_device_memory(self):
        max_device_memory = self.base.calculateMaxDeviceMemory()
        if self.enable_refiner:
            max_device_memory = max(max_device_memory, self.refiner.calculateMaxDeviceMemory())
        return max_device_memory

    def run(self, prompt, negative_prompt, height, width, batch_size, batch_count, num_warmup_runs, use_cuda_graph,
            **kwargs_infer_refiner):
        # Process prompt
        if not isinstance(prompt, list):
            raise ValueError(f"`prompt` must be of type `str` list, but is {type(prompt)}")
        prompt = prompt * batch_size

        if not isinstance(negative_prompt, list):
            raise ValueError(f"`--negative-prompt` must be of type `str` list, but is {type(negative_prompt)}")
        if len(negative_prompt) == 1:
            negative_prompt = negative_prompt * batch_size

        num_warmup_runs = max(1, num_warmup_runs) if use_cuda_graph else num_warmup_runs
        if num_warmup_runs > 0:
            print("[I] Warming up ..")
            for _ in range(num_warmup_runs):
                images, _ = self.base.infer(prompt, negative_prompt, height, width, warmup=True)
                if args.enable_refiner:
                    images, _ = self.refiner.infer(prompt, negative_prompt, height, width, input_image=images,
                                                   warmup=True, **kwargs_infer_refiner)

        ret = []
        for _ in range(batch_count):
            print("[I] Running StableDiffusionXL pipeline")
            if self.nvtx_profile:
                cudart.cudaProfilerStart()
            latents, time_base = self.base.infer(prompt, negative_prompt, height, width, warmup=False)
            if self.enable_refiner:
                images, time_refiner = self.refiner.infer(prompt, negative_prompt, height, width, input_image=latents,
                                                          warmup=False, **kwargs_infer_refiner)
                ret.append(images)
            else:
                ret.append(latents)

            if self.nvtx_profile:
                cudart.cudaProfilerStop()
            if self.enable_refiner:
                print('|-----------------|--------------|')
                print('| {:^15} | {:>9.2f} ms |'.format('e2e', time_base + time_refiner))
                print('|-----------------|--------------|')

            image_path_dir = "./output"
            image_name = "saveimage"
            save_image(images, image_path_dir, image_name) if self.enable_refiner else save_image(images,
                                                                                                  image_path_dir,
                                                                                                  image_name)

        return ret

    def teardown(self):
        self.base.teardown()
        if self.enable_refiner:
            self.refiner.teardown()


if __name__ == "__main__":
    print("[I] Initializing TensorRT accelerated StableDiffusionXL txt2img pipeline")
    args = parseArgs()

    kwargs_init_pipeline, kwargs_load_engine, args_run_demo = process_pipeline_args(args)

    # Initialize demo
    demo = StableDiffusionXLPipeline(vae_scaling_factor=0.13025, enable_refiner=args.enable_refiner,
                                     **kwargs_init_pipeline)
    # Load TensorRT engines and pytorch modules
    kwargs_load_refiner = {'onnx_refiner_dir': args.onnx_refiner_dir,
                           'engine_refiner_dir': args.engine_refiner_dir} if args.enable_refiner else {}
    demo.loadEngines(
        args.framework_model_dir,
        args.onnx_dir,
        args.engine_dir,
        **kwargs_load_refiner,
        **kwargs_load_engine)

    # split prompts
    args.prompt = split_prompts(args.prompt)

    # Load resources
    _, shared_device_memory = cudart.cudaMalloc(demo.get_max_device_memory())
    demo.activateEngines(shared_device_memory)
    demo.loadResources(args.height, args.width, args.batch_size, len(args.prompt), args.seed)

    # Run inference
    kwargs_infer_refiner = {'image_strength': args.image_strength} if args.enable_refiner else {}
    demo.run(*args_run_demo, **kwargs_infer_refiner)

    demo.teardown()
