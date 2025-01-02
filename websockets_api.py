#This is an example that uses the websockets api to know when a prompt execution is done
#Once the prompt execution is done it downloads the images using the /history endpoint

import websocket #NOTE: websocket-client (https://github.com/websocket-client/websocket-client)
import uuid
import json
import urllib.request
import urllib.parse
import os
import random

server_address = "home.plantplanethome.com:8188"
client_id = str(uuid.uuid4())

def queue_prompt(prompt):
    p = {"prompt": prompt, "client_id": client_id}
    data = json.dumps(p).encode('utf-8')
    req =  urllib.request.Request("http://{}/prompt".format(server_address), data=data)
    return json.loads(urllib.request.urlopen(req).read())

def get_image(filename, subfolder, folder_type):
    data = {"filename": filename, "subfolder": subfolder, "type": folder_type}
    url_values = urllib.parse.urlencode(data)
    with urllib.request.urlopen("http://{}/view?{}".format(server_address, url_values)) as response:
        return response.read()

def get_history(prompt_id):
    with urllib.request.urlopen("http://{}/history/{}".format(server_address, prompt_id)) as response:
        return json.loads(response.read())

def get_images(ws, prompt):
    prompt_id = queue_prompt(prompt)['prompt_id']
    output_images = {}
    while True:
        out = ws.recv()
        if isinstance(out, str):
            message = json.loads(out)
            if message['type'] == 'executing':
                data = message['data']
                if data['node'] is None and data['prompt_id'] == prompt_id:
                    break #Execution is done
        else:
            # If you want to be able to decode the binary stream for latent previews, here is how you can do it:
            # bytesIO = BytesIO(out[8:])
            # preview_image = Image.open(bytesIO) # This is your preview in PIL image format, store it in a global
            continue #previews are binary data

    history = get_history(prompt_id)[prompt_id]
    for node_id in history['outputs']:
        node_output = history['outputs'][node_id]
        images_output = []
        if 'images' in node_output:
            for image in node_output['images']:
                image_data = get_image(image['filename'], image['subfolder'], image['type'])
                images_output.append(image_data)
        output_images[node_id] = images_output

    return output_images

text2image_prompt_text = """
{
  "6": {
    "inputs": {
      "text": "",
      "clip": [
        "30",
        1
      ]
    },
    "class_type": "CLIPTextEncode"
  },
  "8": {
    "inputs": {
      "samples": [
        "31",
        0
      ],
      "vae": [
        "30",
        2
      ]
    },
    "class_type": "VAEDecode"
  },
  "9": {
    "inputs": {
      "filename_prefix": "ComfyUI",
      "images": [
        "8",
        0
      ]
    },
    "class_type": "SaveImage"
  },
  "27": {
    "inputs": {
      "width": 192,
      "height": 720,
      "batch_size": 1
    },
    "class_type": "EmptySD3LatentImage"
  },
  "30": {
    "inputs": {
      "ckpt_name": "flux/flux1-dev-fp8.safetensors"
    },
    "class_type": "CheckpointLoaderSimple"
  },
  "31": {
    "inputs": {
      "seed": 1022274292238835,
      "steps": 20,
      "cfg": 1,
      "sampler_name": "euler",
      "scheduler": "simple",
      "denoise": 1,
      "model": [
        "30",
        0
      ],
      "positive": [
        "35",
        0
      ],
      "negative": [
        "33",
        0
      ],
      "latent_image": [
        "27",
        0
      ]
    },
    "class_type": "KSampler"
  },
  "33": {
    "inputs": {
      "text": "text, watermark",
      "clip": [
        "30",
        1
      ]
    },
    "class_type": "CLIPTextEncode"
  },
  "35": {
    "inputs": {
      "guidance": 3.5,
      "conditioning": [
        "6",
        0
      ]
    },
    "class_type": "FluxGuidance"
  }
}
"""

image2image_prompt_text = '''{
  "3": {
    "inputs": {
      "seed": 572208726223836,
      "steps": 20,
      "cfg": 8,
      "sampler_name": "dpmpp_2m",
      "scheduler": "normal",
      "denoise": 0.8700000000000001,
      "model": [
        "14",
        0
      ],
      "positive": [
        "6",
        0
      ],
      "negative": [
        "7",
        0
      ],
      "latent_image": [
        "12",
        0
      ]
    },
    "class_type": "KSampler"
  },
  "6": {
    "inputs": {
      "text": "Add glasses on the basis of the original image",
      "clip": [
        "14",
        1
      ]
    },
    "class_type": "CLIPTextEncode"
  },
  "7": {
    "inputs": {
      "text": "watermark, text",
      "clip": [
        "14",
        1
      ]
    },
    "class_type": "CLIPTextEncode"
  },
  "8": {
    "inputs": {
      "samples": [
        "3",
        0
      ],
      "vae": [
        "14",
        2
      ]
    },
    "class_type": "VAEDecode"
  },
  "9": {
    "inputs": {
      "filename_prefix": "ComfyUI",
      "images": [
        "8",
        0
      ]
    },
    "class_type": "SaveImage"
  },
  "12": {
    "inputs": {
      "pixels": [
        "18",
        0
      ],
      "vae": [
        "14",
        2
      ]
    },
    "class_type": "VAEEncode"
  },
  "14": {
    "inputs": {
      "ckpt_name": "flux/flux1-dev-fp8.safetensors"
    },
    "class_type": "CheckpointLoaderSimple"
  },
  "16": {
    "inputs": {
      "image": "example.png",
      "upload": "image"
    },
    "class_type": "LoadImage"
  },
  "18": {
    "inputs": {
      "upscale_method": "nearest-exact",
      "width": 512,
      "height": 512,
      "crop": "disabled",
      "image": [
        "16",
        0
      ]
    },
    "class_type": "ImageScale"
  }
}'''

def generate_image(
    width: int,
    height: int,
    positive_prompt: str,
    negative_prompt: str = "text, watermark",
    reference_image: str = None,
) -> str:
    """
    生成图片并返回服务器上的图片路径
    
    Args:
        width: 图片宽度
        height: 图片高度
        positive_prompt: 正向提示词
        negative_prompt: 负向提示词
        reference_image: 参考图片路径（可选）
        
    Returns:
        str: 服务器上的图片路径
    """
    # 根据是否有参考图选择prompt模板
    if reference_image:
        prompt = json.loads(image2image_prompt_text)
        prompt["16"]["inputs"]["image"] = reference_image
        prompt["18"]["inputs"]["width"] = width
        prompt["18"]["inputs"]["height"] = height
    else:
        prompt = json.loads(text2image_prompt_text)
        prompt["27"]["inputs"]["width"] = width
        prompt["27"]["inputs"]["height"] = height
    
    # 设置提示词
    prompt["6"]["inputs"]["text"] = positive_prompt
    prompt["33" if not reference_image else "7"]["inputs"]["text"] = negative_prompt
    
    # 生成随机种子
    seed = random.randint(1, 1000000000000000)
    if reference_image:
        prompt["3"]["inputs"]["seed"] = seed
    else:
        prompt["31"]["inputs"]["seed"] = seed
    
    # 连接websocket并生成图片
    ws = websocket.WebSocket()
    ws.connect(f"ws://{server_address}/ws?clientId={client_id}")
    
    # 获取prompt_id
    prompt_id = queue_prompt(prompt)['prompt_id']
    images = get_images(ws, prompt)
    ws.close()
    
    # 获取图片路径
    server_path = None
    for node_id, node_images in images.items():
        if node_images:  # 如果该节点有图片输出
            # 获取第一张图片的信息
            history = get_history(prompt_id)[prompt_id]
            image_info = history['outputs'][node_id]['images'][0]
            # 构建服务器路径
            server_path = f"/home/ComfyUI/output/{image_info['filename']}"
            break
            
    return server_path

# 使用示例
if __name__ == "__main__":
    # 文生图示例
    # server_path = generate_image(
    #     width=512,
    #     height=512,
    #     positive_prompt="masterpiece best quality man",
    #     negative_prompt="text, watermark",
    # )
    # print(f"生成的图片服务器路径: {server_path}")
    
    # 图生图示例
    server_path = generate_image(
        width=512,
        height=512,
        positive_prompt="Add glasses on the basis of the original image",
        negative_prompt="text, watermark",
        reference_image="example.png"
    )
    print(f"生成的图片保存在: {server_path}")

