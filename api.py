from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import uvicorn
from websockets_api import generate_image

app = FastAPI(title="ComfyUI API", description="ComfyUI的HTTP接口封装")

class ImageGenerationRequest(BaseModel):
    width: int = 512
    height: int = 512
    positive_prompt: str
    negative_prompt: str = "text, watermark"
    reference_image: Optional[str] = None

class ImageGenerationResponse(BaseModel):
    image_path: str

@app.post("/generate", response_model=ImageGenerationResponse)
async def generate(request: ImageGenerationRequest):
    """
    生成图片接口
    
    - 文生图：不传reference_image
    - 图生图：传入reference_image（服务器上的图片路径）
    """
    try:
        image_path = generate_image(
            width=request.width,
            height=request.height,
            positive_prompt=request.positive_prompt,
            negative_prompt=request.negative_prompt,
            reference_image=request.reference_image
        )
        
        if not image_path:
            raise HTTPException(status_code=500, detail="图片生成失败")
            
        return ImageGenerationResponse(image_path=image_path)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8189) 