from fastapi import FastAPI, HTTPException, Depends, status, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import httpx
import uvicorn
import json
import logging
import time
from starlette.responses import Response
import llm
import config
from comfyui_client import ComfyUIClient
from workflow_manager import WorkflowManager, WorkflowTemplates


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("fastapi_app")

app = FastAPI()
timeout = httpx.Timeout(120.0)
security = HTTPBearer()

# ComfyUI 클라이언트와 워크플로우 매니저 초기화
comfy_client = ComfyUIClient("127.0.0.1:8188")
workflow_manager = WorkflowManager("./workflows/pixel_art_server.json")

# CORS 설정 (Unity 클라이언트용)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


def verify_api_key(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """API 키 검증"""
    if credentials.credentials != config.API_SECRET_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API key"
        )
    return credentials.credentials


# 요청 데이터 모델
class ImageRequest(BaseModel):
    prompt: str
    width: int = 64
    height: int = 64


# 응답 데이터 모델
class ImageResponse(BaseModel):
    base64_image: str


class ReactionRequest(BaseModel):
    location: str
    human: str
    boat: str
    fish: str
    size: str


class ReactionResponse(BaseModel):
    reaction: str


@app.middleware("http")
async def log_requests(request: Request, call_next):
    """요청과 응답을 로깅하는 미들웨어"""
    start_time = time.time()

    # 요청 정보 로깅
    logger.info(f"=== Incoming Request ===")
    logger.info(f"Method: {request.method}")
    logger.info(f"URL: {str(request.url)}")
    logger.info(f"Headers: {dict(request.headers)}")
    logger.info(f"Client IP: {request.client.host}")

    # 요청 Body 로깅 (POST 요청인 경우)
    if request.method in ["POST", "PUT", "PATCH"]:
        try:
            body = await request.body()
            if body:
                try:
                    body_json = json.loads(body.decode("utf-8"))
                    logger.info(
                        f"Request Body (JSON): {json.dumps(body_json, indent=2, ensure_ascii=False)}"
                    )
                except json.JSONDecodeError:
                    logger.info(f"Request Body (Raw): {body.decode('utf-8')[:1000]}...")
            else:
                logger.info("Request Body: Empty")

            async def receive():
                return {"type": "http.request", "body": body}

            request._receive = receive

        except Exception as e:
            logger.error(f"Error reading request body: {e}")

    # 실제 요청 처리
    response = await call_next(request)

    # 응답 시간 계산
    process_time = time.time() - start_time

    # 응답 정보 로깅
    logger.info(f"=== Response ===")
    logger.info(f"Status Code: {response.status_code}")
    logger.info(f"Response Headers: {dict(response.headers)}")
    logger.info(f"Process Time: {process_time:.4f} seconds")

    return response


@app.middleware("http")
async def add_security_headers(request, call_next):
    """보안 헤더 추가"""
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    return response


async def generate_with_comfyui(
    prompt: str, target_size: int = 64, remove_bg: bool = True
) -> str:
    """ComfyUI를 사용하여 이미지 생성"""
    try:
        # 프롬프트 향상
        enhanced_prompt = (
            llm.enhance_prompt(prompt) + ", pixel art style, clean, simple"
        )

        # 워크플로우 준비 (배경 제거 옵션 포함)
        workflow = workflow_manager.prepare_workflow(
            prompt=enhanced_prompt, target_size=target_size, remove_bg=remove_bg
        )

        # 이미지 생성
        logger.info(
            f"Generating image with ComfyUI: '{enhanced_prompt}', size: {target_size}, remove_bg: {remove_bg}"
        )
        base64_image = await comfy_client.generate_image(workflow, timeout=300)

        logger.info(
            f"ComfyUI generation successful, base64 length: {len(base64_image)}"
        )
        return base64_image

    except Exception as e:
        logger.error(f"ComfyUI generation failed: {e}")
        raise HTTPException(status_code=500, detail=f"이미지 생성 실패: {str(e)}")


@app.post("/generate-image", response_model=ImageResponse)
async def generate_image(request: ImageRequest, _: str = Depends(verify_api_key)):
    """
    사용자가 보낸 prompt로 이미지를 생성하는 엔드포인트 (ComfyUI 사용)
    """
    logger.info(
        f"Starting ComfyUI image generation: {request.prompt}, remove_bg: {True}"
    )

    try:
        # ComfyUI로 이미지 생성 (배경 제거 옵션 전달)
        base64_image = await generate_with_comfyui(
            prompt=request.prompt,
            target_size=max(request.width, request.height),
            remove_bg=True,
        )

        return ImageResponse(base64_image=base64_image)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in generate_image: {e}")
        raise HTTPException(status_code=500, detail=f"예상치 못한 오류: {str(e)}")


@app.post("/generate-fish", response_model=ImageResponse)
async def generate_fish(request: ImageRequest, _: str = Depends(verify_api_key)):
    """
    사용자가 보낸 prompt로 물고기 이미지를 생성하는 엔드포인트 (ComfyUI 사용)
    """
    logger.info(
        f"Starting ComfyUI fish generation: {request.prompt}, remove_bg: {True}"
    )

    try:
        # 물고기 특화 프롬프트
        fish_prompt = f"{request.prompt}, fish, aquatic creature, swimming"

        base64_image = await generate_with_comfyui(
            prompt=fish_prompt, target_size=64, remove_bg=True
        )

        return ImageResponse(base64_image=base64_image)

    except Exception as e:
        logger.error(f"Fish generation error: {e}")
        raise HTTPException(
            status_code=500, detail=f"물고기 이미지 생성 실패: {str(e)}"
        )


@app.post("/generate-human", response_model=ImageResponse)
async def generate_human(request: ImageRequest, _: str = Depends(verify_api_key)):
    """
    사용자가 보낸 prompt로 인간 이미지를 생성하는 엔드포인트 (ComfyUI 사용)
    """
    logger.info(
        f"Starting ComfyUI human generation: {request.prompt}, remove_bg: {True}"
    )

    try:
        # 인간 특화 프롬프트
        human_prompt = (
            f"{request.prompt}, human character, person, full body, 2D platformer style"
        )

        base64_image = await generate_with_comfyui(
            prompt=human_prompt, target_size=64, remove_bg=True
        )

        return ImageResponse(base64_image=base64_image)

    except Exception as e:
        logger.error(f"Human generation error: {e}")
        raise HTTPException(status_code=500, detail=f"인간 이미지 생성 실패: {str(e)}")


@app.post("/generate-boat", response_model=ImageResponse)
async def generate_boat(request: ImageRequest, _: str = Depends(verify_api_key)):
    """
    사용자가 보낸 prompt로 보트 이미지를 생성하는 엔드포인트 (ComfyUI 사용)
    """
    logger.info(
        f"Starting ComfyUI boat generation: {request.prompt}, remove_bg: {True}"
    )

    try:
        # 보트 특화 프롬프트
        boat_prompt = f"{request.prompt}, boat, ship, vessel, nautical, side view"

        base64_image = await generate_with_comfyui(
            prompt=boat_prompt, target_size=128, remove_bg=True
        )

        return ImageResponse(base64_image=base64_image)

    except Exception as e:
        logger.error(f"Boat generation error: {e}")
        raise HTTPException(status_code=500, detail=f"보트 이미지 생성 실패: {str(e)}")


@app.post("/generate-background", response_model=ImageResponse)
async def generate_background(request: ImageRequest, _: str = Depends(verify_api_key)):
    """
    사용자가 보낸 prompt로 배경 이미지를 생성하는 엔드포인트 (ComfyUI 사용)
    """
    logger.info(f"Starting ComfyUI background generation: {request.prompt}")

    try:
        # 배경 특화 프롬프트
        background_prompt = (
            f"{request.prompt}, landscape, background, scenery, environment"
        )

        # 배경은 기본적으로 배경 제거 안함 (request.remove_bg 무시하고 False 사용)
        base64_image = await generate_with_comfyui(
            prompt=background_prompt,
            target_size=256,
            remove_bg=False,  # 배경 이미지는 배경 제거 안함
        )

        return ImageResponse(base64_image=base64_image)

    except Exception as e:
        logger.error(f"Background generation error: {e}")
        raise HTTPException(status_code=500, detail=f"배경 이미지 생성 실패: {str(e)}")


@app.post("/generate-reaction", response_model=ReactionResponse)
async def generate_reaction(request: ReactionRequest, _: str = Depends(verify_api_key)):
    reaction = llm.generate_reaction(
        request.location, request.human, request.boat, request.fish, request.size
    )

    return ReactionResponse(reaction=reaction)


@app.get("/health")
async def health_check():
    """서버 및 ComfyUI 상태 확인"""
    try:
        # ComfyUI 연결 확인
        comfy_health = await comfy_client.health_check()
        queue_info = await comfy_client.get_queue_info()

        return {
            "status": "healthy",
            "comfyui_connected": comfy_health,
            "comfyui_queue": queue_info,
            "message": "서버가 정상 작동 중입니다",
        }
    except Exception as e:
        return {
            "status": "partial",
            "comfyui_connected": False,
            "error": str(e),
            "message": "서버는 작동 중이지만 ComfyUI 연결에 문제가 있습니다",
        }


@app.get("/")
async def root():
    """
    기본 엔드포인트 - 서버 상태 확인
    """
    return {"message": "ComfyUI 통합 이미지 생성 API 서버가 실행 중입니다"}


if __name__ == "__main__":
    import os

    # Cloud Run은 PORT 환경변수를 사용, 로컬은 8000
    port = int(os.getenv("PORT", 8000))

    print(f"Starting ComfyUI integrated server on port {port}")
    print(
        f"Environment: {'Production (Cloud Run)' if os.getenv('PORT') else 'Development'}"
    )
    print(f"ComfyUI Address: {comfy_client.server_address}")

    uvicorn.run(app, host="0.0.0.0", port=port)
