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
from remove_background import remove_background
import config


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("fastapi_app")

app = FastAPI()
timeout = httpx.Timeout(120.0)
security = HTTPBearer()

# CORS 설정 (Unity 클라이언트용)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Unity는 origin이 null이므로 * 필요
    allow_credentials=False,  # Unity에서는 credentials 사용 안 함
    allow_methods=["GET", "POST", "OPTIONS"],  # OPTIONS는 preflight 요청용
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
    num_images: int = 1
    prompt_style: str = "rd_plus__retro"


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
            # body를 읽기 위해 복사본 생성
            body = await request.body()
            if body:
                try:
                    # JSON으로 파싱 시도
                    body_json = json.loads(body.decode("utf-8"))
                    logger.info(
                        f"Request Body (JSON): {json.dumps(body_json, indent=2, ensure_ascii=False)}"
                    )
                except json.JSONDecodeError:
                    # JSON이 아닌 경우 텍스트로 로깅
                    logger.info(
                        f"Request Body (Raw): {body.decode('utf-8')[:1000]}..."
                    )  # 첫 1000자만
            else:
                logger.info("Request Body: Empty")

            # body를 다시 읽을 수 있도록 request 객체 수정
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


@app.post("/generate-image", response_model=ImageResponse)
async def generate_image(request: ImageRequest, _: str = Depends(verify_api_key)):
    """
    사용자가 보낸 prompt로 이미지를 생성하는 엔드포인트
    """
    logger.info(f"Starting image generation: {request.prompt}")
    url = "https://api.retrodiffusion.ai/v1/inferences"

    headers = {
        "X-RD-Token": config.RD_TOKEN,
    }

    enhanced_prompt = llm.enhance_prompt(request.prompt) + ", full body, full shape"

    payload = {
        "width": request.width,
        "height": request.height,
        "prompt": enhanced_prompt,
        "num_images": request.num_images,
        "prompt_style": request.prompt_style,
        # "remove_bg": True,
    }

    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(url, headers=headers, json=payload)
            response.raise_for_status()  # HTTP 에러가 있으면 예외 발생

            # JSON 응답 파싱
            response_data = response.json()

            # base64_images에서 첫 번째 이미지 추출
            if (
                "base64_images" not in response_data
                or not response_data["base64_images"]
            ):
                raise HTTPException(status_code=500, detail="응답에 이미지가 없습니다")

            first_image = response_data["base64_images"][0]
            nobg_image = remove_background(first_image)

            return ImageResponse(base64_image=nobg_image)

    except httpx.RequestError as e:
        raise HTTPException(status_code=500, detail=f"API 호출 실패: {str(e)}")
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=500, detail=f"JSON 파싱 실패: {str(e)}")
    except KeyError as e:
        raise HTTPException(status_code=500, detail=f"응답 형식 오류: {str(e)}")


@app.post("/generate-fish", response_model=ImageResponse)
async def generate_fish(request: ImageRequest, _: str = Depends(verify_api_key)):
    """
    사용자가 보낸 prompt로 물고기 이미지를 생성하는 엔드포인트
    """
    logger.info(f"Starting fish generation: {request.prompt}")
    url = "https://api.retrodiffusion.ai/v1/inferences"

    headers = {
        "X-RD-Token": config.RD_TOKEN,
    }

    enhanced_prompt = llm.enhance_prompt(request.prompt) + ", full body"

    payload = {
        "width": request.width,
        "height": request.height,
        "prompt": enhanced_prompt,
        "num_images": request.num_images,
        "prompt_style": request.prompt_style,
        # "remove_bg": True,
    }

    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(url, headers=headers, json=payload)
            response.raise_for_status()  # HTTP 에러가 있으면 예외 발생

            # JSON 응답 파싱
            response_data = response.json()

            # base64_images에서 첫 번째 이미지 추출
            if (
                "base64_images" not in response_data
                or not response_data["base64_images"]
            ):
                raise HTTPException(status_code=500, detail="응답에 이미지가 없습니다")

            first_image = response_data["base64_images"][0]
            nobg_image = remove_background(first_image)

            return ImageResponse(base64_image=nobg_image)

    except httpx.RequestError as e:
        print(f"Unexpected HTTPX error: {e}")
        raise HTTPException(status_code=500, detail=f"API 호출 실패: {str(e)}")
    except json.JSONDecodeError as e:
        print(f"Unexpected JSON error: {e}")
        raise HTTPException(status_code=500, detail=f"JSON 파싱 실패: {str(e)}")
    except KeyError as e:
        print(f"Unexpected Key error: {e}")
        raise HTTPException(status_code=500, detail=f"응답 형식 오류: {str(e)}")
    except Exception as e:
        print(f"Unexpected error: {e}")
        raise HTTPException(status_code=500, detail=f"예상치 못한 오류: {str(e)}")


@app.post("/generate-human", response_model=ImageResponse)
async def generate_human(request: ImageRequest, _: str = Depends(verify_api_key)):
    """
    사용자가 보낸 prompt로 인간 이미지를 생성하는 엔드포인트
    """
    logger.info(f"Starting human generation: {request.prompt}")
    url = "https://api.retrodiffusion.ai/v1/inferences"

    headers = {
        "X-RD-Token": config.RD_TOKEN,
    }

    enhanced_prompt = (
        llm.enhance_prompt(request.prompt)
        + ", 2D platformer style side view, full body, head, shoes"
    )

    payload = {
        "width": 64,
        "height": 128,
        "prompt": enhanced_prompt,
        "num_images": request.num_images,
        "prompt_style": "rd_fast__game_asset",
        # "remove_bg": True,
    }

    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(url, headers=headers, json=payload)
            response.raise_for_status()  # HTTP 에러가 있으면 예외 발생

            # JSON 응답 파싱
            response_data = response.json()

            # base64_images에서 첫 번째 이미지 추출
            if (
                "base64_images" not in response_data
                or not response_data["base64_images"]
            ):
                raise HTTPException(status_code=500, detail="응답에 이미지가 없습니다")

            first_image = response_data["base64_images"][0]
            nobg_image = remove_background(first_image)

            return ImageResponse(base64_image=nobg_image)

    except httpx.RequestError as e:
        raise HTTPException(status_code=500, detail=f"API 호출 실패: {str(e)}")
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=500, detail=f"JSON 파싱 실패: {str(e)}")
    except KeyError as e:
        raise HTTPException(status_code=500, detail=f"응답 형식 오류: {str(e)}")


@app.post("/generate-boat", response_model=ImageResponse)
async def generate_boat(request: ImageRequest, _: str = Depends(verify_api_key)):
    """
    사용자가 보낸 prompt로 보트 이미지를 생성하는 엔드포인트
    """
    logger.info(f"Starting boat generation: {request.prompt}")
    url = "https://api.retrodiffusion.ai/v1/inferences"

    headers = {
        "X-RD-Token": config.RD_TOKEN,
    }

    enhanced_prompt = (
        llm.enhance_prompt(request.prompt) + ", 2D platformer style side view"
    )

    payload = {
        "width": 128 + 64,
        "height": 64 + 32,
        "prompt": enhanced_prompt,
        "num_images": request.num_images,
        "prompt_style": "rd_fast__game_asset",
        # "remove_bg": True,
    }

    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(url, headers=headers, json=payload)
            response.raise_for_status()  # HTTP 에러가 있으면 예외 발생

            # JSON 응답 파싱
            response_data = response.json()

            # base64_images에서 첫 번째 이미지 추출
            if (
                "base64_images" not in response_data
                or not response_data["base64_images"]
            ):
                raise HTTPException(status_code=500, detail="응답에 이미지가 없습니다")

            first_image = response_data["base64_images"][0]
            nobg_image = remove_background(first_image)

            return ImageResponse(base64_image=nobg_image)

    except httpx.RequestError as e:
        raise HTTPException(status_code=500, detail=f"API 호출 실패: {str(e)}")
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=500, detail=f"JSON 파싱 실패: {str(e)}")
    except KeyError as e:
        raise HTTPException(status_code=500, detail=f"응답 형식 오류: {str(e)}")


@app.post("/generate-background", response_model=ImageResponse)
async def generate_background(request: ImageRequest, _: str = Depends(verify_api_key)):
    """
    사용자가 보낸 prompt로 배경 이미지를 생성하는 엔드포인트
    """
    logger.info(f"Starting background generation: {request.prompt}")
    url = "https://api.retrodiffusion.ai/v1/inferences"

    headers = {
        "X-RD-Token": config.RD_TOKEN,
    }

    enhanced_prompt = llm.enhance_prompt(request.prompt)

    payload = {
        "width": 320,
        "height": 180,
        "prompt": enhanced_prompt,
        "num_images": request.num_images,
        "prompt_style": request.prompt_style,
    }

    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(url, headers=headers, json=payload)
            response.raise_for_status()  # HTTP 에러가 있으면 예외 발생

            # JSON 응답 파싱
            response_data = response.json()

            # base64_images에서 첫 번째 이미지 추출
            if (
                "base64_images" not in response_data
                or not response_data["base64_images"]
            ):
                raise HTTPException(status_code=500, detail="응답에 이미지가 없습니다")

            first_image = response_data["base64_images"][0]

            return ImageResponse(base64_image=first_image)

    except httpx.RequestError as e:
        raise HTTPException(status_code=500, detail=f"API 호출 실패: {str(e)}")
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=500, detail=f"JSON 파싱 실패: {str(e)}")
    except KeyError as e:
        raise HTTPException(status_code=500, detail=f"응답 형식 오류: {str(e)}")


@app.post("/generate-reaction", response_model=ReactionResponse)
async def generate_reaction(request: ReactionRequest, _: str = Depends(verify_api_key)):
    reaction = llm.generate_reaction(
        request.location, request.human, request.boat, request.fish, request.size
    )

    return ReactionResponse(reaction=reaction)


@app.get("/")
async def root():
    """
    기본 엔드포인트 - 서버 상태 확인
    """
    return {"message": "이미지 생성 API 서버가 실행 중입니다"}


if __name__ == "__main__":
    import os

    # Cloud Run은 PORT 환경변수를 사용, 로컬은 8000
    port = int(os.getenv("PORT", 8000))

    print(f"Starting server on port {port}")
    print(
        f"Environment: {'Production (Cloud Run)' if os.getenv('PORT') else 'Development'}"
    )

    uvicorn.run(app, host="0.0.0.0", port=port)
