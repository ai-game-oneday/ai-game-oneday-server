from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import httpx
import uvicorn
import json
import llm
from remove_background import remove_background
import traceback
import asyncio

app = FastAPI()


class ImageRequest(BaseModel):
    prompt: str
    width: int = 64
    height: int = 64
    num_images: int = 1
    prompt_style: str = "rd_plus__retro"


class ImageResponse(BaseModel):
    base64_image: str


@app.post("/generate-fish", response_model=ImageResponse)
async def generate_fish(request: ImageRequest):
    """
    사용자가 보낸 prompt로 물고기 이미지를 생성하는 엔드포인트
    """
    url = "https://api.retrodiffusion.ai/v1/inferences"

    headers = {
        "X-RD-Token": "rdpk-c9c6911ae1e01e3a986e25209740aa50",
    }

    try:
        print(f"🐟 Fish generation started")

        enhanced_prompt = llm.enhance_prompt(request.prompt) + ", full body"
        print(f"✅ Enhanced prompt: {enhanced_prompt}")

        payload = {
            "width": request.width,
            "height": request.height,
            "prompt": enhanced_prompt,
            "num_images": request.num_images,
            "prompt_style": request.prompt_style,
        }
        print(f"📦 Payload: {payload}")
        print(f"🌐 URL: {url}")
        print(f"📋 Headers: {headers}")

        # httpx 버전 확인
        print(f"📚 httpx version: {httpx.__version__}")

        print(f"🔍 AsyncClient 생성 시도...")

        try:
            # 1. 기본 AsyncClient 생성 테스트
            client = httpx.AsyncClient()
            print(f"✅ AsyncClient 생성 성공")

            print(f"🌐 POST 요청 시도...")

            # 2. POST 요청 시도
            response = await client.post(url, headers=headers, json=payload)
            print(f"✅ POST 요청 성공!")
            print(f"📡 Response status: {response.status_code}")

            await client.aclose()  # 클라이언트 명시적 종료

        except Exception as post_error:
            print(f"❌ POST 요청 실패: {post_error}")
            print(f"🔍 Error type: {type(post_error)}")
            print(f"📍 Full traceback: {traceback.format_exc()}")

            # 대안 1: requests로 테스트해보기
            try:
                import requests

                print(f"🔄 requests로 시도...")
                req_response = requests.post(
                    url, headers=headers, json=payload, timeout=30
                )
                print(f"✅ requests 성공: {req_response.status_code}")

                # requests가 성공하면 httpx 설정 문제
                raise HTTPException(
                    status_code=500,
                    detail=f"httpx 문제 - requests는 성공: {str(post_error)}",
                )

            except requests.exceptions.RequestException as req_error:
                print(f"❌ requests도 실패: {req_error}")
                raise HTTPException(
                    status_code=500, detail=f"네트워크 문제: {str(req_error)}"
                )

            raise HTTPException(
                status_code=500, detail=f"POST 요청 실패: {str(post_error)}"
            )

        # 정상 처리 계속...
        response.raise_for_status()
        response_data = response.json()

        if "base64_images" not in response_data or not response_data["base64_images"]:
            raise HTTPException(status_code=500, detail="응답에 이미지가 없습니다")

        first_image = response_data["base64_images"][0]
        nobg_image = remove_background(first_image)

        return ImageResponse(base64_image=nobg_image)

    except HTTPException:
        raise
    except Exception as e:
        print(f"💥 예상치 못한 에러: {e}")
        print(f"🔍 Error type: {type(e)}")
        print(f"📍 Full traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"예상치 못한 오류: {str(e)}")


# 추가 테스트 엔드포인트
@app.get("/test-httpx")
async def test_httpx():
    """httpx 기본 동작 테스트"""
    try:
        print(f"🧪 httpx 테스트 시작...")

        async with httpx.AsyncClient() as client:
            # 간단한 GET 요청으로 테스트
            response = await client.get("https://httpbin.org/get")
            print(f"✅ httpx GET 성공: {response.status_code}")

            # 간단한 POST 요청으로 테스트
            test_response = await client.post(
                "https://httpbin.org/post", json={"test": "data"}
            )
            print(f"✅ httpx POST 성공: {test_response.status_code}")

        return {"status": "httpx working fine"}

    except Exception as e:
        print(f"❌ httpx 테스트 실패: {e}")
        return {"status": f"httpx error: {str(e)}"}


@app.get("/")
async def root():
    return {"message": "디버깅 서버 실행 중"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
