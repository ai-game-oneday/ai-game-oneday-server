import requests
import base64
from io import BytesIO
from PIL import Image
import httpx
import asyncio

url = "https://api.retrodiffusion.ai/v1/inferences"
headers = {
    "X-RD-Token": "rdpk-86e108fa1d830ec5a4c9e286cb0b8899",
}

payload = {
    "width": 64,
    "height": 64,
    "prompt": "A fish, 2D platformer style side view, full body",
    "num_images": 1,
    "prompt_style": "rd_fast__game_asset",
}


async def get_async():
    async with httpx.AsyncClient() as client:
        response = await client.post(url, headers=headers, json=payload)
        return response


async def main():
    print(payload)
    # response = requests.post(url, headers=headers, json=payload)
    response = await get_async()
    print(response)

    # 1) JSON 파싱
    data = response.json()

    # 2) 첫 번째 base64 문자열 얻기
    b64_str = data["base64_images"][0]

    # 3) base64 → bytes → Pillow Image 변환
    img_bytes = base64.b64decode(b64_str)
    img = Image.open(BytesIO(img_bytes))

    # 4) 보기 (로컬 스크립트라면 OS 기본 이미지 뷰어로 열림)
    img.show()

    # 선택) 파일로 저장하고 싶다면
    img.save("result.png")


if __name__ == "__main__":
    asyncio.run(main())
