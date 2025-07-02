# comfyui_client.py - 수정된 버전
import httpx
import asyncio
import json
import base64
import uuid
import websockets
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class ComfyUIClient:
    def __init__(self, server_address: str = "127.0.0.1:8188"):
        self.server_address = server_address
        self.base_url = f"http://{server_address}"
        self.ws_url = f"ws://{server_address}/ws"
        self.client_id = str(uuid.uuid4())

    async def generate_image(self, workflow: Dict[str, Any], timeout: int = 300) -> str:
        """
        워크플로우를 실행하고 base64 인코딩된 이미지 반환
        """
        try:
            # 1. 워크플로우를 큐에 추가
            prompt_id = await self._queue_prompt(workflow)
            logger.info(f"Prompt queued with ID: {prompt_id}")

            # 2. 완료될 때까지 대기
            result = await self._wait_for_completion(prompt_id, timeout)
            logger.info(f"Generation completed")

            # 3. 이미지 다운로드 및 base64 인코딩
            image_base64 = await self._download_and_encode_image(result)

            return image_base64

        except Exception as e:
            logger.error(f"ComfyUI generation failed: {e}")
            raise

    async def _queue_prompt(self, workflow: Dict[str, Any]) -> str:
        """워크플로우를 ComfyUI 큐에 추가"""
        # 워크플로우를 API 형식으로 변환
        api_workflow = self._convert_to_api_format(workflow)
        payload = {"prompt": api_workflow, "client_id": self.client_id}

        logger.info("워크플로우를 ComfyUI 큐에 추가 중...")

        async with httpx.AsyncClient(timeout=30.0) as client:
            try:
                response = await client.post(f"{self.base_url}/prompt", json=payload)

                if response.status_code != 200:
                    error_text = response.text
                    logger.error(f"API 에러: {error_text}")
                    raise Exception(f"ComfyUI API 에러: {error_text}")

                result = response.json()
                prompt_id = result["prompt_id"]

                logger.info(f"워크플로우 큐 추가 성공: {prompt_id}")
                return prompt_id

            except Exception as e:
                logger.error(f"워크플로우 큐 추가 실패: {e}")
                raise

    def _convert_to_api_format(self, workflow: Dict[str, Any]) -> Dict[str, Any]:
        """워크플로우를 ComfyUI API 형식으로 변환"""
        # 이미 API 형식인지 확인 (nodes 키가 있으면 UI 형식)
        if "nodes" not in workflow:
            return workflow

        api_format = {}

        # 1단계: 모든 노드 기본 구조 생성
        for node in workflow["nodes"]:
            node_id = str(node["id"])
            api_format[node_id] = {"inputs": {}, "class_type": node["type"]}

        # 2단계: widgets_values를 inputs로 변환
        for node in workflow["nodes"]:
            node_id = str(node["id"])
            if "widgets_values" in node and node["widgets_values"]:
                inputs = self._map_widgets_to_inputs(node)
                api_format[node_id]["inputs"].update(inputs)

        # 3단계: 연결된 입력들 처리
        for node in workflow["nodes"]:
            node_id = str(node["id"])
            if "inputs" in node:
                for input_info in node["inputs"]:
                    if input_info.get("link") is not None:
                        connected_input = self._find_connected_input(
                            workflow, input_info["link"]
                        )
                        if connected_input:
                            input_name = input_info["name"]
                            if node["type"] == "Reroute":
                                api_format[node_id]["inputs"]["input"] = connected_input
                            else:
                                api_format[node_id]["inputs"][
                                    input_name
                                ] = connected_input

        return api_format

    def _map_widgets_to_inputs(self, node: Dict[str, Any]) -> Dict[str, Any]:
        """노드의 widgets_values를 inputs로 매핑"""
        inputs = {}
        node_type = node["type"]
        widgets_values = node.get("widgets_values", [])

        # 주요 노드 타입별 매핑
        widget_mappings = {
            "CLIPTextEncode": {"text": 0},
            "Int": {"Number": 0},
            "DualCLIPLoader": {
                "clip_name1": 0,
                "clip_name2": 1,
                "type": 2,
                "device": 3,
            },
            "UNETLoader": {"unet_name": 0, "weight_dtype": 1},
            "VAELoader": {"vae_name": 0},
            "FluxGuidance": {"guidance": 0},
            "BasicScheduler": {"scheduler": 0, "steps": 1, "denoise": 2},
            "KSamplerSelect": {"sampler_name": 0},
            "ModelSamplingFlux": {
                "max_shift": 0,
                "base_shift": 1,
                "width": 2,
                "height": 3,
            },
            "RandomNoise": {"noise_seed": 0},
            "ADE_EmptyLatentImageLarge": {"width": 0, "height": 1, "batch_size": 2},
            "Seed (rgthree)": {"seed": 0},
            "SaveImage": {"filename_prefix": 0},
            "LatentResolutionAdjuster": {"width": 0, "height": 1},
        }

        # 복잡한 노드들은 개별 처리
        if node_type == "pixelerate":
            if len(widgets_values) >= 8:
                inputs.update(
                    {
                        "palette": widgets_values[0],
                        "dither_strength": widgets_values[1],
                        "upscale_factor": widgets_values[4],
                        "shrink_mode": widgets_values[5],
                        "mode_ratio": widgets_values[6],
                        "mode_threshold": widgets_values[7],
                    }
                )
        elif node_type == "RMBG":
            if len(widgets_values) >= 8:
                inputs.update(
                    {
                        "model": widgets_values[0],
                        "sensitivity": widgets_values[1],
                        "process_res": widgets_values[2],
                        "mask_blur": widgets_values[3],
                        "mask_offset": widgets_values[4],
                        "invert_output": widgets_values[5],
                        "refine_foreground": widgets_values[6],
                        "background": widgets_values[7],
                    }
                )
        elif node_type == "ColorMaskToDepthMask //Inspire":
            if len(widgets_values) >= 4:
                inputs.update(
                    {
                        "spec": widgets_values[0],
                        "base_value": widgets_values[1],
                        "dilation": widgets_values[2],
                        "flatten_method": widgets_values[3],
                    }
                )
        elif node_type == "pixel_outline":
            if len(widgets_values) >= 9:
                inputs.update(
                    {
                        "outline_enabled": widgets_values[0],
                        "outline_r": widgets_values[1],
                        "outline_g": widgets_values[2],
                        "outline_b": widgets_values[3],
                        "outline_a": widgets_values[4],
                        "black_threshold": widgets_values[5],
                        "remove_lonely_enabled": widgets_values[6],
                        "lonely_n": widgets_values[7],
                        "scale_factor": widgets_values[8],
                    }
                )
        else:
            # 일반적인 매핑 적용
            if node_type in widget_mappings:
                mapping = widget_mappings[node_type]
                for param_name, index in mapping.items():
                    if index < len(widgets_values):
                        inputs[param_name] = widgets_values[index]

        return inputs

    def _find_connected_input(
        self, workflow: Dict[str, Any], link_id: int
    ) -> Optional[list]:
        """링크 ID로 연결된 입력 찾기"""
        if "links" not in workflow:
            return None

        for link in workflow["links"]:
            if len(link) >= 6 and link[0] == link_id:
                source_node_id = str(link[1])
                source_slot = link[2]
                return [source_node_id, source_slot]

        return None

    async def _wait_for_completion(
        self, prompt_id: str, timeout: int
    ) -> Dict[str, Any]:
        """WebSocket을 통해 완료 대기"""
        try:
            async with websockets.connect(
                f"{self.ws_url}?clientId={self.client_id}"
            ) as websocket:
                start_time = asyncio.get_event_loop().time()

                while True:
                    # 타임아웃 체크
                    if asyncio.get_event_loop().time() - start_time > timeout:
                        raise TimeoutError(
                            f"Generation timeout after {timeout} seconds"
                        )

                    try:
                        message = await asyncio.wait_for(websocket.recv(), timeout=10.0)
                        data = json.loads(message)

                        # 완료 이벤트 확인
                        if (
                            data.get("type") == "executed"
                            and data.get("data", {}).get("prompt_id") == prompt_id
                        ):
                            logger.info("WebSocket 완료 이벤트 수신")

                            # WebSocket 이벤트만 믿지 말고 실제 히스토리 확인
                            await asyncio.sleep(0.5)  # 잠시 대기
                            status = await self._check_queue_status(prompt_id)
                            if status["completed"]:
                                logger.info("히스토리에서 완료 확인됨")
                                return status["result"]
                            else:
                                logger.warning(
                                    "WebSocket 완료 이벤트와 히스토리 불일치, 계속 대기..."
                                )
                                continue

                        # 에러 이벤트 확인
                        if data.get("type") == "execution_error":
                            error_data = data.get("data", {})
                            raise Exception(f"ComfyUI 실행 에러: {error_data}")

                    except asyncio.TimeoutError:
                        # 주기적으로 상태 확인
                        status = await self._check_queue_status(prompt_id)
                        if status["completed"]:
                            return status["result"]
                        elif status["error"]:
                            raise Exception(f"ComfyUI 에러: {status['error']}")

        except Exception as e:
            logger.warning(f"WebSocket 연결 실패, 폴링으로 전환: {e}")
            return await self._wait_by_polling(prompt_id, timeout)

    async def _wait_by_polling(self, prompt_id: str, timeout: int) -> Dict[str, Any]:
        """폴링을 통해 완료 대기"""
        start_time = asyncio.get_event_loop().time()

        while True:
            if asyncio.get_event_loop().time() - start_time > timeout:
                raise TimeoutError(f"Generation timeout after {timeout} seconds")

            status = await self._check_queue_status(prompt_id)
            if status["completed"]:
                return status["result"]
            elif status["error"]:
                raise Exception(f"ComfyUI 에러: {status['error']}")

            await asyncio.sleep(2)

    async def _check_queue_status(self, prompt_id: str) -> Dict[str, Any]:
        """큐 상태 확인"""
        async with httpx.AsyncClient(timeout=10.0) as client:
            try:
                # 히스토리 확인
                response = await client.get(f"{self.base_url}/history/{prompt_id}")
                if response.status_code == 200:
                    history = response.json()
                    if prompt_id in history:
                        return {
                            "completed": True,
                            "result": history[prompt_id],
                            "error": None,
                        }

                # 큐 상태 확인
                response = await client.get(f"{self.base_url}/queue")
                queue_data = response.json()

                # 실행 중이거나 대기 중인지 확인
                for item in queue_data.get("queue_running", []) + queue_data.get(
                    "queue_pending", []
                ):
                    if item[1] == prompt_id:
                        return {"completed": False, "result": None, "error": None}

                return {
                    "completed": False,
                    "result": None,
                    "error": f"Prompt {prompt_id} not found",
                }

            except Exception as e:
                return {"completed": False, "result": None, "error": str(e)}

    async def _download_and_encode_image(self, result_data: Dict[str, Any]) -> str:
        """결과에서 이미지를 다운로드하고 base64로 인코딩"""
        try:
            # 상태 확인
            if "status" in result_data:
                status = result_data["status"]
                if status.get("status_str") == "error":
                    raise Exception(f"ComfyUI 실행 실패: {status.get('messages', [])}")

            # 출력에서 이미지 정보 추출
            outputs = result_data.get("outputs")
            if not outputs:
                # 히스토리 데이터 구조 확인을 위한 디버깅
                logger.error(
                    f"출력이 없습니다. 전체 결과 데이터: {json.dumps(result_data, indent=2, default=str)}"
                )
                raise Exception("ComfyUI 실행 실패 - 출력이 없습니다")

            # 이미지 정보 찾기 - 더 포괄적으로 검색
            images_info = None

            # 1. 직접 "images" 키가 있는 경우
            if "images" in outputs:
                images_info = outputs["images"]
                logger.info("직접 images 키에서 발견")

            # 2. 노드 23 (SaveImage) 확인
            elif (
                "23" in outputs
                and isinstance(outputs["23"], dict)
                and "images" in outputs["23"]
            ):
                images_info = outputs["23"]["images"]
                logger.info("노드 23에서 이미지 발견")

            # 3. 모든 출력 노드에서 images 키 검색
            else:
                for node_id, node_output in outputs.items():
                    if isinstance(node_output, dict) and "images" in node_output:
                        images_info = node_output["images"]
                        logger.info(f"노드 {node_id}에서 이미지 발견")
                        break

            if not images_info:
                # 더 자세한 디버깅 정보
                logger.error(f"이미지를 찾을 수 없습니다.")
                logger.error(f"사용 가능한 출력 노드: {list(outputs.keys())}")
                for node_id, node_output in outputs.items():
                    logger.error(f"노드 {node_id} 출력 타입: {type(node_output)}")
                    if isinstance(node_output, dict):
                        logger.error(f"노드 {node_id} 키들: {list(node_output.keys())}")

                raise Exception(
                    f"이미지 출력을 찾을 수 없습니다. 출력 노드: {list(outputs.keys())}"
                )

            # 첫 번째 이미지 정보 추출
            if isinstance(images_info, list) and len(images_info) > 0:
                image_info = images_info[0]
            else:
                raise Exception(f"잘못된 이미지 정보 형식: {type(images_info)}")

            # 이미지 다운로드
            filename = image_info.get("filename")
            subfolder = image_info.get("subfolder", "")
            folder_type = image_info.get("type", "output")

            if not filename:
                raise Exception(f"파일명을 찾을 수 없습니다: {image_info}")

            logger.info(f"이미지 다운로드 중: {filename}")
            image_data = await self._download_image(filename, subfolder, folder_type)

            # base64 인코딩
            image_base64 = base64.b64encode(image_data).decode("utf-8")
            logger.info(f"이미지 인코딩 완료, 길이: {len(image_base64)}")

            return image_base64

        except Exception as e:
            logger.error(f"이미지 다운로드 및 인코딩 실패: {e}")
            raise

    async def _download_image(
        self, filename: str, subfolder: str, folder_type: str
    ) -> bytes:
        """ComfyUI에서 이미지 다운로드"""
        params = {"filename": filename, "type": folder_type}
        if subfolder:
            params["subfolder"] = subfolder

        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(f"{self.base_url}/view", params=params)
            response.raise_for_status()
            return response.content

    async def health_check(self) -> bool:
        """ComfyUI 서버 상태 확인"""
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(f"{self.base_url}/queue")
                return response.status_code == 200
        except:
            return False

    async def get_queue_info(self) -> Dict[str, Any]:
        """현재 큐 정보 조회"""
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(f"{self.base_url}/queue")
            response.raise_for_status()

            queue_data = response.json()
            return {
                "running": len(queue_data.get("queue_running", [])),
                "pending": len(queue_data.get("queue_pending", [])),
                "total": len(queue_data.get("queue_running", []))
                + len(queue_data.get("queue_pending", [])),
            }
