# comfyui_client.py
import httpx
import asyncio
import json
import base64
import uuid
import websockets
from typing import Dict, Any, Optional, Tuple
import logging
from io import BytesIO

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

        Args:
            workflow: ComfyUI 워크플로우 JSON
            timeout: 타임아웃 (초)

        Returns:
            base64 인코딩된 이미지 문자열
        """
        try:
            # 1. 워크플로우를 큐에 추가
            prompt_id = await self._queue_prompt(workflow)
            logger.info(f"Prompt queued with ID: {prompt_id}")

            # 2. 완료될 때까지 대기
            result = await self._wait_for_completion(prompt_id, timeout)
            logger.info(f"Generation completed: {result}")

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

        # 디버깅: 요청 페이로드 로깅
        logger.info(f"Sending payload to ComfyUI API...")
        logger.debug(f"Payload keys: {list(payload.keys())}")
        logger.debug(f"Prompt keys: {list(api_workflow.keys())}")

        async with httpx.AsyncClient(timeout=30.0) as client:
            try:
                response = await client.post(f"{self.base_url}/prompt", json=payload)

                # 응답 상태 확인
                logger.info(f"Response status: {response.status_code}")

                if response.status_code != 200:
                    # 에러 응답 내용 로깅
                    error_text = response.text
                    logger.error(f"API Error Response: {error_text}")

                response.raise_for_status()

                result = response.json()
                return result["prompt_id"]

            except Exception as e:
                logger.error(f"Failed to queue prompt: {e}")
                raise

    def _convert_to_api_format(self, workflow: Dict[str, Any]) -> Dict[str, Any]:
        """워크플로우를 ComfyUI API 형식으로 변환"""

        # 이미 API 형식인지 확인 (nodes 키가 있으면 UI 형식)
        if "nodes" in workflow:
            # UI 형식을 API 형식으로 변환
            api_format = {}

            for node in workflow["nodes"]:
                node_id = str(node["id"])
                api_format[node_id] = {"inputs": {}, "class_type": node["type"]}

                # widgets_values를 inputs로 변환
                if "widgets_values" in node and node["widgets_values"]:
                    # 노드 타입별로 inputs 매핑
                    inputs = self._map_widgets_to_inputs(node)
                    api_format[node_id]["inputs"].update(inputs)

                # 연결된 입력들 처리 (이게 widgets_values를 덮어쓸 수 있음)
                if "inputs" in node:
                    for input_info in node["inputs"]:
                        if input_info.get("link") is not None:
                            # 링크 정보로부터 연결 정보 찾기
                            connected_input = self._find_connected_input(
                                workflow, input_info["link"]
                            )
                            if connected_input:
                                # Reroute 노드의 경우 입력 이름을 "input"으로 설정
                                if node["type"] == "Reroute":
                                    api_format[node_id]["inputs"][
                                        "input"
                                    ] = connected_input
                                else:
                                    # 연결된 입력은 widgets_values를 덮어씀
                                    api_format[node_id]["inputs"][
                                        input_info["name"]
                                    ] = connected_input

            return api_format
        else:
            # 이미 API 형식이면 그대로 반환
            return workflow

    def _map_widgets_to_inputs(self, node: Dict[str, Any]) -> Dict[str, Any]:
        """노드의 widgets_values를 inputs로 매핑"""
        inputs = {}
        node_type = node["type"]
        widgets_values = node.get("widgets_values", [])

        # 노드 타입별 매핑 규칙
        if node_type == "CLIPTextEncode" and len(widgets_values) > 0:
            inputs["text"] = widgets_values[0]
        elif node_type == "Int" and len(widgets_values) > 0:
            inputs["Number"] = widgets_values[0]
        elif node_type == "DualCLIPLoader" and len(widgets_values) >= 2:
            inputs["clip_name1"] = widgets_values[0]
            inputs["clip_name2"] = widgets_values[1]
            if len(widgets_values) > 2:
                inputs["type"] = widgets_values[2]
            if len(widgets_values) > 3:
                inputs["device"] = widgets_values[3]
        elif node_type == "UNETLoader" and len(widgets_values) > 0:
            inputs["unet_name"] = widgets_values[0]
            if len(widgets_values) > 1:
                inputs["weight_dtype"] = widgets_values[1]
        elif node_type == "VAELoader" and len(widgets_values) > 0:
            inputs["vae_name"] = widgets_values[0]
        elif node_type == "FluxGuidance" and len(widgets_values) > 0:
            inputs["guidance"] = widgets_values[0]
        elif node_type == "BasicScheduler" and len(widgets_values) >= 3:
            inputs["scheduler"] = widgets_values[0]
            inputs["steps"] = widgets_values[1]
            inputs["denoise"] = widgets_values[2]
        elif node_type == "KSamplerSelect" and len(widgets_values) > 0:
            inputs["sampler_name"] = widgets_values[0]
        elif node_type == "ModelSamplingFlux" and len(widgets_values) >= 4:
            # 모든 필수 입력값 설정
            inputs["max_shift"] = widgets_values[0]
            inputs["base_shift"] = widgets_values[1]
            inputs["width"] = widgets_values[2]
            inputs["height"] = widgets_values[3]
        elif node_type == "RandomNoise" and len(widgets_values) > 0:
            inputs["noise_seed"] = widgets_values[0]
        elif node_type == "ADE_EmptyLatentImageLarge" and len(widgets_values) >= 3:
            inputs["width"] = widgets_values[0]
            inputs["height"] = widgets_values[1]
            inputs["batch_size"] = widgets_values[2]
        elif node_type == "Seed (rgthree)" and len(widgets_values) > 0:
            inputs["seed"] = widgets_values[0]
        elif node_type == "pixelerate" and len(widgets_values) >= 8:
            # 모든 필수 입력값 설정
            inputs["palette"] = widgets_values[0]
            inputs["dither_strength"] = widgets_values[1]
            # widgets_values[2], [3]은 target_width, target_height (연결된 입력)
            inputs["upscale_factor"] = widgets_values[4]
            inputs["shrink_mode"] = widgets_values[5]
            inputs["mode_ratio"] = widgets_values[6]
            inputs["mode_threshold"] = widgets_values[7]
        elif node_type == "RMBG" and len(widgets_values) >= 8:
            inputs["model"] = widgets_values[0]
            inputs["sensitivity"] = widgets_values[1]
            # widgets_values[2]는 process_res (연결된 입력)
            inputs["mask_blur"] = widgets_values[3]
            inputs["mask_offset"] = widgets_values[4]
            inputs["invert_output"] = widgets_values[5]
            inputs["refine_foreground"] = widgets_values[6]
            inputs["background"] = widgets_values[7]
        elif node_type == "ColorMaskToDepthMask //Inspire" and len(widgets_values) >= 4:
            inputs["spec"] = widgets_values[0]
            inputs["base_value"] = widgets_values[1]
            inputs["dilation"] = widgets_values[2]
            inputs["flatten_method"] = widgets_values[3]
        elif node_type == "pixel_outline" and len(widgets_values) >= 9:
            inputs["outline_enabled"] = widgets_values[0]
            inputs["outline_r"] = widgets_values[1]
            inputs["outline_g"] = widgets_values[2]
            inputs["outline_b"] = widgets_values[3]
            inputs["outline_a"] = widgets_values[4]
            inputs["black_threshold"] = widgets_values[5]
            inputs["remove_lonely_enabled"] = widgets_values[6]
            inputs["lonely_n"] = widgets_values[7]
            inputs["scale_factor"] = widgets_values[8]
        elif node_type == "PreviewTextNode" and len(widgets_values) > 0:
            inputs["text"] = widgets_values[0]
        elif node_type == "SaveImage" and len(widgets_values) > 0:
            inputs["filename_prefix"] = widgets_values[0]
        elif node_type == "ImpactSwitch" and len(widgets_values) >= 1:
            inputs["select"] = widgets_values[0]
            if len(widgets_values) > 1:
                inputs["sel_mode"] = widgets_values[1]
        elif node_type == "PreviewImage":
            # PreviewImage는 widgets_values가 없음
            pass
        elif node_type == "Reroute":
            # Reroute 노드는 widgets_values가 없고 입력만 전달
            pass
        elif node_type == "JoinImageWithAlpha":
            # JoinImageWithAlpha는 widgets_values가 없음
            pass
        elif node_type == "VAEDecode":
            # VAEDecode는 widgets_values가 없음
            pass
        elif node_type == "SamplerCustomAdvanced":
            # SamplerCustomAdvanced는 widgets_values가 없음
            pass
        elif node_type == "BasicGuider":
            # BasicGuider는 widgets_values가 없음
            pass
        # 기타 노드들에 대한 기본 처리
        else:
            # 알려지지 않은 노드 타입에 대한 로그
            logger.debug(
                f"Unknown node type: {node_type} with widgets: {widgets_values}"
            )

        return inputs

    def _find_connected_input(
        self, workflow: Dict[str, Any], link_id: int
    ) -> Optional[list]:
        """링크 ID로 연결된 입력 찾기"""
        if "links" not in workflow:
            return None

        for link in workflow["links"]:
            if len(link) >= 6 and link[0] == link_id:
                # [link_id, source_node_id, source_slot, target_node_id, target_slot, type]
                source_node_id = str(link[1])
                source_slot = link[2]
                return [source_node_id, source_slot]

        return None

    async def _wait_for_completion(
        self, prompt_id: str, timeout: int
    ) -> Dict[str, Any]:
        """WebSocket을 통해 완료 대기"""
        try:
            # WebSocket 연결 시 timeout 파라미터 제거
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
                            logger.info("Received execution completion event")

                            # 디버깅: 완료 이벤트 데이터 구조 로깅
                            logger.info(f"WebSocket completion data structure: {data}")

                            return {
                                "prompt_id": prompt_id,
                                "outputs": data["data"].get("output", {}),
                            }

                        # 에러 이벤트 확인
                        if data.get("type") == "execution_error":
                            raise Exception(f"ComfyUI execution error: {data}")

                    except asyncio.TimeoutError:
                        # 큐 상태 확인으로 폴백
                        status = await self._check_queue_status(prompt_id)
                        if status["completed"]:
                            return status["result"]
                        elif status["error"]:
                            raise Exception(f"ComfyUI error: {status['error']}")

        except (
            websockets.exceptions.ConnectionClosed,
            websockets.exceptions.InvalidURI,
            OSError,
            Exception,
        ) as e:
            # WebSocket 연결 실패시 폴링으로 폴백
            logger.warning(f"WebSocket connection failed: {e}, falling back to polling")
            return await self._wait_by_polling(prompt_id, timeout)

    async def _check_queue_status(self, prompt_id: str) -> Dict[str, Any]:
        """큐 상태 확인"""
        async with httpx.AsyncClient(timeout=10.0) as client:
            try:
                # 히스토리 확인
                response = await client.get(f"{self.base_url}/history/{prompt_id}")
                if response.status_code == 200:
                    history = response.json()
                    logger.info(f"History response type: {type(history)}")
                    logger.info(
                        f"History response keys: {list(history.keys()) if isinstance(history, dict) else 'Not a dict'}"
                    )

                    # 히스토리 전체 내용을 파일로 저장
                    import json

                    with open("debug_full_history.json", "w", encoding="utf-8") as f:
                        json.dump(history, f, indent=2, ensure_ascii=False, default=str)
                    logger.info("Saved full history to debug_full_history.json")

                    if prompt_id in history:
                        history_item = history[prompt_id]
                        logger.info(f"History item type: {type(history_item)}")
                        logger.info(
                            f"History item keys: {list(history_item.keys()) if isinstance(history_item, dict) else 'Not a dict'}"
                        )

                        # outputs 구조 자세히 확인
                        if isinstance(history_item, dict) and "outputs" in history_item:
                            outputs = history_item["outputs"]
                            logger.info(f"Outputs type: {type(outputs)}")
                            logger.info(
                                f"Outputs keys: {list(outputs.keys()) if isinstance(outputs, dict) else 'Not a dict'}"
                            )

                            # 각 출력 노드 상세 분석
                            for node_id, node_output in outputs.items():
                                logger.info(f"Node {node_id}:")
                                logger.info(f"  Type: {type(node_output)}")
                                logger.info(f"  Content: {node_output}")

                                if isinstance(node_output, dict):
                                    logger.info(f"  Keys: {list(node_output.keys())}")
                                elif isinstance(node_output, list):
                                    logger.info(f"  List length: {len(node_output)}")
                                    if len(node_output) > 0:
                                        logger.info(
                                            f"  First item type: {type(node_output[0])}"
                                        )
                                        logger.info(f"  First item: {node_output[0]}")

                        return {
                            "completed": True,
                            "result": history_item,
                            "error": None,
                        }

                # 큐 상태 확인
                response = await client.get(f"{self.base_url}/queue")
                queue_data = response.json()

                # 실행 중인지 확인
                for item in queue_data.get("queue_running", []):
                    if item[1] == prompt_id:
                        return {"completed": False, "result": None, "error": None}

                # 대기 중인지 확인
                for item in queue_data.get("queue_pending", []):
                    if item[1] == prompt_id:
                        return {"completed": False, "result": None, "error": None}

                # 큐에 없으면 에러로 간주
                return {
                    "completed": False,
                    "result": None,
                    "error": f"Prompt {prompt_id} not found in queue",
                }

            except Exception as e:
                logger.error(f"Error in _check_queue_status: {e}")
                import traceback

                logger.error(f"Full traceback: {traceback.format_exc()}")
                return {"completed": False, "result": None, "error": str(e)}

    async def _download_and_encode_image(self, result_data: Dict[str, Any]) -> str:
        """결과에서 이미지를 다운로드하고 base64로 인코딩"""
        try:
            # 디버깅: 결과 데이터 타입과 구조 확인
            logger.info(f"Result data type: {type(result_data)}")
            logger.info(
                f"Result data keys: {list(result_data.keys()) if isinstance(result_data, dict) else 'Not a dict'}"
            )

            # result_data가 리스트인 경우 처리
            if isinstance(result_data, list):
                logger.warning("Result data is a list, taking first item")
                if len(result_data) > 0:
                    result_data = result_data[0]
                else:
                    raise Exception("Result data is empty list")

            # 상태 확인
            if "status" in result_data:
                status = result_data["status"]
                logger.info(f"Execution status: {status}")

                if status.get("status_str") == "error":
                    error_details = status.get("messages", [])
                    logger.error(f"ComfyUI execution error: {error_details}")
                    raise Exception(f"ComfyUI execution failed: {error_details}")

            # 출력 노드에서 이미지 정보 추출
            outputs = result_data.get("outputs", {})
            logger.info(f"Available output keys: {list(outputs.keys())}")
            logger.info(f"Outputs type: {type(outputs)}")

            # outputs 구조를 파일로 저장
            import json

            with open("debug_outputs_v2.json", "w", encoding="utf-8") as f:
                json.dump(outputs, f, indent=2, ensure_ascii=False, default=str)
            logger.info("Saved outputs structure to debug_outputs_v2.json")

            images_info = None
            found_source = None

            # 방법 1: 직접 "images" 키가 있는 경우 (통합된 출력)
            if "images" in outputs:
                images_info = outputs["images"]
                found_source = "unified_images"
                logger.info(f"Found images in unified output: {images_info}")

            # 방법 2: 노드별 출력에서 찾기 (기존 방식)
            else:
                image_nodes = ["34", "28", "21", "19"]

                for node_id in image_nodes:
                    if node_id in outputs:
                        node_output = outputs[node_id]
                        logger.info(
                            f"Checking node {node_id}, type: {type(node_output)}"
                        )

                        # 노드 출력이 딕셔너리이고 images 키가 있는 경우
                        if isinstance(node_output, dict) and "images" in node_output:
                            images_info = node_output["images"]
                            found_source = f"node_{node_id}"
                            logger.info(f"Found images in node {node_id} (dict format)")
                            break

                        # 노드 출력이 리스트인 경우
                        elif isinstance(node_output, list) and len(node_output) > 0:
                            first_item = node_output[0]
                            if isinstance(first_item, dict) and (
                                "filename" in first_item or "type" in first_item
                            ):
                                images_info = node_output
                                found_source = f"node_{node_id}_list"
                                logger.info(
                                    f"Found images in node {node_id} (list format)"
                                )
                                break

            if not images_info:
                logger.warning("No images found in any output")

                if not outputs:
                    raise Exception(
                        "No outputs found - ComfyUI execution may have failed"
                    )
                else:
                    raise Exception(
                        f"No images found in outputs. Available keys: {list(outputs.keys())}"
                    )

            logger.info(f"Found images from: {found_source}")
            logger.info(f"Images info: {images_info}")

            # 첫 번째 이미지 정보 추출
            if isinstance(images_info, list) and len(images_info) > 0:
                image_info = images_info[0]
            else:
                raise Exception(f"Invalid images_info format: {type(images_info)}")

            logger.info(f"Image info: {image_info}")

            # 이미지 정보에서 필요한 필드 추출
            filename = image_info.get("filename")
            subfolder = image_info.get("subfolder", "")
            folder_type = image_info.get("type", "output")

            if not filename:
                raise Exception(f"No filename found in image info: {image_info}")

            logger.info(f"Downloading image: {filename}")
            logger.info(f"Subfolder: '{subfolder}', Type: '{folder_type}'")

            # 이미지 다운로드
            image_data = await self._download_image(filename, subfolder, folder_type)

            # base64 인코딩
            image_base64 = base64.b64encode(image_data).decode("utf-8")
            logger.info(
                f"Successfully encoded image to base64, length: {len(image_base64)}"
            )

            return image_base64

        except Exception as e:
            logger.error(f"Failed to download and encode image: {e}")
            import traceback

            logger.error(f"Full traceback: {traceback.format_exc()}")
            raise

    async def _get_image_from_history(self, result_data: Dict[str, Any]) -> str:
        """히스토리에서 이미지 찾기"""
        try:
            # prompt_id가 있다면 히스토리 API 호출
            if "prompt_id" in result_data:
                prompt_id = result_data["prompt_id"]
                async with httpx.AsyncClient(timeout=30.0) as client:
                    response = await client.get(f"{self.base_url}/history/{prompt_id}")
                    response.raise_for_status()
                    history = response.json()

                    if prompt_id in history:
                        history_outputs = history[prompt_id].get("outputs", {})
                        logger.info(
                            f"History output nodes: {list(history_outputs.keys())}"
                        )

                        # 히스토리에서 이미지 찾기 (34번 SaveImage 노드 우선)
                        for node_id in ["34", "28", "21", "19"]:
                            if (
                                node_id in history_outputs
                                and "images" in history_outputs[node_id]
                            ):
                                images_info = history_outputs[node_id]["images"]
                                image_info = images_info[0]

                                filename = image_info["filename"]
                                subfolder = image_info.get("subfolder", "")
                                folder_type = image_info.get("type", "output")

                                logger.info(
                                    f"Found image in history node {node_id}: {filename}"
                                )

                                # 이미지 다운로드
                                image_data = await self._download_image(
                                    filename, subfolder, folder_type
                                )
                                return base64.b64encode(image_data).decode("utf-8")

            raise Exception("No images found in history either")

        except Exception as e:
            logger.error(f"Failed to get image from history: {e}")
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
