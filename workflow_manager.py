# workflow_manager.py (ImpactSwitch 지원)
import json
import os
from typing import Dict, Any, Optional


class WorkflowManager:
    def __init__(self, workflow_path: str = "./workflows/pixel_art_server.json"):
        self.base_workflow = self._load_workflow(workflow_path)

    def _load_workflow(self, path: str) -> Dict[str, Any]:
        """워크플로우 JSON 파일 로드"""
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def prepare_workflow(
        self, prompt: str, target_size: int = 64, remove_bg: bool = True
    ) -> Dict[str, Any]:
        """
        워크플로우에 프롬프트, 타겟 크기, 배경 제거 여부 주입

        Args:
            prompt: 텍스트 프롬프트
            target_size: 타겟 크기 (정사각형)
            remove_bg: True면 배경 제거, False면 배경 유지
        """
        workflow = json.loads(json.dumps(self.base_workflow))  # Deep copy

        # nodes 배열에서 노드 찾기
        if "nodes" in workflow:
            for node in workflow["nodes"]:
                # 1. 프롬프트 설정 (32번 노드: CLIPTextEncode)
                if node.get("id") == 32:
                    if "widgets_values" in node and len(node["widgets_values"]) > 0:
                        node["widgets_values"][0] = prompt

                # 2. 타겟 크기 설정 (20번 노드: Target Width)
                elif node.get("id") == 20:
                    if "widgets_values" in node:
                        node["widgets_values"] = [str(target_size)]

                # 3. 배경 제거 스위치 설정 (37번 노드: ImpactSwitch)
                elif node.get("id") == 37:
                    if "widgets_values" in node and len(node["widgets_values"]) >= 1:
                        # select 값 설정: 1=배경유지, 2=배경제거
                        switch_value = 2 if remove_bg else 1
                        node["widgets_values"][0] = switch_value

        return workflow

    def get_output_node_id(self) -> str:
        """최종 출력 노드 ID 반환 (SaveImage 노드)"""
        return "34"

    def validate_workflow(self, workflow: Dict[str, Any]) -> bool:
        """워크플로우 유효성 검사"""
        required_nodes = [
            "1",
            "2",
            "7",
            "32",
            "17",
            "20",
            "34",
            "37",
        ]  # ImpactSwitch(37) 추가

        if "nodes" in workflow:
            existing_node_ids = [str(node.get("id")) for node in workflow["nodes"]]
            for node_id in required_nodes:
                if node_id not in existing_node_ids:
                    return False
        else:
            for node_id in required_nodes:
                if node_id not in workflow:
                    return False
        return True


# 엔드포인트별 설정
class WorkflowTemplates:
    """각 엔드포인트별 특화된 설정"""

    @staticmethod
    def for_general_image(
        prompt: str, target_size: int = 64, remove_bg: bool = True
    ) -> Dict:
        """기본 이미지 생성용"""
        return {"prompt": prompt, "target_size": target_size, "remove_bg": remove_bg}

    @staticmethod
    def for_fish(prompt: str, target_size: int = 64, remove_bg: bool = True) -> Dict:
        """물고기 이미지 생성용"""
        return {"prompt": prompt, "target_size": target_size, "remove_bg": remove_bg}

    @staticmethod
    def for_human(prompt: str, target_size: int = 64, remove_bg: bool = True) -> Dict:
        """인간 이미지 생성용"""
        return {"prompt": prompt, "target_size": target_size, "remove_bg": remove_bg}

    @staticmethod
    def for_boat(prompt: str, target_size: int = 128, remove_bg: bool = True) -> Dict:
        """보트 이미지 생성용"""
        return {"prompt": prompt, "target_size": target_size, "remove_bg": remove_bg}

    @staticmethod
    def for_background(
        prompt: str, target_size: int = 256, remove_bg: bool = False
    ) -> Dict:
        """배경 이미지 생성용 - 기본적으로 배경 제거 안함"""
        return {"prompt": prompt, "target_size": target_size, "remove_bg": remove_bg}
