# workflow_manager.py - 수정된 버전

import json
import os
from typing import Dict, Any, Optional


class WorkflowManager:
    def __init__(
        self,
        bg_remove_path: str = "./workflows/pixel_art_bg_remove.json",
        bg_keep_path: str = "./workflows/pixel_art_bg_keep.json",
    ):
        # 두 개의 워크플로우 파일 로드
        self.bg_remove_workflow = self._load_workflow(bg_remove_path)
        self.bg_keep_workflow = self._load_workflow(bg_keep_path)

    def _load_workflow(self, path: str) -> Dict[str, Any]:
        """워크플로우 JSON 파일 로드"""
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except FileNotFoundError:
            return None

    def prepare_workflow(
        self, prompt: str, width: int = 64, height: int = 64, remove_bg: bool = True
    ) -> Dict[str, Any]:
        """
        remove_bg 값에 따라 적절한 워크플로우 선택 + 파라미터 주입
        """
        # 워크플로우 선택
        if remove_bg:
            if self.bg_remove_workflow is None:
                raise Exception(
                    "배경 제거 워크플로우 파일이 없습니다: pixel_art_bg_remove.json"
                )
            workflow = json.loads(json.dumps(self.bg_remove_workflow))
        else:
            if self.bg_keep_workflow is None:
                raise Exception(
                    "배경 유지 워크플로우 파일이 없습니다: pixel_art_bg_keep.json"
                )
            workflow = json.loads(json.dumps(self.bg_keep_workflow))

        # 파라미터 주입 (실제 노드 ID 기준)
        if "nodes" in workflow:
            for node in workflow["nodes"]:
                # 프롬프트 설정 (노드 14 - CLIPTextEncode)
                if node.get("id") == 14:
                    if "widgets_values" in node and len(node["widgets_values"]) > 0:
                        node["widgets_values"][0] = prompt

                # 너비 설정 (노드 18 - Target Width)
                elif node.get("id") == 18:
                    if "widgets_values" in node:
                        node["widgets_values"] = [str(width)]

                # 높이 설정 (노드 19 - Target Height)
                elif node.get("id") == 19:
                    if "widgets_values" in node:
                        node["widgets_values"] = [str(height)]

        return workflow

    def get_output_node_id(self) -> str:
        """최종 출력 노드 ID 반환 (SaveImage 노드)"""
        return "23"

    def validate_workflow(self, workflow: Dict[str, Any]) -> bool:
        """워크플로우 유효성 검사"""
        # 필수 노드들 (실제 워크플로우 기준)
        required_nodes = ["1", "2", "5", "14", "18", "19", "23"]

        if "nodes" in workflow:
            existing_node_ids = [str(node.get("id")) for node in workflow["nodes"]]
            for node_id in required_nodes:
                if node_id not in existing_node_ids:
                    return False
        return True


# 템플릿 클래스는 그대로 유지
class WorkflowTemplates:
    @staticmethod
    def for_fish(
        prompt: str, width: int = 256, height: int = 256, remove_bg: bool = True
    ) -> Dict:
        return {
            "prompt": prompt,
            "width": width,
            "height": height,
            "remove_bg": remove_bg,
        }

    @staticmethod
    def for_human(
        prompt: str, width: int = 64, height: int = 128, remove_bg: bool = True
    ) -> Dict:
        return {
            "prompt": prompt,
            "width": width,
            "height": height,
            "remove_bg": remove_bg,
        }

    @staticmethod
    def for_boat(
        prompt: str, width: int = 128, height: int = 64, remove_bg: bool = True
    ) -> Dict:
        return {
            "prompt": prompt,
            "width": width,
            "height": height,
            "remove_bg": remove_bg,
        }

    @staticmethod
    def for_background(
        prompt: str, width: int = 320, height: int = 180, remove_bg: bool = False
    ) -> Dict:
        return {
            "prompt": prompt,
            "width": width,
            "height": height,
            "remove_bg": remove_bg,
        }
