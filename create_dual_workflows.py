# create_dual_workflows.py - 기존 워크플로우에서 2개 버전 생성

import json
import os


def create_dual_workflows():
    """기존 워크플로우에서 배경제거/유지 버전 2개 생성"""

    # 기존 워크플로우 로드
    base_path = "./workflows/pixel_art_server.json"
    with open(base_path, "r", encoding="utf-8") as f:
        base_workflow = json.load(f)

    print("기존 워크플로우 로드 완료")

    # 🔥 1. 배경 제거 워크플로우 (기존과 동일)
    bg_remove_workflow = json.loads(json.dumps(base_workflow))
    # 이미 28번 -> 34번으로 연결되어 있으므로 그대로 사용

    # 🔥 2. 배경 유지 워크플로우 (21번 -> 34번 직접 연결)
    bg_keep_workflow = json.loads(json.dumps(base_workflow))

    # SaveImage 노드 연결 변경
    if "nodes" in bg_keep_workflow:
        for node in bg_keep_workflow["nodes"]:
            if node.get("id") == 34:  # SaveImage 노드
                if "inputs" in node:
                    for input_info in node["inputs"]:
                        if input_info["name"] == "images":
                            input_info["link"] = 27  # 21번 pixelerate에서 직접

    # links 배열 수정
    if "links" in bg_keep_workflow:
        # 기존 SaveImage 연결 제거 (28번 -> 34번)
        bg_keep_workflow["links"] = [
            link
            for link in bg_keep_workflow["links"]
            if not (len(link) >= 6 and link[1] == 28 and link[3] == 34)
        ]

        # 새 연결 추가 (21번 -> 34번)
        new_link = [27, 21, 0, 34, 0, "IMAGE"]
        bg_keep_workflow["links"].append(new_link)

    # 🔥 3. 파일 저장
    bg_remove_path = "./workflows/pixel_art_bg_remove.json"
    bg_keep_path = "./workflows/pixel_art_bg_keep.json"

    with open(bg_remove_path, "w", encoding="utf-8") as f:
        json.dump(bg_remove_workflow, f, indent=2, ensure_ascii=False)

    with open(bg_keep_path, "w", encoding="utf-8") as f:
        json.dump(bg_keep_workflow, f, indent=2, ensure_ascii=False)

    print(f"✅ 배경 제거 워크플로우 생성: {bg_remove_path}")
    print(f"✅ 배경 유지 워크플로우 생성: {bg_keep_path}")

    # 🔥 4. 연결 확인
    print("\n=== 연결 확인 ===")

    # 배경 제거 워크플로우 연결 확인
    bg_remove_saveimage = None
    for node in bg_remove_workflow["nodes"]:
        if node.get("id") == 34:
            for input_info in node["inputs"]:
                if input_info["name"] == "images":
                    bg_remove_saveimage = input_info["link"]

    # 배경 유지 워크플로우 연결 확인
    bg_keep_saveimage = None
    for node in bg_keep_workflow["nodes"]:
        if node.get("id") == 34:
            for input_info in node["inputs"]:
                if input_info["name"] == "images":
                    bg_keep_saveimage = input_info["link"]

    print(
        f"배경 제거: SaveImage 연결 = link {bg_remove_saveimage} (28번 pixel_outline)"
    )
    print(f"배경 유지: SaveImage 연결 = link {bg_keep_saveimage} (21번 pixelerate)")

    return bg_remove_path, bg_keep_path


if __name__ == "__main__":
    create_dual_workflows()
