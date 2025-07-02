# test_comfyui_connection.py
import asyncio
import json
import base64
import os
from datetime import datetime
from workflow_manager import WorkflowManager
from comfyui_client import ComfyUIClient


async def test_basic_connection():
    """기본 연결 테스트"""
    print("=== ComfyUI 기본 연결 테스트 ===")

    # 1. ComfyUI 클라이언트 생성
    client = ComfyUIClient("127.0.0.1:8188")

    # 2. 헬스 체크
    print("1. 헬스 체크 중...")
    is_healthy = await client.health_check()
    print(f"   ComfyUI 상태: {'정상' if is_healthy else '연결 실패'}")

    if not is_healthy:
        print("❌ ComfyUI가 실행되지 않았거나 연결할 수 없습니다.")
        print("   run_nvidia_gpu_fast_fp16_accumulation.bat을 실행했는지 확인하세요.")
        return False

    # 3. 큐 정보 확인
    print("2. 큐 정보 확인 중...")
    try:
        queue_info = await client.get_queue_info()
        print(f"   실행 중: {queue_info['running']}")
        print(f"   대기 중: {queue_info['pending']}")
        print(f"   총계: {queue_info['total']}")
    except Exception as e:
        print(f"   큐 정보 확인 실패: {e}")

    print("✅ 기본 연결 테스트 완료!")
    return True


async def test_workflow_loading():
    """워크플로우 로딩 테스트"""
    print("\n=== 워크플로우 로딩 테스트 ===")

    try:
        # 1. WorkflowManager 생성
        print("1. 워크플로우 파일 로딩 중...")
        wm = WorkflowManager("./workflows/pixel_art_server.json")
        print("   ✅ 워크플로우 파일 로딩 성공!")

        # 2. 워크플로우 구조 확인
        print("2. 워크플로우 구조 분석 중...")

        # 최상위 키들 확인
        top_keys = list(wm.base_workflow.keys())
        print(f"   최상위 키들: {top_keys}")

        # nodes 키가 있는지 확인
        if "nodes" in wm.base_workflow:
            nodes = wm.base_workflow["nodes"]
            print(f"   노드 개수: {len(nodes)}")

            # 각 노드의 ID 확인 (자동 감지)
            node_ids = []
            for node in nodes:
                if "id" in node:
                    node_ids.append(str(node["id"]))

            print(f"   노드 ID들: {sorted(node_ids)}")

            # 중요한 노드들 확인
            target_node_ids = [
                "32",
                "20",
                "34",
            ]  # CLIPTextEncode, Target Size, SaveImage
            for target_id in target_node_ids:
                if target_id in node_ids:
                    print(f"   ✅ 노드 {target_id} 존재함")
                else:
                    print(f"   ❌ 노드 {target_id} 없음")
        else:
            # nodes 키가 없다면 직접 노드 구조 확인
            print("   'nodes' 키가 없음. 직접 노드 구조 확인 중...")
            if "32" in wm.base_workflow:
                print("   ✅ 노드 32 존재함 (직접 접근)")
            else:
                print("   ❌ 노드 32 없음")

            if "20" in wm.base_workflow:
                print("   ✅ 노드 20 존재함 (직접 접근)")
            else:
                print("   ❌ 노드 20 없음")

        # 3. 워크플로우 파라미터 주입 테스트
        print("3. 워크플로우 파라미터 주입 테스트...")
        test_prompt = "a cute cat"
        test_target_size = 64

        workflow = wm.prepare_workflow(prompt=test_prompt, target_size=test_target_size)

        # 4. 결과 확인
        print("4. 파라미터 주입 결과 확인...")

        # 프롬프트 확인 시도
        try:
            if "nodes" in workflow:
                # nodes 배열에서 찾기
                prompt_set = False
                for node in workflow["nodes"]:
                    if node.get("id") == 32:
                        actual_prompt = node["widgets_values"][0]
                        print(f"   프롬프트 설정 확인: {actual_prompt}")
                        if actual_prompt == test_prompt:
                            print("   ✅ 프롬프트 주입 성공!")
                            prompt_set = True
                        break

                if not prompt_set:
                    print("   ❌ 프롬프트 주입 실패!")
            else:
                # 직접 접근
                actual_prompt = workflow["32"]["widgets_values"][0]
                print(f"   프롬프트 설정 확인: {actual_prompt}")
                if actual_prompt == test_prompt:
                    print("   ✅ 프롬프트 주입 성공!")
                else:
                    print("   ❌ 프롬프트 주입 실패!")
        except Exception as e:
            print(f"   ❌ 프롬프트 확인 실패: {e}")

        # 타겟 크기 확인 시도
        try:
            if "nodes" in workflow:
                # nodes 배열에서 찾기
                size_set = False
                for node in workflow["nodes"]:
                    if node.get("id") == 20:
                        actual_size = node["widgets_values"][0]
                        print(f"   타겟 크기 설정 확인: {actual_size}")
                        if actual_size == str(test_target_size):
                            print("   ✅ 타겟 크기 주입 성공!")
                            size_set = True
                        break

                if not size_set:
                    print("   ❌ 타겟 크기 주입 실패!")
            else:
                # 직접 접근
                actual_size = workflow["20"]["widgets_values"][0]
                print(f"   타겟 크기 설정 확인: {actual_size}")
                if actual_size == str(test_target_size):
                    print("   ✅ 타겟 크기 주입 성공!")
                else:
                    print("   ❌ 타겟 크기 주입 실패!")
        except Exception as e:
            print(f"   ❌ 타겟 크기 확인 실패: {e}")

        print("✅ 워크플로우 로딩 테스트 완료!")
        return True

    except FileNotFoundError:
        print("❌ 워크플로우 파일을 찾을 수 없습니다.")
        print("   ./workflows/pixel_art_server.json 파일이 있는지 확인하세요.")
        return False
    except Exception as e:
        print(f"❌ 워크플로우 로딩 실패: {e}")
        print(f"   에러 타입: {type(e)}")
        import traceback

        print(f"   상세 에러: {traceback.format_exc()}")
        return False


async def test_image_generation():
    """실제 이미지 생성 테스트"""
    print("\n=== 실제 이미지 생성 테스트 ===")

    try:
        # 1. 준비
        print("1. 클라이언트 및 워크플로우 준비 중...")
        client = ComfyUIClient("127.0.0.1:8188")
        wm = WorkflowManager("./workflows/pixel_art_server.json")

        # 2. 워크플로우 생성
        test_prompt = "a simple red circle"
        target_size = 64

        workflow = wm.prepare_workflow(prompt=test_prompt, target_size=target_size)

        print("2. 이미지 생성 시작... (시간이 좀 걸릴 수 있습니다)")
        print(f"   프롬프트: '{test_prompt}'")
        print(f"   타겟 크기: {target_size}")

        # 3. API 형식 변환 테스트
        print("3. API 형식 변환 테스트...")
        api_workflow = client._convert_to_api_format(workflow)

        print(f"   변환된 노드 개수: {len(api_workflow)}")
        print(f"   노드 ID들: {list(api_workflow.keys())}")

        # 특정 노드들 확인 (Reroute 노드들 제거됨)
        important_nodes = [
            "32",
            "20",
            "34",
        ]  # CLIPTextEncode, Target Size, SaveImage
        for node_id in important_nodes:
            if node_id in api_workflow:
                print(f"   노드 {node_id}: {api_workflow[node_id]['class_type']}")
            else:
                print(f"   ❌ 노드 {node_id} 누락!")

        # 4. API 형식을 파일로 저장 (디버깅용)
        import json

        with open("debug_api_workflow.json", "w", encoding="utf-8") as f:
            json.dump(api_workflow, f, indent=2, ensure_ascii=False)
        print("   API 형식을 debug_api_workflow.json에 저장했습니다.")

        # 5. 이미지 생성 시도
        result = await client.generate_image(workflow, timeout=120)

        print(f"6. 이미지 생성 완료!")
        print(f"   Base64 길이: {len(result)} 문자")
        print(f"   Base64 시작: {result[:50]}...")

        # 7. 이미지 저장
        await save_test_image(result, test_prompt, target_size)

        print("✅ 실제 이미지 생성 테스트 완료!")
        return True

    except Exception as e:
        print(f"❌ 이미지 생성 실패: {e}")
        return False


async def save_test_image(base64_data: str, prompt: str, target_size: int):
    """생성된 이미지를 파일로 저장"""
    try:
        # test_images 폴더 생성
        if not os.path.exists("test_images"):
            os.makedirs("test_images")

        # 파일명 생성 (타임스탬프 포함)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_prompt = "".join(
            c for c in prompt if c.isalnum() or c in (" ", "-", "_")
        ).rstrip()
        safe_prompt = safe_prompt.replace(" ", "_")[:20]  # 20자 제한

        filename = f"test_{timestamp}_{safe_prompt}_{target_size}.png"
        filepath = os.path.join("test_images", filename)

        # Base64 디코딩 및 파일 저장
        image_data = base64.b64decode(base64_data)

        with open(filepath, "wb") as f:
            f.write(image_data)

        print(f"4. 이미지 저장 완료!")
        print(f"   저장 경로: {filepath}")
        print(f"   파일 크기: {len(image_data)} bytes")

        # 현재 디렉토리의 절대 경로 표시
        abs_path = os.path.abspath(filepath)
        print(f"   절대 경로: {abs_path}")

        return True

    except Exception as e:
        print(f"   ⚠️ 이미지 저장 실패: {e}")
        return False


async def test_multiple_images():
    """여러 이미지 생성 테스트"""
    print("\n=== 다양한 프롬프트로 이미지 생성 테스트 ===")

    test_cases = [
        {"prompt": "a red apple", "target_size": 64},
        {"prompt": "a blue cat", "target_size": 128},
        {"prompt": "a green tree", "target_size": 64},
    ]

    client = ComfyUIClient("127.0.0.1:8188")
    wm = WorkflowManager("./workflows/pixel_art_server.json")

    successful = 0
    total = len(test_cases)

    for i, test_case in enumerate(test_cases, 1):
        try:
            print(f"\n{i}/{total}. 생성 중: '{test_case['prompt']}'")

            workflow = wm.prepare_workflow(
                prompt=test_case["prompt"], target_size=test_case["target_size"]
            )

            result = await client.generate_image(workflow, timeout=120)
            await save_test_image(result, test_case["prompt"], test_case["target_size"])

            successful += 1
            print(f"   ✅ 성공!")

        except Exception as e:
            print(f"   ❌ 실패: {e}")

    print(f"\n📊 결과: {successful}/{total} 성공")
    return successful == total


async def main():
    """전체 테스트 실행"""
    print("ComfyUI 연동 테스트를 시작합니다...\n")

    # 1. 기본 연결 테스트
    connection_ok = await test_basic_connection()
    if not connection_ok:
        return

    # 2. 워크플로우 로딩 테스트
    workflow_ok = await test_workflow_loading()
    if not workflow_ok:
        return

    # 3. 실제 이미지 생성 테스트
    print("\n실제 이미지 생성 테스트를 진행할까요? (y/n): ", end="")
    user_input = input().strip().lower()

    if user_input == "y":
        generation_ok = await test_image_generation()
        if generation_ok:
            print("\n추가로 여러 이미지 생성 테스트를 해볼까요? (y/n): ", end="")
            multi_input = input().strip().lower()

            if multi_input == "y":
                multi_ok = await test_multiple_images()
                if multi_ok:
                    print("\n🎉 모든 테스트가 성공적으로 완료되었습니다!")
                else:
                    print("\n⚠️ 일부 이미지 생성에서 문제가 발생했습니다.")
            else:
                print("\n🎉 모든 테스트가 성공적으로 완료되었습니다!")
        else:
            print("\n⚠️ 이미지 생성 테스트에서 문제가 발생했습니다.")
    else:
        print("\n✅ 기본 연동 테스트가 완료되었습니다!")

    print(f"\n📁 생성된 이미지는 ./test_images/ 폴더에서 확인할 수 있습니다.")
    print("다음 단계: comfyui_load_balancer.py 생성 및 server.py 통합")


if __name__ == "__main__":
    asyncio.run(main())
