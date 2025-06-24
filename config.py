import os
from pathlib import Path
from dotenv import load_dotenv
import secrets

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
BASE_DIR: Path = Path(__file__).resolve().parent  # server/
PROJECT_ROOT: Path = BASE_DIR  # repository root

# ---------------------------------------------------------------------------
# .env.local 로부터 환경변수 로드
# ---------------------------------------------------------------------------
ENV_PATH = PROJECT_ROOT / ".env.local"
if ENV_PATH.exists():
    load_dotenv(ENV_PATH)
else:
    # 개발 초기엔 .env.local 이 없을 수도 있으므로 경고만 출력
    print("[config] .env.local not found – falling back to os environment only.")

# ---------------------------------------------------------------------------
# 필수 환경변수
# ---------------------------------------------------------------------------
GOOGLE_API_KEY: str | None = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise RuntimeError(
        "[config] GOOGLE_API_KEY is missing. Add it to .env.local or system env."
    )

RD_TOKEN: str | None = os.getenv("RD_TOKEN")
if not GOOGLE_API_KEY:
    raise RuntimeError(
        "[config] RD_TOKEN is missing. Add it to .env.local or system env."
    )

API_SECRET_KEY = os.getenv("API_SECRET_KEY")
if not API_SECRET_KEY:
    raise RuntimeError(
        "[config] API_SECRET_KEY is missing. Add it to .env.local or system env."
    )

# ---------------------------------------------------------------------------
# Hugging Face 캐시 경로 (선택)
# ---------------------------------------------------------------------------
HF_HOME: str = os.getenv("HF_HOME", "E:/Huggingface_Cache")
os.environ["HF_HOME"] = HF_HOME  # 하위 라이브러리에서 즉시 인식하도록 보장

# ---------------------------------------------------------------------------
# Instruction 파일 경로
# ---------------------------------------------------------------------------
INSTR_DIR: Path = PROJECT_ROOT
ENHANCER_PROMPT: str = (INSTR_DIR / "prompt_enhancer.md").read_text(encoding="utf-8")
REACTION_PROMPT: str = (INSTR_DIR / "reaction_system.md").read_text(encoding="utf-8")

# ---------------------------------------------------------------------------
# FastAPI 설정 값 (예: CORS origin)
# ---------------------------------------------------------------------------
ALLOWED_ORIGINS: list[str] = ["*"]  # 프로덕션에서는 구체 도메인으로 제한

# ---------------------------------------------------------------------------
# Gemini 모델 이름 모음
# ---------------------------------------------------------------------------
GEMINI_MODEL_FLASH = "gemini-2.5-flash-preview-05-20"
GEMINI_MAX_TOKENS = 500
