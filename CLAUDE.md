# CLAUDE.md — yolo-harness

## 하네스: YOLOv8 객체 탐지 파이프라인

**목표:** YOLOv8 기반 객체 탐지 추론 파이프라인 코드베이스를 에이전트 팀으로 구현한다.

**트리거:** YOLOv8 파이프라인 구현, 모듈 수정, 테스트 실행, `python main.py` 관련 작업 요청 시 `yolo-orchestrator` 스킬을 사용하라. 단순 질문은 직접 응답 가능.

**실행 명령:** `python main.py --input ./images --output ./results`

---

## 프로젝트 구조

```
yolo-harness/
├── src/
│   ├── preprocess.py   # 이미지/영상 전처리
│   ├── inference.py    # YOLOv8 추론
│   └── visualize.py    # 결과 시각화
├── tests/              # pytest 테스트 스위트
├── images/             # 입력 이미지 (git 미추적, 별도 공유)
├── results/            # 출력물 (git 미추적)
├── main.py             # 파이프라인 진입점
└── requirements.txt    # 의존성
```

> **모델 파일 (`*.pt`)은 git에 포함되지 않습니다.** 팀 공유 드라이브나 LFS를 사용하세요.

---

## Git 협업 가이드

### `git clone` vs `git pull` 차이

| 명령 | 시점 | 역할 |
|------|------|------|
| `git clone` | **최초 1회** | 저장소 전체를 처음 내려받음 |
| `git pull` | **작업 시작할 때마다** | 다른 팀원이 올린 변경사항을 내 로컬에 반영 |

> clone은 "처음 입사할 때", pull은 "매일 출근해서 어제 올라온 것 확인"하는 것과 같다.

---

### 신규 팀원 환경 세팅 절차

**1단계 — 사전 준비 (최초 1회)**

Git 설치 확인:
```bash
git --version
```

Git이 없다면 OS에 맞게 설치:

- **Windows**
  1. https://git-scm.com/download/win 접속
  2. 설치 파일 다운로드 후 실행 (기본 옵션으로 Next → Next → Install)
  3. 설치 완료 후 `Git Bash` 실행해서 사용

- **Mac**
  ```bash
  # Homebrew가 있는 경우
  brew install git

  # Homebrew가 없는 경우 — Xcode Command Line Tools 설치
  xcode-select --install
  ```

- **Linux (Ubuntu/Debian)**
  ```bash
  sudo apt update && sudo apt install git -y
  ```

설치 후 사용자 정보 등록 (최초 1회):
```bash
git config --global user.name "본인이름"
git config --global user.email "본인이메일@회사.com"
```

Python 설치 확인 (3.9 이상):
```bash
python --version
# 없으면: https://www.python.org/downloads 에서 설치
```

**SSH 키 생성 및 GitHub 등록**

```bash
# SSH 키 생성
ssh-keygen -t ed25519 -C "본인이메일@회사.com"

# 공개키 확인 후 복사
cat ~/.ssh/id_ed25519.pub
# → GitHub > Settings > SSH and GPG keys > New SSH key 에 붙여넣기

# 연결 테스트
ssh -T git@github.com
# Hi <본인의 GitHub 사용자명>! You've successfully authenticated...
# 위 메시지가 나오면 성공
```

**2단계 — 저장소 클론 (최초 1회)**

```bash
git clone git@github.com:awbs33/yolo-harness.git
cd yolo-harness
```

**3단계 — 가상환경 생성 및 의존성 설치**

```bash
python -m venv .venv
source .venv/bin/activate        # Mac/Linux
.venv\Scripts\activate           # Windows

pip install -r requirements.txt
```

**4단계 — 모델 파일 준비 (git에 없으므로 별도 수령)**

```bash
# 팀장에게 yolov8n.pt 파일을 받아 프로젝트 루트에 복사
cp /shared/models/yolov8n.pt ~/yolo-harness/
```

**5단계 — 테스트 이미지 준비**

```bash
cp /shared/data/images/* ~/yolo-harness/images/
```

**6단계 — 테스트 실행으로 환경 검증**

```bash
pytest tests/ -v
# 전체 통과 시 환경 세팅 완료 (테스트는 mock 기반으로 모델 파일 불필요)
```

**7단계 — 파이프라인 실행 확인**

```bash
python main.py --input ./images --output ./results
```

**8단계 — 개발 시작 전 브랜치 생성**

```bash
# 작업 시작 전 항상 최신 코드 반영
git pull origin main

# 작업용 브랜치 생성
git checkout -b feature/내작업이름
```

**세팅 체크리스트**

| 순서 | 항목 | 확인 |
|------|------|------|
| 1 | Git / Python 설치 | ☐ |
| 2 | SSH 키 생성 및 GitHub 등록 | ☐ |
| 3 | `git clone` | ☐ |
| 4 | 가상환경 생성 및 `pip install` | ☐ |
| 5 | `yolov8n.pt` 파일 수령 및 배치 | ☐ |
| 6 | 테스트 이미지 준비 | ☐ |
| 7 | `pytest tests/ -v` 전체 통과 | ☐ |
| 8 | `python main.py` 실행 확인 | ☐ |

### 브랜치 전략

| 브랜치 | 용도 |
|--------|------|
| `main` | 항상 실행 가능한 안정 버전 |
| `dev` | 통합 개발 브랜치 |
| `feature/<이름>` | 개별 기능 개발 |
| `fix/<이름>` | 버그 수정 |

```bash
# 새 기능 개발 시
git checkout dev
git pull origin dev
git checkout -b feature/inference-batch-size

# 작업 완료 후
git add src/inference.py tests/test_inference.py
git commit -m "feat(inference): 배치 크기 파라미터 추가"
git push origin feature/inference-batch-size
# → GitHub/GitLab에서 dev 브랜치로 PR 생성
```

### 커밋 메시지 규칙

```
<type>(<scope>): <한 줄 설명>
```

| type | 사용 시점 |
|------|----------|
| `feat` | 새 기능 |
| `fix` | 버그 수정 |
| `test` | 테스트 추가/수정 |
| `refactor` | 코드 개선 (동작 변경 없음) |
| `docs` | 문서 수정 |
| `chore` | 빌드/설정 변경 |

scope 예시: `preprocess`, `inference`, `visualize`, `pipeline`, `ci`

### PR 머지 기준

- `pytest` 전체 통과 필수: `pytest tests/ -v`
- 최소 1명 리뷰 승인
- `main` 직접 푸시 금지

### 충돌 해결 원칙

```bash
git fetch origin
git rebase origin/dev   # merge 대신 rebase 권장 (히스토리 선형 유지)
```

---

## 로컬 개발 규칙

- **git에 추가하면 안 되는 것:** `*.pt`, `results/`, `images/` 내 실제 데이터, `_workspace/`
- **공유 대상:** `src/`, `tests/`, `main.py`, `requirements.txt`, `CLAUDE.md`, `.gitignore`
- 테스트는 mock을 활용해 모델 파일 없이도 실행 가능해야 한다 (`tests/conftest.py` 참고)

---

## 변경 이력

| 날짜 | 변경 내용 | 대상 | 사유 |
|------|----------|------|------|
| 2026-04-26 | 초기 구성 | 전체 | YOLOv8 파이프라인 하네스 신규 구축 |
| 2026-04-28 | Git 협업 가이드 추가, .gitignore 생성 | CLAUDE.md, .gitignore | 팀 공동 작업 체계 구축 |
| 2026-05-08 | 신규 팀원 세팅 절차 및 clone/pull 차이 설명 추가 | CLAUDE.md | 공동 개발 온보딩 가이드 보완 |
| 2026-05-08 | OS별 Git 설치 방법 추가 | CLAUDE.md | Git 미설치 팀원 대응 |
