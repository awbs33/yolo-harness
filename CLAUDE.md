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

## Git 협업 전체 흐름

```
[팀장] 프로젝트 생성 → GitHub 등록 → push
          ↓
[팀원] clone → 환경 세팅 → 브랜치 생성 → 코드 수정 → push → PR 생성
          ↓
[팀장] 코드 리뷰 → 승인 → main 머지
```

---

## PHASE 1 — 팀장: 프로젝트 최초 생성 및 GitHub 등록

> 프로젝트를 처음 만들고 GitHub에 올리는 과정입니다. **최초 1회만** 진행합니다.

```bash
# 1. 프로젝트 폴더에서 git 초기화
git init
git branch -M main

# 2. GitHub에서 빈 저장소 생성 후 원격 등록 (SSH 방식)
git remote add origin git@github.com:awbs33/yolo-harness.git

# 3. 공유할 파일 스테이징
git add src/ tests/ main.py requirements.txt CLAUDE.md .gitignore

# 4. 최초 커밋
git commit -m "chore: 초기 프로젝트 구성"

# 5. GitHub에 push
git push -u origin main
```

**git에 올리면 안 되는 것**

| 항목 | 이유 |
|------|------|
| `*.pt` (모델 파일) | 용량이 크고 팀 공유 드라이브로 별도 관리 |
| `results/` | 파이프라인 출력물, 개인별 상이 |
| `images/` 실제 데이터 | 용량 큰 바이너리, 별도 공유 |
| `_workspace/` | 에이전트 로컬 작업 공간 |

---

## PHASE 2 — 팀원: 최초 환경 세팅 (입사 후 1회)

### Git 설치

```bash
# 설치 여부 확인
git --version
```

Git이 없다면 OS에 맞게 설치:

- **Windows:** https://git-scm.com/download/win → 설치 파일 실행 (기본 옵션) → `Git Bash` 사용
- **Mac:**
  ```bash
  brew install git          # Homebrew가 있는 경우
  xcode-select --install    # Homebrew가 없는 경우
  ```
- **Linux (Ubuntu/Debian):**
  ```bash
  sudo apt update && sudo apt install git -y
  ```

설치 후 사용자 정보 등록 (최초 1회):
```bash
git config --global user.name "본인이름"
git config --global user.email "본인이메일@회사.com"
```

### SSH 키 생성 및 GitHub 등록

```bash
# SSH 키 생성
ssh-keygen -t ed25519 -C "본인이메일@회사.com"

# 공개키 확인 후 복사
cat ~/.ssh/id_ed25519.pub
# → GitHub > Settings > SSH and GPG keys > New SSH key 에 붙여넣기

# 연결 테스트
ssh -T git@github.com
# Hi <본인의 GitHub 사용자명>! You've successfully authenticated...
```

### 저장소 클론 및 환경 구성

```bash
# 저장소 클론 (최초 1회)
git clone git@github.com:awbs33/yolo-harness.git
cd yolo-harness

# 가상환경 생성 및 활성화
python -m venv .venv
source .venv/bin/activate      # Mac/Linux
.venv\Scripts\activate         # Windows

# 의존성 설치
pip install -r requirements.txt

# 모델 파일 수령 (팀장에게 별도 전달받아 프로젝트 루트에 복사)
cp /shared/models/yolov8n.pt .

# 테스트 이미지 준비
cp /shared/data/images/* ./images/

# 환경 검증
pytest tests/ -v               # 전체 통과 확인
python main.py --input ./images --output ./results
```

**세팅 체크리스트**

| 순서 | 항목 | 확인 |
|------|------|------|
| 1 | Git / Python 설치 및 사용자 정보 등록 | ☐ |
| 2 | SSH 키 생성 및 GitHub 등록 | ☐ |
| 3 | `git clone` | ☐ |
| 4 | 가상환경 생성 및 `pip install` | ☐ |
| 5 | `yolov8n.pt` 파일 수령 및 배치 | ☐ |
| 6 | 테스트 이미지 준비 | ☐ |
| 7 | `pytest tests/ -v` 전체 통과 | ☐ |
| 8 | `python main.py` 실행 확인 | ☐ |

---

## PHASE 3 — 팀원: 코드 수정 및 PR 생성

> 매번 작업할 때마다 반복하는 흐름입니다.

```bash
# 1. 작업 전 최신 코드 반영 (매번 필수)
#    clone은 최초 1회, pull은 작업 시작할 때마다
git pull origin main

# 2. 작업용 브랜치 생성 (main에 직접 작업 금지)
git checkout -b feature/inference-add-batch

# 3. 코드 수정
#    src/inference.py 등 필요한 파일 수정

# 4. 변경 내용 확인
git status                        # 어떤 파일이 바뀌었는지
git diff src/inference.py         # 줄 단위 변경 내용

# 5. 테스트 실행 (커밋 전 필수 — 실패 상태로 커밋 금지)
pytest tests/ -v

# 6. 스테이징
git add src/inference.py
git add src/inference.py tests/test_inference.py   # 테스트도 수정했다면 함께

# 7. 커밋
git commit -m "feat(inference): 배치 처리 함수 추가"

# 8. GitHub에 push
git push origin feature/inference-add-batch
```

push 후 GitHub에서 `feature/inference-add-batch` → `main` 으로 **Pull Request(PR)** 를 생성합니다.

---

## PHASE 4 — 팀장: 코드 리뷰 및 머지

### GitHub 웹에서 리뷰 (일반적인 경우)

1. GitHub → `Pull requests` 탭에서 팀원이 올린 PR 선택
2. `Files changed` 탭에서 변경된 코드 줄 단위로 확인
3. 특정 줄에 코멘트 작성 가능 (`+` 버튼 클릭)
4. `Review changes` → 문제 없으면 `Approve` → `Submit review`
5. `Merge pull request` → `Confirm merge`

### 로컬에서 직접 실행 후 리뷰 (동작 확인이 필요한 경우)

```bash
# 팀원 브랜치를 로컬로 가져와서 테스트
git fetch origin
git checkout feature/inference-add-batch

pytest tests/ -v
python main.py --input ./images --output ./results

# 확인 후 GitHub으로 돌아가 Approve → Merge
```

---

## 커밋 메시지 규칙

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

---

## 브랜치 전략

| 브랜치 | 용도 |
|--------|------|
| `main` | 항상 실행 가능한 안정 버전 — 직접 push 금지 |
| `feature/<이름>` | 새 기능 개발 |
| `fix/<이름>` | 버그 수정 |

---

## 로컬 개발 규칙

- 테스트는 mock을 활용해 모델 파일 없이도 실행 가능해야 한다 (`tests/conftest.py` 참고)
- 충돌 발생 시 `merge` 대신 `rebase` 권장 (히스토리 선형 유지)
  ```bash
  git fetch origin
  git rebase origin/main
  ```

---

## 변경 이력

| 날짜 | 변경 내용 | 대상 | 사유 |
|------|----------|------|------|
| 2026-04-26 | 초기 구성 | 전체 | YOLOv8 파이프라인 하네스 신규 구축 |
| 2026-04-28 | Git 협업 가이드 추가, .gitignore 생성 | CLAUDE.md, .gitignore | 팀 공동 작업 체계 구축 |
| 2026-05-08 | Git 협업 가이드 전면 재작성 | CLAUDE.md | 팀장→팀원→리뷰→머지 전체 흐름으로 통합, 중복 제거 |
