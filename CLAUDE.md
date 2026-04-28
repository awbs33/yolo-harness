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

### 초기 설정 (신규 팀원)

```bash
git clone <repository-url> yolo-harness
cd yolo-harness
pip install -r requirements.txt
# 모델 파일을 별도 공유 경로에서 다운로드
cp /shared/models/yolov8n.pt .
# 테스트 이미지 준비
cp /shared/data/images/* ./images/
```

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
