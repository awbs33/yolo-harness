---
name: yolo-orchestrator
description: "YOLOv8 객체 탐지 파이프라인(전처리/추론/시각화/QA)을 에이전트 팀으로 조율하는 오케스트레이터. 'YOLOv8 파이프라인 구현', '객체 탐지 코드 작성', 'main.py 구현', 'python main.py 실행 준비', '파이프라인 구축', '전처리/추론/시각화 모듈 구현', 'QA 테스트 실행' 요청 시 반드시 이 스킬을 사용할 것. 파이프라인 수정, 재실행, 업데이트, 보완, 다시 실행, 특정 모듈만 수정, 이전 결과 개선 요청에도 사용."
---

# YOLOv8 파이프라인 오케스트레이터

YOLOv8 객체 탐지 파이프라인을 구성하는 4개 에이전트를 조율하여  
`python main.py --input ./images --output ./results`로 실행 가능한 코드베이스를 완성한다.

## 실행 모드: 에이전트 팀 (파이프라인 패턴)

## 에이전트 구성

| 팀원 | 에이전트 파일 | 역할 | 스킬 | 핵심 산출물 |
|------|-------------|------|------|------------|
| preprocessor | agents/preprocessor.md | 전처리 + main.py | preprocess-input | `src/preprocess.py`, `main.py` |
| yolo-inferencer | agents/yolo-inferencer.md | YOLOv8 추론 | yolo-inference | `src/inference.py`, `requirements.txt` |
| visualizer | agents/visualizer.md | 결과 시각화 | visualize-results | `src/visualize.py` |
| qa-tester | agents/qa-tester.md | QA 테스트 | qa-pipeline | `tests/*.py` |

## 워크플로우

### Phase 0: 컨텍스트 확인

1. `_workspace/` 디렉토리 존재 여부 확인
2. 실행 모드 결정:
   - **`_workspace/` 미존재** → 초기 실행 (Phase 1로 진행)
   - **`_workspace/` 존재 + 부분 수정 요청** → 부분 재실행
     - "전처리만" → preprocessor만 재호출
     - "추론만" → yolo-inferencer만
     - "시각화만" → visualizer만
     - "QA만" → qa-tester만
   - **`_workspace/` 존재 + 새 파라미터/입력** → 새 실행
     - 기존 `_workspace/`를 `_workspace_{YYYYMMDD_HHMMSS}/`로 이동 후 Phase 1 진행
3. 부분 재실행 시: 이전 산출물 경로를 에이전트 프롬프트에 포함

### Phase 1: 준비

1. 사용자 요청에서 파라미터 파악:
   - 입력 경로 (기본: `./images`)
   - 출력 경로 (기본: `./results`)
   - 모델명 (기본: `yolov8n.pt`)
   - 신뢰도 임계값 (기본: 0.25)

2. 작업 디렉토리 구조 생성:
   ```
   _workspace/
   ├── 01_preprocessed/
   ├── 02_predictions/
   └── 04_qa_report/
   src/
   tests/
   results/   ← 출력 경로
   ```

3. `src/__init__.py` 생성 (빈 파일, Python 패키지로 인식)

### Phase 2: 팀 구성 및 작업 등록

```
TeamCreate(
  team_name: "yolo-pipeline-team",
  members: [
    {
      name: "preprocessor",
      agent_type: "general-purpose",
      model: "opus",
      prompt: "당신은 preprocessor 에이전트입니다. .claude/agents/preprocessor.md와 .claude/skills/preprocess-input/SKILL.md를 읽고 src/preprocess.py와 main.py를 구현하세요. 완료 후 리더에게 알립니다."
    },
    {
      name: "yolo-inferencer",
      agent_type: "general-purpose",
      model: "opus",
      prompt: "당신은 yolo-inferencer 에이전트입니다. .claude/agents/yolo-inferencer.md와 .claude/skills/yolo-inference/SKILL.md를 읽고 src/inference.py와 requirements.txt를 구현하세요. preprocessor 완료 신호 후 시작합니다."
    },
    {
      name: "visualizer",
      agent_type: "general-purpose",
      model: "opus",
      prompt: "당신은 visualizer 에이전트입니다. .claude/agents/visualizer.md와 .claude/skills/visualize-results/SKILL.md를 읽고 src/visualize.py를 구현하세요. yolo-inferencer 완료 신호 후 시작합니다."
    },
    {
      name: "qa-tester",
      agent_type: "general-purpose",
      model: "opus",
      prompt: "당신은 qa-tester 에이전트입니다. .claude/agents/qa-tester.md와 .claude/skills/qa-pipeline/SKILL.md를 읽고 tests/ 디렉토리에 pytest 테스트를 작성하고 실행하세요. 모든 모듈 완성 확인 후 시작합니다."
    }
  ]
)
```

```
TaskCreate(tasks: [
  {
    title: "전처리 모듈 구현",
    assignee: "preprocessor",
    description: "src/preprocess.py 구현. preprocess-input 스킬 참조. letterbox 리사이즈, manifest.json 생성 포함. 완료 시 리더에게 SendMessage로 알림."
  },
  {
    title: "main.py 구현",
    assignee: "preprocessor",
    description: "argparse(--input, --output, --model, --conf, --iou)를 갖는 파이프라인 진입점. preprocess-input 스킬의 main.py 명세 참조. 완료 시 리더에게 알림."
  },
  {
    title: "추론 모듈 구현",
    assignee: "yolo-inferencer",
    description: "src/inference.py 구현. yolo-inference 스킬 참조. run_batch_inference가 manifest.json을 읽어 배치 처리.",
    depends_on: ["전처리 모듈 구현"]
  },
  {
    title: "requirements.txt 생성",
    assignee: "yolo-inferencer",
    description: "ultralytics>=8.0.0, opencv-python>=4.8.0, matplotlib>=3.7.0, numpy>=1.24.0, pytest>=7.4.0, pytest-mock>=3.11.0"
  },
  {
    title: "시각화 모듈 구현",
    assignee: "visualizer",
    description: "src/visualize.py 구현. visualize-results 스킬 참조. 황금각 팔레트, draw_detections, generate_summary_plot 포함.",
    depends_on: ["추론 모듈 구현"]
  },
  {
    title: "QA 테스트 작성 및 실행",
    assignee: "qa-tester",
    description: "tests/conftest.py, test_preprocess.py, test_inference.py, test_visualize.py, test_integration.py 작성. pytest 실행 후 _workspace/04_qa_report.md 생성.",
    depends_on: ["전처리 모듈 구현", "추론 모듈 구현", "시각화 모듈 구현", "main.py 구현"]
  }
])
```

### Phase 3: 구현 실행

**실행 방식:** 팀원들이 공유 작업 목록에서 자체 조율

**파이프라인 의존 순서:**
1. preprocessor: 전처리 모듈 + main.py 구현
2. yolo-inferencer: preprocessor 완료 후 추론 모듈 구현
3. visualizer: yolo-inferencer 완료 후 시각화 모듈 구현
4. qa-tester: 1~3 모두 완료 후 테스트 실행

**팀원 간 통신 규칙:**
- preprocessor → 리더: 전처리 완료, manifest 경로 알림
- yolo-inferencer → 리더: 추론 완료, predictions 경로 알림
- visualizer → 리더: 시각화 완료, 파일 수 보고
- qa-tester → 리더: 테스트 결과 요약 (통과/실패 수)

**산출물 저장:**

| 팀원 | 출력 파일 |
|------|----------|
| preprocessor | `src/preprocess.py`, `main.py`, `src/__init__.py` |
| yolo-inferencer | `src/inference.py`, `requirements.txt` |
| visualizer | `src/visualize.py` |
| qa-tester | `tests/conftest.py`, `tests/test_*.py`, `_workspace/04_qa_report.md` |

**리더 모니터링:**
- 각 모듈 완성 후 Python 구문 검사: `python -m py_compile {파일}`
- 의존성 위반(순서 역전) 발생 시 해당 에이전트에 대기 지시
- Phase 3 완료 조건: 전체 6개 태스크 Done 상태

### Phase 4: 통합 검증

리더가 직접 수행:

1. **파일 존재 확인**
   ```bash
   ls src/preprocess.py src/inference.py src/visualize.py main.py requirements.txt
   ls tests/conftest.py tests/test_preprocess.py tests/test_inference.py tests/test_visualize.py
   ```

2. **구문 검사**
   ```bash
   python -m py_compile src/preprocess.py src/inference.py src/visualize.py main.py
   ```

3. **CLI 기본 동작 확인**
   ```bash
   python main.py --help
   ```

4. **QA 리포트 확인**
   - `_workspace/04_qa_report.md` 읽기
   - 실패한 테스트 있으면 담당 에이전트 재호출 (1회)

### Phase 5: 완료 보고

사용자에게 보고할 내용:
- 생성된 파일 목록
- QA 결과 요약 (통과/실패 수)
- 실행 방법:
  ```bash
  pip install -r requirements.txt
  mkdir -p images results
  python main.py --input ./images --output ./results
  ```
- 피드백 요청: "결과에서 개선할 부분이 있나요?"

## 에러 핸들링

| 상황 | 처리 |
|------|------|
| 에이전트 구현 실패 | 1회 재호출. 재실패 시 오류와 함께 리포트, 나머지 진행 |
| 구문 오류 | 해당 에이전트에 오류 내용 포함하여 수정 요청 |
| QA 테스트 실패 | 실패 내용을 담당 에이전트에 전달, 수정 후 재테스트 |
| 팀 구성 실패 | 서브 에이전트 모드로 폴백 (Agent 도구 순차 호출) |

## 데이터 전달 전략
- **태스크 기반**: TaskCreate/TaskUpdate로 의존성과 진행 상태 관리
- **파일 기반**: `_workspace/`, `src/`, `tests/`로 산출물 교환
- **메시지 기반**: SendMessage로 모듈 완성 알림 및 인터페이스 공유

## 테스트 시나리오

**정상 흐름:**
```
입력: "YOLOv8 파이프라인 구현해줘"
기대: src/*.py, main.py, tests/*.py, requirements.txt 생성
      python main.py --help 정상 실행
      pytest 전체 통과 or 실패 리포트 생성
```

**에러 흐름:**
```
상황: ultralytics 미설치 환경
기대: requirements.txt 생성, inference.py에서 ImportError 시 안내 메시지
```

**부분 재실행:**
```
입력: "시각화 색상을 더 선명하게 수정해줘"
기대: visualizer만 재호출, src/visualize.py 수정
      나머지 파일 유지
```
