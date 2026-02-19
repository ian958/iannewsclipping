"""
Phase 4: 스코어링 시스템
AI 에듀테크 뉴스클리핑 자동화 시스템

A. 프롬프트 v1.0 (Naive)
B. 프롬프트 v2.0 (Structured - 최종 채택)
C. Subagent 실행 프로세스 설명

Subagent가 결정하는 것:
  - 각 뉴스별 1~10점 점수
  - 점수 분포 분석 후 최적 필터링 임계점 제안
  - 임계점별 예상 출력 건수 제시

주의: API 호출 없음, Subagent = Claude.ai (무료 플랜)
"""

import json


# ============================================================
# 프롬프트 v1.0: Naive (비교용 실패 사례)
# ============================================================

PROMPT_V1_NAIVE = """
다음 뉴스가 AI 챗봇 에듀테크 회사와 관련있는지 1-10점으로 평가해주세요.

뉴스 제목: {title}

점수:
"""

# v1.0 문제점:
#   - 회사 컨텍스트 없음 → 평가 기준 불명확
#   - 단일 차원 평가 → 세분화 부족
#   - 출력 형식 미정의 → 파싱 어려움
#   - 필터링 임계점 미지정 → 사람이 다시 결정해야 함


# ============================================================
# 프롬프트 v2.0: Structured (최종 채택)
# ============================================================

PROMPT_V2_STRUCTURED = """
당신은 AI 챗봇 에듀테크 회사의 전략 분석가입니다.

[회사 컨텍스트]
- 주력 제품: 대학용 AI 학생 상담 챗봇
- 핵심 기술: LLM, RAG, 대화형 AI
- 타깃 고객: 국내 4년제 대학교
- 주요 기능: 학사 상담, 수강신청 지원, 졸업요건 안내
- 경쟁 환경: 에듀테크 스타트업, 대기업 AI 솔루션 부문

[평가 관점]
다음 세 가지 관점에서 종합 평가하세요:

1. 산업 트렌드 연관성 (가중치 40%)
   - AI 기술, LLM, 챗봇, 대화형 AI 관련 동향
   - 에듀테크 시장 변화, 경쟁사 동향

2. 제품/서비스 연관성 (가중치 40%)
   - 대학 학사 업무, 수강신청, 상담 자동화
   - 대학교 AI 도입 사례, 교육 디지털 전환

3. 정책/제도 영향도 (가중치 20%)
   - 교육부 정책, 대학혁신지원사업
   - 글로컬대학, 무전공 제도 등 제도 변화

[스코어링 기준]
10점: 핵심 사업에 직접적으로 영향 (예: 대학 AI 챗봇 도입 사례)
8~9점: 높은 연관성 (예: AI 학사 시스템 구축, LLM 에듀테크 적용)
6~7점: 중간 연관성 (예: 에듀테크 시장 동향, 교육 AI 일반)
4~5점: 낮은 연관성 (예: 교육부 일반 정책, 대학 기사)
1~3점: 거의 무관 (예: 초중등 교육, 해외 사례만)

[중요] 전체 점수 분포를 분석한 후 최적의 필터링 임계점을 제안하세요.

[입력 데이터]
{news_json}

[출력 형식 - JSON]
{{
  "scored_news": [
    {{
      "id": "뉴스 ID",
      "title": "뉴스 제목",
      "industry_score": 0,
      "product_score": 0,
      "policy_score": 0,
      "final_score": 0,
      "reasoning": "평가 근거 1~2문장"
    }}
  ],
  "score_distribution": {{
    "mean": 0.0,
    "median": 0.0,
    "std": 0.0,
    "percentiles": {{
      "25": 0.0,
      "50": 0.0,
      "75": 0.0,
      "90": 0.0
    }}
  }},
  "recommendation": {{
    "suggested_threshold": 0,
    "reasoning": "점수 분포를 분석한 결과 X점을 임계값으로 제안합니다. 이유는...",
    "expected_output_count": 0,
    "expected_compression_rate": "X%"
  }}
}}
"""


def build_scoring_prompt_v2(news_list):
    """
    뉴스 리스트를 받아 v2.0 스코어링 프롬프트 생성
    (실제 Claude.ai에 복붙하여 사용)
    """
    news_json = json.dumps(
        [{"id": n["id"], "title": n["title"]} for n in news_list],
        ensure_ascii=False,
        indent=2,
    )
    return PROMPT_V2_STRUCTURED.format(news_json=news_json)


# ============================================================
# Subagent 실행 프로세스 설명
# ============================================================

APPROACH_C_PROCESS = """
=== Subagent 실행 프로세스 (최종 채택) ===

[사람의 역할]
  1. 클러스터링된 뉴스 CSV를 Claude.ai에 업로드
  2. v2.0 스코어링 프롬프트 입력 (회사 컨텍스트 포함)
  3. Subagent가 제안한 임계점 검토 및 승인/조정
  4. 최종 필터링 실행 (임계점 이상만 선별)

[Subagent (Claude)가 결정하는 것]
  - 각 뉴스의 1~10점 점수 (산업/제품/정책 세부 + 종합)
  - 점수 분포 통계 (평균, 중앙값, 표준편차, 백분위수)
  - 최적 임계점 제안 (예: "7점 이상 추천")
  - 임계점별 예상 결과 건수
  - 압축률 계산 및 제시

[사람이 최종 결정하는 것]
  - 임계점 승인 (Subagent 제안 그대로) 또는 조정
  - 특이 케이스 수동 처리 (점수 경계선 근처 기사)

[선택 이유 - v1.0 대비 v2.0 개선점]
  v1.0 문제: 기준 불명확, 파싱 어려움, 임계점 미결정
  v2.0 개선:
    - 회사 컨텍스트 명시 → 일관된 평가 기준
    - 3가지 관점 + 가중치 → 세분화된 평가
    - JSON 출력 형식 → 자동 파싱 가능
    - 점수 분포 분석 + 임계점 자동 제안 → Subagent 자율성 ✅

[메트릭 (빈 템플릿 - 측정 후 기입)]
  사람 작업 시간:
    - 파일 업로드: [측정 필요]분
    - 프롬프트 작성: [측정 필요]분
    - 제안 검토: [측정 필요]분
    - 필터링 실행: [측정 필요]분
  Subagent 실행 시간: [측정 필요]분
  총 소요 시간: [계산됨]분
  API 비용: $0 (Claude.ai 사용)
  처리 건수: [입력]건 → [출력]건
  정확도: [샘플 평가 필요]%

  Subagent가 결정한 파라미터:
    - 제안 임계점: [Subagent 결정]점
    - 예상 출력 건수: [Subagent 계산]건
    - 예상 압축률: [Subagent 계산]%
"""


def run_subagent_process():
    print(APPROACH_C_PROCESS)


# ============================================================
# 스코어링 결과 파싱 유틸리티
# ============================================================

def parse_scoring_result(raw_json_str):
    """
    Subagent(Claude)가 반환한 JSON 문자열을 파싱합니다.
    마크다운 코드블록 래핑 자동 제거.
    """
    # 코드블록 제거
    clean = raw_json_str.strip()
    if clean.startswith("```"):
        lines = clean.split("\n")
        start = next((i for i, l in enumerate(lines) if l.startswith("```") and i > 0), 1)
        end = next((i for i in range(len(lines) - 1, 0, -1) if lines[i].startswith("```")), len(lines) - 1)
        clean = "\n".join(lines[1:end])

    data = json.loads(clean)

    # 필드 검증
    assert "scored_news" in data, "scored_news 필드 없음"
    assert "score_distribution" in data, "score_distribution 필드 없음"
    assert "recommendation" in data, "recommendation 필드 없음"

    threshold = data["recommendation"]["suggested_threshold"]
    filtered = [n for n in data["scored_news"] if n["final_score"] >= threshold]

    print(f"[파싱 완료]")
    print(f"  전체 뉴스: {len(data['scored_news'])}건")
    print(f"  제안 임계점: {threshold}점")
    print(f"  필터링 후: {len(filtered)}건")
    print(f"  압축률: {data['recommendation']['expected_compression_rate']}")

    return data, filtered


def apply_threshold_filter(scoring_data, threshold=None):
    """
    Subagent 제안 임계점(또는 사람이 조정한 임계점)으로 필터링
    threshold=None이면 Subagent 제안값 자동 사용
    """
    if threshold is None:
        threshold = scoring_data["recommendation"]["suggested_threshold"]
        print(f"  Subagent 제안 임계점 사용: {threshold}점")
    else:
        print(f"  사람이 조정한 임계점 사용: {threshold}점")

    filtered = [
        n for n in scoring_data["scored_news"]
        if n["final_score"] >= threshold
    ]

    return filtered, threshold


# ============================================================
# 메인
# ============================================================

if __name__ == "__main__":
    print("=== Phase 4: 스코어링 시스템 ===\n")

    # 프롬프트 v1.0 출력
    print("--- 프롬프트 v1.0 (Naive) ---")
    print(PROMPT_V1_NAIVE)
    print("v1.0 문제점:")
    print("  - 회사 컨텍스트 없음")
    print("  - 단일 차원 평가")
    print("  - 출력 형식 미정의")
    print("  - 필터링 임계점 미지정\n")

    # 프롬프트 v2.0 샘플 생성
    print("--- 프롬프트 v2.0 (Structured) ---")
    sample_news = [
        {"id": "1", "title": "서울대 ChatGPT 학생상담 시범운영"},
        {"id": "2", "title": "교육부 대학 AI 가이드라인 발표"},
        {"id": "3", "title": "글로컬대학 30 최종 선정"},
    ]
    prompt = build_scoring_prompt_v2(sample_news)
    print("[생성된 프롬프트 미리보기 (앞 500자)]")
    print(prompt[:500] + "...\n")

    # Subagent 프로세스 설명
    print("--- Subagent 실행 프로세스 ---")
    run_subagent_process()

    # 파싱 유틸리티 사용 예시
    print("--- parse_scoring_result 사용법 ---")
    print("  raw = open('subagent_output.json').read()")
    print("  data, filtered = parse_scoring_result(raw)")
    print("  filtered_final, threshold = apply_threshold_filter(data)")
    print("  # threshold=None이면 Subagent 제안값 자동 사용")
    print("  # threshold=7이면 사람이 7점으로 조정")
