"""
샘플 데이터 구조 정의
AI 에듀테크 뉴스클리핑 자동화 시스템

각 Phase의 입출력 데이터 스키마와 샘플 데이터를 정의합니다.
실제 값은 측정/실행 후 기입하세요 (None = 미측정).
"""

import json
import csv
import io
from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict, Any


# ============================================================
# Phase 2 입력: 빅카인즈 CSV 스키마
# ============================================================

BIGKINDS_CSV_SAMPLE = """\
id,title,date,source,url
1,서울대 ChatGPT 학생상담 시범운영,2024-01-15,연합뉴스,https://...
2,교육부 대학 AI 가이드라인 발표,2024-01-16,한국경제,https://...
3,글로컬대학30 최종 선정 결과 공개,2024-01-17,조선일보,https://...
4,에듀테크 스타트업 투자 급증,2024-01-18,매일경제,https://...
5,무전공 입학 확대 추진 발표,2024-01-19,중앙일보,https://...
"""

# CSV 필드 정의
BIGKINDS_FIELDS = {
    "id": "뉴스 고유 ID (빅카인즈 부여)",
    "title": "뉴스 제목",
    "date": "발행일 (YYYY-MM-DD)",
    "source": "언론사명",
    "url": "원문 링크",
}


# ============================================================
# Phase 3 출력: 클러스터링 결과 JSON 스키마
# ============================================================

@dataclass
class ClusterItem:
    cluster_id: int
    representative_id: Optional[str] = None          # Subagent가 선정한 대표 기사 ID
    representative_title: Optional[str] = None
    member_ids: List[str] = field(default_factory=list)
    size: Optional[int] = None
    avg_similarity: Optional[float] = None            # Subagent가 계산한 클러스터 내 평균 유사도


@dataclass
class ClusteringMetadata:
    total_input: Optional[int] = None                 # 입력 뉴스 건수
    total_clusters: Optional[int] = None              # 최종 클러스터 수 (Subagent 결정)
    similarity_threshold: Optional[float] = None      # Subagent가 결정한 유사도 임계값
    threshold_reasoning: Optional[str] = None         # Subagent가 제공한 결정 근거
    excluded_small_clusters: Optional[int] = None     # 제외한 소규모 클러스터 수 (Subagent 결정)
    compression_rate: Optional[float] = None          # 압축률 (자동 계산)


@dataclass
class ClusteringResult:
    metadata: ClusteringMetadata = field(default_factory=ClusteringMetadata)
    clusters: List[ClusterItem] = field(default_factory=list)

    def to_dict(self):
        return {
            "metadata": asdict(self.metadata),
            "clusters": [asdict(c) for c in self.clusters],
        }

    def to_json(self, indent=2):
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)


CLUSTERING_RESULT_TEMPLATE = {
    "metadata": {
        "total_input": None,                  # 측정 필요
        "total_clusters": None,               # Subagent 결정
        "similarity_threshold": None,         # Subagent 결정 (예: 0.80)
        "threshold_reasoning": "Subagent가 제공한 이유",
        "excluded_small_clusters": None,      # Subagent 결정
        "compression_rate": None,             # 자동 계산 (%)
    },
    "clusters": [
        {
            "cluster_id": 1,
            "representative_id": None,        # Subagent 선정
            "representative_title": None,
            "member_ids": [],                 # Subagent 그룹핑
            "size": None,
            "avg_similarity": None,           # Subagent 계산
        }
    ],
}


# ============================================================
# Phase 4 출력: 스코어링 결과 JSON 스키마
# ============================================================

@dataclass
class NewsScore:
    id: Optional[str] = None
    title: Optional[str] = None
    industry_score: Optional[int] = None     # 산업 트렌드 연관성 (1~10)
    product_score: Optional[int] = None      # 제품/서비스 연관성 (1~10)
    policy_score: Optional[int] = None       # 정책/제도 영향도 (1~10)
    final_score: Optional[float] = None      # 가중 합산 (40%+40%+20%)
    reasoning: Optional[str] = None          # Subagent 평가 근거


@dataclass
class ScoreDistribution:
    mean: Optional[float] = None
    median: Optional[float] = None
    std: Optional[float] = None
    p25: Optional[float] = None
    p50: Optional[float] = None
    p75: Optional[float] = None
    p90: Optional[float] = None


@dataclass
class ScoringRecommendation:
    suggested_threshold: Optional[int] = None    # Subagent 제안 임계점
    reasoning: Optional[str] = None              # Subagent 제안 근거
    expected_output_count: Optional[int] = None  # 예상 출력 건수
    expected_compression_rate: Optional[str] = None  # 예상 압축률


@dataclass
class ScoringResult:
    scored_news: List[NewsScore] = field(default_factory=list)
    score_distribution: ScoreDistribution = field(default_factory=ScoreDistribution)
    recommendation: ScoringRecommendation = field(default_factory=ScoringRecommendation)

    def to_dict(self):
        d = asdict(self)
        # percentiles 형식 정리
        dist = d["score_distribution"]
        d["score_distribution"] = {
            "mean": dist["mean"],
            "median": dist["median"],
            "std": dist["std"],
            "percentiles": {
                "25": dist["p25"],
                "50": dist["p50"],
                "75": dist["p75"],
                "90": dist["p90"],
            },
        }
        return d

    def to_json(self, indent=2):
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)


SCORING_RESULT_TEMPLATE = {
    "scored_news": [
        {
            "id": None,
            "title": "",
            "scores": {
                "industry": None,             # Subagent 결정
                "product": None,              # Subagent 결정
                "policy": None,               # Subagent 결정
                "final": None,                # Subagent 계산 (가중 합산)
            },
            "reasoning": "",                  # Subagent 제공
        }
    ],
    "distribution": {
        "mean": None,                         # Subagent 계산
        "median": None,
        "std": None,
        "percentiles": {
            "25": None,
            "50": None,
            "75": None,
            "90": None,
        },
    },
    "recommendation": {
        "suggested_threshold": None,          # Subagent 자율 결정
        "reasoning": "",                      # Subagent 설명
        "expected_output": None,              # Subagent 계산
        "compression_rate": None,             # Subagent 계산
    },
}


# ============================================================
# Phase 5 출력: 요약/분류 결과 스키마
# ============================================================

CATEGORIES = [
    "AI/기술 동향",
    "대학/교육기관 동향",
    "정책/제도",
    "시장/경쟁사",
]

@dataclass
class SummarizedNews:
    id: Optional[str] = None
    title: Optional[str] = None
    summary: Optional[str] = None           # Subagent 생성 요약
    category: Optional[str] = None         # Subagent 분류 (CATEGORIES 중 하나)
    final_score: Optional[float] = None
    source: Optional[str] = None
    date: Optional[str] = None


SUMMARY_RESULT_TEMPLATE = {
    "metadata": {
        "total_input": None,
        "categories": CATEGORIES,
        "category_distribution": {          # Subagent 자동 집계
            "AI/기술 동향": None,
            "대학/교육기관 동향": None,
            "정책/제도": None,
            "시장/경쟁사": None,
        },
        "summary_length_decision": None,    # Subagent 결정 (예: "3문장")
        "summary_length_reasoning": None,   # Subagent 근거
    },
    "news": [
        {
            "id": None,
            "title": None,
            "summary": None,                # Subagent 생성
            "category": None,               # Subagent 분류
            "final_score": None,
            "source": None,
            "date": None,
        }
    ],
}


# ============================================================
# 유틸리티: JSON 스키마 출력
# ============================================================

def print_schemas():
    """모든 스키마를 보기 좋게 출력"""
    print("=" * 60)
    print("Phase 2 입력: 빅카인즈 CSV 샘플")
    print("=" * 60)
    print(BIGKINDS_CSV_SAMPLE)
    print("필드 설명:")
    for field, desc in BIGKINDS_FIELDS.items():
        print(f"  {field}: {desc}")

    print("\n" + "=" * 60)
    print("Phase 3 출력: 클러스터링 결과 JSON 스키마")
    print("=" * 60)
    print(json.dumps(CLUSTERING_RESULT_TEMPLATE, ensure_ascii=False, indent=2))

    print("\n" + "=" * 60)
    print("Phase 4 출력: 스코어링 결과 JSON 스키마")
    print("=" * 60)
    print(json.dumps(SCORING_RESULT_TEMPLATE, ensure_ascii=False, indent=2))

    print("\n" + "=" * 60)
    print("Phase 5 출력: 요약/분류 결과 JSON 스키마")
    print("=" * 60)
    print(json.dumps(SUMMARY_RESULT_TEMPLATE, ensure_ascii=False, indent=2))


def save_templates(output_dir="report"):
    """각 스키마를 JSON 파일로 저장"""
    import os
    os.makedirs(output_dir, exist_ok=True)

    schemas = {
        "schema_clustering_result.json": CLUSTERING_RESULT_TEMPLATE,
        "schema_scoring_result.json": SCORING_RESULT_TEMPLATE,
        "schema_summary_result.json": SUMMARY_RESULT_TEMPLATE,
    }

    for filename, schema in schemas.items():
        path = os.path.join(output_dir, filename)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(schema, f, ensure_ascii=False, indent=2)
        print(f"  스키마 저장: {path}")


if __name__ == "__main__":
    print_schemas()
    save_templates()
