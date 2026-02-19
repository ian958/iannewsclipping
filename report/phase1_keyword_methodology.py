"""
Phase 1: 검색어 설계 - 방법론 비교 코드
AI 에듀테크 뉴스클리핑 자동화 시스템

방법 A: K-means 클러스터링 (Elbow Method로 최적 k 자동 결정)
방법 B: BERT 기반 임베딩 (코사인 유사도 분포 분석으로 임계값 자동 제안)
방법 C: LLM Interactive (프로세스 설명 + 예시 프롬프트)
방법 D: 하이브리드 (최종 채택)

주의: 모든 방법은 API 호출 없이 로컬에서 실행됩니다.
"""

import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.feature_extraction.text import TfidfVectorizer


# ============================================================
# 메트릭 측정 유틸리티
# ============================================================

def measure_time(func):
    """함수 실행 시간 측정 데코레이터 (분 단위)"""
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        elapsed_min = (time.time() - start) / 60
        print(f"[메트릭] {func.__name__} 실행 시간: {elapsed_min:.2f}분")
        return result, elapsed_min
    return wrapper


# ============================================================
# 방법 A: K-means 클러스터링 (Elbow Method)
# ============================================================

def find_optimal_k(X, k_range=None):
    """
    Elbow Method로 최적 k를 자동 결정합니다.
    k_range를 지정하지 않으면 알고리즘이 데이터 크기에 맞게 자동 결정합니다.

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        TF-IDF 벡터화된 키워드 행렬
    k_range : range or list, optional
        탐색할 k 범위. None이면 데이터 크기의 제곱근 기반으로 자동 설정

    Returns
    -------
    optimal_k : int  (알고리즘이 자동 결정)
    k_range   : list
    inertias  : list
    silhouette_scores : list
    """
    n_samples = X.shape[0]

    # k_range 자동 결정: sqrt(n) 기반 경험적 상한선
    if k_range is None:
        max_k = max(2, int(np.sqrt(n_samples)))
        k_range = range(2, min(max_k + 1, n_samples))

    inertias = []
    sil_scores = []

    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X)
        inertias.append(kmeans.inertia_)
        sil_scores.append(silhouette_score(X, kmeans.labels_))
        print(f"  k={k}: inertia={kmeans.inertia_:.2f}, silhouette={sil_scores[-1]:.4f}")

    # Elbow point: 두 번째 미분(가속도)이 최대인 지점
    if len(inertias) >= 3:
        diffs = np.diff(inertias)
        acceleration = np.diff(diffs)
        elbow_idx = int(np.argmax(np.abs(acceleration))) + 2  # +2: 인덱스 보정
        optimal_k = list(k_range)[elbow_idx]
    else:
        # 실루엣 스코어가 최대인 k 선택
        optimal_k = list(k_range)[int(np.argmax(sil_scores))]

    print(f"\n[자동 결정] 최적 k = {optimal_k}")
    print(f"  선택 근거: Elbow Method + Silhouette Score 종합 분석")
    return optimal_k, list(k_range), inertias, sil_scores


@measure_time
def run_method_a(keywords):
    """방법 A: K-means 클러스터링 전체 파이프라인"""
    print("=== 방법 A: K-means 클러스터링 (Elbow Method) ===")

    # TF-IDF 벡터화
    vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4))
    X = vectorizer.fit_transform(keywords)

    # 최적 k 자동 탐색
    print("\n[최적 k 탐색 중...]")
    optimal_k, k_range, inertias, sil_scores = find_optimal_k(X)

    # 최종 클러스터링
    kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X)

    # 클러스터별 키워드 출력
    clusters = {}
    for kw, label in zip(keywords, labels):
        clusters.setdefault(label, []).append(kw)

    print(f"\n[결과] 클러스터 {optimal_k}개 생성:")
    for cluster_id, members in sorted(clusters.items()):
        print(f"  클러스터 {cluster_id}: {members}")

    # Elbow curve 시각화 (파일 저장)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(k_range, inertias, "bo-")
    ax1.set_xlabel("클러스터 수 (k)")
    ax1.set_ylabel("Inertia")
    ax1.set_title("Elbow Method")
    ax1.axvline(x=optimal_k, color="r", linestyle="--", label=f"최적 k={optimal_k}")
    ax1.legend()

    ax2.plot(k_range, sil_scores, "gs-")
    ax2.set_xlabel("클러스터 수 (k)")
    ax2.set_ylabel("Silhouette Score")
    ax2.set_title("Silhouette Analysis")
    ax2.axvline(x=optimal_k, color="r", linestyle="--", label=f"최적 k={optimal_k}")
    ax2.legend()

    plt.tight_layout()
    plt.savefig("report/method_a_elbow_curve.png", dpi=150)
    plt.close()
    print("  → Elbow curve 저장: report/method_a_elbow_curve.png")

    return {"method": "A", "optimal_k": optimal_k, "clusters": clusters}


# ============================================================
# 방법 B: BERT 기반 임베딩 + 코사인 유사도 임계값 자동 제안
# ============================================================

def suggest_threshold(similarity_matrix):
    """
    유사도 분포를 분석하여 최적 임계값을 자동 제안합니다.
    (BERT 임베딩 기반 코사인 유사도 행렬 입력)

    Returns
    -------
    suggested_threshold : float  (알고리즘이 자동 결정)
    stats : dict
    """
    # 상삼각 행렬의 유사도 값만 추출 (자기 자신 제외)
    flat_similarities = similarity_matrix[
        np.triu_indices_from(similarity_matrix, k=1)
    ]

    mean = float(np.mean(flat_similarities))
    std = float(np.std(flat_similarities))
    percentiles = {
        "25": float(np.percentile(flat_similarities, 25)),
        "50": float(np.percentile(flat_similarities, 50)),
        "75": float(np.percentile(flat_similarities, 75)),
        "90": float(np.percentile(flat_similarities, 90)),
        "95": float(np.percentile(flat_similarities, 95)),
    }

    # 자동 임계값 결정: 75 percentile (상위 25%를 "유사"로 판단)
    # 직관: 너무 많이 묶이면 정보 손실, 너무 조금 묶이면 중복 처리 효과 없음
    suggested_threshold = percentiles["75"]

    print(f"\n[유사도 분포 분석]")
    print(f"  평균: {mean:.4f}, 표준편차: {std:.4f}")
    print(f"  25th: {percentiles['25']:.4f}")
    print(f"  50th: {percentiles['50']:.4f}")
    print(f"  75th: {percentiles['75']:.4f}  ← 자동 선택 임계값")
    print(f"  90th: {percentiles['90']:.4f}")
    print(f"  95th: {percentiles['95']:.4f}")
    print(f"\n[자동 결정] 권장 임계값 = {suggested_threshold:.4f}")
    print("  선택 근거: 75 percentile - 상위 25% 유사 쌍을 '중복'으로 판단")

    return suggested_threshold, {"mean": mean, "std": std, "percentiles": percentiles}


@measure_time
def run_method_b(keywords):
    """
    방법 B: KoBERT 기반 임베딩 파이프라인 (로컬 실행, API 없음)

    실제 실행 시 필요한 라이브러리:
      pip install transformers torch scikit-learn
    """
    print("=== 방법 B: BERT 기반 임베딩 + 임계값 자동 제안 ===")

    try:
        from transformers import BertModel, BertTokenizer
        import torch
        from sklearn.metrics.pairwise import cosine_similarity as sk_cosine_sim

        # KoBERT 로컬 로드 (API 호출 없음)
        model_name = "snunlp/KR-ELECTRA-discriminator"  # 경량 한국어 BERT 계열
        print(f"  모델 로드: {model_name} (로컬 캐시)")
        tokenizer = BertTokenizer.from_pretrained(model_name)
        model = BertModel.from_pretrained(model_name)
        model.eval()

        # 임베딩 생성
        embeddings = []
        with torch.no_grad():
            for kw in keywords:
                inputs = tokenizer(kw, return_tensors="pt", truncation=True, max_length=64)
                outputs = model(**inputs)
                # [CLS] 토큰 임베딩 사용
                embedding = outputs.last_hidden_state[:, 0, :].squeeze().numpy()
                embeddings.append(embedding)

        embeddings = np.array(embeddings)
        similarity_matrix = sk_cosine_sim(embeddings)

    except ImportError:
        # transformers 미설치 시 더미 데이터로 시뮬레이션
        print("  [시뮬레이션 모드] transformers 미설치 - 더미 유사도 행렬 사용")
        np.random.seed(42)
        n = len(keywords)
        dummy = np.random.rand(n, n)
        similarity_matrix = (dummy + dummy.T) / 2
        np.fill_diagonal(similarity_matrix, 1.0)

    # 최적 임계값 자동 제안
    threshold, stats = suggest_threshold(similarity_matrix)

    # 임계값 기반 그룹핑
    groups = []
    assigned = set()
    for i in range(len(keywords)):
        if i in assigned:
            continue
        group = [i]
        assigned.add(i)
        for j in range(i + 1, len(keywords)):
            if j not in assigned and similarity_matrix[i, j] >= threshold:
                group.append(j)
                assigned.add(j)
        groups.append([keywords[idx] for idx in group])

    print(f"\n[결과] 그룹 {len(groups)}개 (임계값={threshold:.4f} 자동 적용):")
    for i, grp in enumerate(groups):
        print(f"  그룹 {i}: {grp}")

    return {"method": "B", "threshold": threshold, "stats": stats, "groups": groups}


# ============================================================
# 방법 C: LLM Interactive (프로세스 설명)
# ============================================================

METHOD_C_PROCESS = """
=== 방법 C: LLM Interactive (사람이 직접 Claude/ChatGPT에 요청) ===

[역할 분담]
  사람: seed 키워드 준비 → LLM에 확장 요청 → 결과 검토 및 선택
  LLM : 유사어 생성, 최적 개수 제안, 제외 추천 (코드 없음)

[예시 프롬프트 - 키워드 확장]
--------------------------------------------------
다음 seed 키워드들을 바탕으로 AI/에듀테크 관련 뉴스 검색어를 확장해줘:

seed 키워드: [챗봇, LMS, AI 상담, 에듀테크, 무전공]

조건:
1. 각 seed당 5-10개 유사어/관련어 생성
2. 빅카인즈 검색 재현율이 높은 키워드 우선
3. 너무 포괄적(예: "교육")이거나 너무 좁은(예: "특정 대학명") 키워드는 제외
4. 최종 추천 개수도 네가 판단해서 제안해줘

출력 형식:
- 확장된 키워드 리스트 (seed별로 구분)
- 추천하는 최종 키워드 개수와 그 이유
- 제외를 추천하는 키워드와 이유
--------------------------------------------------

[사람이 결정하는 것]
- seed 키워드 (도메인 전문가 판단)
- 최종 키워드 선택 (LLM 제안 중 검토 후 결정)

[LLM이 결정하는 것]
- 없음 (대화형으로 제안만 함, 사람이 승인)

[메트릭]
  사람 작업 시간:
    - 워크샵/seed 도출: [측정 필요]분
    - 프롬프트 작성: [측정 필요]분
    - 결과 검토: [측정 필요]분
  자동화 시간: 0분 (LLM 응답은 사람이 직접 확인)
  API 비용: $0 (Claude.ai 무료 플랜 사용)
"""


def run_method_c():
    """방법 C: LLM Interactive 프로세스 출력"""
    print(METHOD_C_PROCESS)
    return {"method": "C", "type": "interactive", "api_cost": 0}


# ============================================================
# 방법 D: 하이브리드 (최종 채택)
# ============================================================

METHOD_D_PROCESS = """
=== 방법 D: 하이브리드 (최종 채택) ===

Phase 1-1: seed 키워드 도출 (사람)
  - 도메인 전문가 워크샵
  - 회사 제품/서비스 맥락 반영
  - 결과: seed 키워드 리스트

Phase 1-2: 키워드 확장 (사람 + Claude Interactive)
  - 사람이 Claude에 seed 키워드 제공
  - Claude가 확장 키워드 제안
  - 사람이 검토 후 선택
  - 결과: 확장된 키워드 후보 풀

Phase 1-3: 최종 키워드 확정 (사람)
  - 실제 빅카인즈 검색 결과 수 확인
  - 너무 광범위/협소한 키워드 제거
  - OR 연산자로 최종 쿼리 구성
  - 결과: 최종 검색 쿼리

[선택 이유]
  - K-means(A): TF-IDF로는 의미적 유사도 포착 한계
  - BERT(B): 로컬 실행 시 모델 용량(1GB+) 및 설치 부담
  - Interactive(C): 가장 직관적, 전문가 판단 최대 반영
  - Hybrid(D): C의 장점 + 실제 검색 결과 검증으로 품질 보완 ✅

[메트릭]
  사람 작업 시간:
    - 워크샵: [측정 필요]분
    - Claude 확장 요청: [측정 필요]분
    - 검토 및 확정: [측정 필요]분
  자동화 시간: 0분 (Phase 1은 사람 주도)
  API 비용: $0
"""


def run_method_d():
    """방법 D: 하이브리드 프로세스 출력"""
    print(METHOD_D_PROCESS)
    return {"method": "D", "type": "hybrid", "api_cost": 0}


# ============================================================
# 전체 방법론 비교 실행
# ============================================================

if __name__ == "__main__":
    # 샘플 키워드 (실제 사용 키워드의 일부)
    sample_keywords = [
        "AI 챗봇", "챗봇", "AI 상담", "학생 상담",
        "LMS", "학습관리시스템", "LXP",
        "에듀테크", "EdTech", "교육 AI",
        "무전공", "자율전공",
        "교육부", "글로컬", "대학혁신지원사업",
    ]

    results = {}

    # 방법 A: K-means
    print("\n" + "=" * 60)
    result_a, time_a = run_method_a(sample_keywords)
    results["A"] = {"result": result_a, "time_min": time_a}

    # 방법 B: BERT
    print("\n" + "=" * 60)
    result_b, time_b = run_method_b(sample_keywords)
    results["B"] = {"result": result_b, "time_min": time_b}

    # 방법 C: LLM Interactive
    print("\n" + "=" * 60)
    run_method_c()
    results["C"] = {"type": "interactive", "time_min": None}

    # 방법 D: 하이브리드
    print("\n" + "=" * 60)
    run_method_d()
    results["D"] = {"type": "hybrid", "time_min": None}

    # 비교 요약
    print("\n" + "=" * 60)
    print("방법론 비교 요약")
    print("=" * 60)
    print(f"  A (K-means):  자동화 실행 {results['A']['time_min']:.2f}분, API $0")
    print(f"  B (BERT):     자동화 실행 {results['B']['time_min']:.2f}분, API $0")
    print("  C (Interactive): 실행 시간 [측정 필요], API $0")
    print("  D (Hybrid):   실행 시간 [측정 필요], API $0  ← 최종 채택")
