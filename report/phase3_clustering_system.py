"""
Phase 3: 클러스터링 시스템 코드
AI 에듀테크 뉴스클리핑 자동화 시스템

접근 1: TF-IDF + Cosine Similarity + Dendrogram (임계값 자동 탐색)
접근 2: HDBSCAN (클러스터 수 자동 결정)
접근 3: Subagent 방식 (최종 채택) - 워크플로우 설명

주의: API 호출 없음, 모든 파라미터는 알고리즘이 자동 결정
"""

import time
import csv
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import squareform


# ============================================================
# 메트릭 측정
# ============================================================

def measure_time(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        elapsed_min = (time.time() - start) / 60
        print(f"[메트릭] {func.__name__}: {elapsed_min:.4f}분")
        return result, elapsed_min
    return wrapper


def load_news_csv(filepath):
    """빅카인즈 CSV 로드"""
    news = []
    with open(filepath, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            title_col = next((k for k in row if "제목" in k or "title" in k.lower()), None)
            id_col = next((k for k in row if "번호" in k or "id" in k.lower()), None)
            if title_col and row[title_col].strip():
                news.append({
                    "id": row.get(id_col, str(len(news) + 1)),
                    "title": row[title_col].strip(),
                    "raw": row,
                })
    return news


# ============================================================
# 접근 1: TF-IDF + Cosine Similarity + Dendrogram
# ============================================================

def compute_tfidf_similarity(titles):
    """
    제목 기반 TF-IDF 유사도 행렬 계산
    char n-gram으로 한국어 형태소 분석 없이 처리
    """
    vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(2, 4),  # 2~4글자 n-gram
        min_df=1,
        sublinear_tf=True,
    )
    tfidf_matrix = vectorizer.fit_transform(titles)
    similarity_matrix = cosine_similarity(tfidf_matrix)
    return similarity_matrix


def auto_find_threshold(similarity_matrix, method="ward"):
    """
    계층적 클러스터링 Dendrogram에서 최적 임계값과 클러스터 수를 자동 탐색합니다.

    알고리즘:
    1. 유사도 → 거리 행렬 변환 (distance = 1 - similarity)
    2. Ward 링키지로 계층적 클러스터링
    3. 마지막 10개 병합 단계의 거리 증가폭 계산
    4. 가속도(2차 미분)가 최대인 지점 = 자연스러운 절단점

    Returns
    -------
    optimal_k : int    자동 결정된 클러스터 수
    threshold : float  자동 결정된 거리 임계값
    Z : ndarray        linkage matrix
    """
    n = similarity_matrix.shape[0]
    distance_matrix = 1 - np.clip(similarity_matrix, 0, 1)
    np.fill_diagonal(distance_matrix, 0)

    condensed_dist = squareform(distance_matrix)
    Z = linkage(condensed_dist, method=method)

    # 마지막 병합 단계의 거리 값 분석
    last_merges = Z[-min(20, n - 1):, 2]
    idxs = np.arange(1, len(last_merges) + 1)

    # 가속도: 2차 미분
    if len(last_merges) >= 3:
        acceleration = np.diff(last_merges, 2)
        # 가장 큰 가속도 지점에서 자르기
        cut_idx = int(np.argmax(np.abs(acceleration))) + 2
        optimal_k = len(last_merges) - cut_idx + 1
        threshold = float(last_merges[-(cut_idx + 1)] if cut_idx < len(last_merges) else last_merges[-1])
    else:
        optimal_k = max(1, n // 3)
        threshold = float(np.mean(last_merges))

    optimal_k = max(1, min(optimal_k, n))

    print(f"[자동 결정] 클러스터 수: {optimal_k}, 거리 임계값: {threshold:.4f}")
    print(f"  (유사도 임계값 = 1 - {threshold:.4f} = {1 - threshold:.4f})")
    return optimal_k, threshold, Z


@measure_time
def run_approach1_tfidf(news_list):
    """
    접근 1: TF-IDF + Dendrogram 클러스터링
    모든 파라미터 자동 결정 (API 없음)
    """
    print("=== 접근 1: TF-IDF + Cosine Similarity + Dendrogram ===\n")

    if not news_list:
        print("  뉴스 데이터 없음 (샘플 데이터로 시뮬레이션)")
        news_list = _get_sample_news()

    titles = [n["title"] for n in news_list]
    print(f"  입력: {len(titles)}건")

    # 유사도 행렬 계산
    similarity_matrix = compute_tfidf_similarity(titles)

    # 최적 임계값 + 클러스터 수 자동 결정
    optimal_k, threshold, Z = auto_find_threshold(similarity_matrix)

    # 클러스터 레이블 부여
    distance_matrix = 1 - np.clip(similarity_matrix, 0, 1)
    np.fill_diagonal(distance_matrix, 0)
    condensed_dist = squareform(distance_matrix)
    Z_full = linkage(condensed_dist, method="ward")
    labels = fcluster(Z_full, optimal_k, criterion="maxclust")

    # 클러스터 구성
    clusters = {}
    for idx, label in enumerate(labels):
        clusters.setdefault(int(label), []).append(idx)

    # 대표 기사 선정 (클러스터 중심과 가장 가까운 기사)
    result_clusters = []
    for cluster_id, members in clusters.items():
        if len(members) == 1:
            rep_idx = members[0]
        else:
            sub_sim = similarity_matrix[np.ix_(members, members)]
            avg_sim = sub_sim.mean(axis=1)
            rep_idx = members[int(np.argmax(avg_sim))]

        result_clusters.append({
            "cluster_id": cluster_id,
            "representative_id": news_list[rep_idx]["id"],
            "representative_title": titles[rep_idx],
            "member_ids": [news_list[i]["id"] for i in members],
            "size": len(members),
            "avg_similarity": float(similarity_matrix[np.ix_(members, members)].mean()),
        })

    # Dendrogram 시각화
    plt.figure(figsize=(14, 6))
    dendrogram(Z_full, labels=[t[:15] + "…" if len(t) > 15 else t for t in titles],
               leaf_rotation=90, leaf_font_size=7)
    plt.axhline(y=threshold, color="r", linestyle="--", label=f"자동 임계값={threshold:.4f}")
    plt.title("계층적 클러스터링 Dendrogram (임계값 자동 결정)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("report/approach1_dendrogram.png", dpi=150)
    plt.close()

    print(f"\n[결과] 클러스터 {len(result_clusters)}개 (입력 {len(news_list)}건)")
    print(f"  압축률: {(1 - len(result_clusters)/len(news_list))*100:.1f}%")
    print("  → Dendrogram 저장: report/approach1_dendrogram.png")

    return {
        "method": "TF-IDF + Dendrogram",
        "optimal_k": optimal_k,
        "similarity_threshold": 1 - threshold,
        "clusters": result_clusters,
        "compression_rate": (1 - len(result_clusters) / len(news_list)) * 100,
    }


# ============================================================
# 접근 2: HDBSCAN (클러스터 수 자동 결정)
# ============================================================

@measure_time
def run_approach2_hdbscan(news_list):
    """
    접근 2: HDBSCAN - 클러스터 수를 알고리즘이 완전 자동 결정
    API 없음, 로컬 실행
    """
    print("=== 접근 2: HDBSCAN 자동 클러스터링 ===\n")

    if not news_list:
        news_list = _get_sample_news()

    titles = [n["title"] for n in news_list]
    print(f"  입력: {len(titles)}건")

    # TF-IDF 임베딩
    vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), sublinear_tf=True)
    X = vectorizer.fit_transform(titles).toarray()

    try:
        import umap
        import hdbscan

        # UMAP 차원 축소 (n_components 자동: 데이터 크기의 로그 기반)
        n_components = max(2, min(10, int(np.log2(len(titles)) + 1)))
        print(f"  UMAP 차원: {n_components} (자동 결정, log2({len(titles)})+1 기반)")

        reducer = umap.UMAP(n_components=n_components, random_state=42, metric="cosine")
        reduced = reducer.fit_transform(X)

        # HDBSCAN - 클러스터 수 완전 자동 결정
        # min_cluster_size: 최소 2개는 묶여야 '중복'으로 의미 있음
        min_cluster_size = max(2, len(titles) // 20)
        print(f"  min_cluster_size: {min_cluster_size} (자동: n//20)")

        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=1,
            cluster_selection_method="eom",  # Excess of Mass
            metric="euclidean",
        )
        labels = clusterer.fit_predict(reduced)

        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        noise_count = int((labels == -1).sum())
        cluster_sizes = np.bincount(labels[labels >= 0]).tolist() if n_clusters > 0 else []

        print(f"\n[자동 결정] 클러스터 수: {n_clusters}")
        print(f"  노이즈(미분류) 포인트: {noise_count}건")
        print(f"  클러스터별 크기: {cluster_sizes}")

    except ImportError:
        print("  [시뮬레이션] umap/hdbscan 미설치 - 근사 결과 출력")
        # 간단한 근사: 유사도 행렬 기반 그리디 클러스터링
        sim_matrix = compute_tfidf_similarity(titles)
        threshold = float(np.percentile(sim_matrix[np.triu_indices_from(sim_matrix, k=1)], 75))

        labels = np.full(len(titles), -1, dtype=int)
        cluster_id = 0
        for i in range(len(titles)):
            if labels[i] != -1:
                continue
            labels[i] = cluster_id
            for j in range(i + 1, len(titles)):
                if labels[j] == -1 and sim_matrix[i, j] >= threshold:
                    labels[j] = cluster_id
            cluster_id += 1

        n_clusters = int(labels.max()) + 1
        noise_count = 0
        cluster_sizes = np.bincount(labels).tolist()

    # 결과 구성
    clusters_dict = {}
    for idx, label in enumerate(labels):
        if label == -1:
            continue
        clusters_dict.setdefault(int(label), []).append(idx)

    result_clusters = []
    for cluster_id, members in clusters_dict.items():
        sub_sim = compute_tfidf_similarity([titles[i] for i in members])
        avg_sim = sub_sim.mean(axis=1)
        rep_local_idx = int(np.argmax(avg_sim))
        rep_idx = members[rep_local_idx]

        result_clusters.append({
            "cluster_id": cluster_id,
            "representative_id": news_list[rep_idx]["id"],
            "representative_title": titles[rep_idx],
            "member_ids": [news_list[i]["id"] for i in members],
            "size": len(members),
        })

    print(f"\n[결과] 클러스터 {len(result_clusters)}개 (노이즈 {noise_count}건 별도)")
    print(f"  압축률: {(1 - len(result_clusters)/len(news_list))*100:.1f}%")

    return {
        "method": "HDBSCAN",
        "n_clusters_auto": n_clusters,
        "noise_points": noise_count,
        "cluster_sizes": cluster_sizes,
        "clusters": result_clusters,
    }


# ============================================================
# 접근 3: Subagent 방식 (최종 채택) - 워크플로우
# ============================================================

APPROACH3_PROCESS = """
=== 접근 3: Subagent 방식 (최종 채택) ===

[사람의 역할]
  1. 빅카인즈에서 다운로드한 CSV 파일을 Claude에 업로드
  2. 아래 프롬프트 입력:

  -----------------------------------------------
  이 뉴스들 중 비슷한 제목을 가진 것들을 묶어줘.

  조건:
  - 유사도 임계값은 네가 데이터를 분석해서 최적값을 정해
  - 클러스터 개수도 자동으로 결정
  - 각 클러스터에서 대표 기사 1개 선정
  - 너무 작은 클러스터(1~2개) 제외 여부를 판단해서 제안

  출력에 포함할 것:
  - 최종 클러스터 개수와 선정 이유
  - 사용한 유사도 임계값과 선정 근거
  - 제외를 추천하는 클러스터와 이유
  - 각 클러스터의 대표 기사
  -----------------------------------------------

  3. 결과 확인 및 Subagent 제안 검토
  4. 필요 시 수동 조정 (클러스터 병합/분리)

[Subagent (Claude)가 결정하는 것]
  - 제목 유사도 계산 방법 (Levenshtein, Jaccard, 의미론적 등)
  - 유사도 분포 분석 후 최적 임계값 자동 결정
  - 클러스터 개수 자동 결정
  - 대표 기사 선정 기준 (정보량, 출처 신뢰도 등)
  - 작은 클러스터 처리 방안 제안

[선택 이유]
  - TF-IDF(1): 의미론적 유사도 포착 한계 (동음이의어, 표현 다양성)
  - HDBSCAN(2): 설치 부담 + 한국어 특화 튜닝 어려움
  - Subagent(3): 맥락 이해 기반 자연어 처리, 결정 근거 설명 가능 ✅

[메트릭 (빈 템플릿 - 측정 후 기입)]
  사람 작업 시간:
    - CSV 업로드: [측정 필요]분
    - 프롬프트 작성: [측정 필요]분
    - 결과 검토: [측정 필요]분
  Subagent 실행 시간: [측정 필요]분
  총 소요 시간: [계산됨]분
  API 비용: $0 (Claude.ai 사용)
  처리 건수: [입력]건 → [출력]건
  압축률: [자동 계산]%

  Subagent가 결정한 파라미터:
    - 유사도 임계값: [Subagent 결정]
    - 클러스터 개수: [Subagent 결정]
    - 제외한 작은 클러스터: [Subagent 결정]개
"""


def run_approach3_subagent():
    print(APPROACH3_PROCESS)


# ============================================================
# 유틸리티: 샘플 데이터
# ============================================================

def _get_sample_news():
    """테스트용 샘플 뉴스 데이터"""
    return [
        {"id": "1", "title": "서울대, ChatGPT 기반 학생 상담 챗봇 도입"},
        {"id": "2", "title": "서울대학교 AI 챗봇 학생상담 시스템 시범운영"},
        {"id": "3", "title": "교육부, 대학 AI 활용 가이드라인 발표"},
        {"id": "4", "title": "교육부 대학 인공지능 활용 지침 공개"},
        {"id": "5", "title": "에듀테크 기업 투자 유치 잇따라"},
        {"id": "6", "title": "글로컬대학 30 최종 선정 결과 발표"},
        {"id": "7", "title": "글로컬대학30 선정 대학 명단 공개"},
        {"id": "8", "title": "무전공 입학제도 확대 추진"},
        {"id": "9", "title": "자율전공학부 신설 대학 증가"},
        {"id": "10", "title": "LLM 기반 교육용 AI 튜터 개발 현황"},
        {"id": "11", "title": "한양대 AI 학사 상담 서비스 출시"},
        {"id": "12", "title": "연세대 챗봇 수강신청 도우미 도입"},
    ]


# ============================================================
# 클러스터링 결과 CSV 저장
# ============================================================

def save_clustering_result(clusters, output_path="report/clustering_result.csv"):
    """클러스터링 결과를 CSV로 저장"""
    with open(output_path, "w", newline="", encoding="utf-8-sig") as f:
        fieldnames = ["cluster_id", "representative_id", "representative_title",
                      "member_ids", "size", "avg_similarity"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for c in clusters:
            row = {k: c.get(k, "") for k in fieldnames}
            if isinstance(row.get("member_ids"), list):
                row["member_ids"] = "|".join(str(x) for x in row["member_ids"])
            if isinstance(row.get("avg_similarity"), float):
                row["avg_similarity"] = f"{row['avg_similarity']:.4f}"
            writer.writerow(row)
    print(f"  클러스터링 결과 저장: {output_path}")


if __name__ == "__main__":
    sample_news = _get_sample_news()

    print("\n" + "=" * 60)
    result1, time1 = run_approach1_tfidf(sample_news)
    save_clustering_result(result1["clusters"], "report/approach1_result.csv")

    print("\n" + "=" * 60)
    result2, time2 = run_approach2_hdbscan(sample_news)

    print("\n" + "=" * 60)
    run_approach3_subagent()

    print("\n" + "=" * 60)
    print("클러스터링 방법 비교 요약")
    print(f"  접근 1 (TF-IDF):   자동 결정 클러스터={result1['optimal_k']}개, 실행={time1:.2f}분")
    print(f"  접근 2 (HDBSCAN):  자동 결정 클러스터={result2['n_clusters_auto']}개, 실행={time2:.2f}분")
    print("  접근 3 (Subagent): [측정 필요] ← 최종 채택")
