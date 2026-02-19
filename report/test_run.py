"""
실제 키워드로 Phase 1 방법론 테스트
"""
import sys
sys.path.insert(0, '/home/user/iannewsclipping/report')

import numpy as np
import time
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from sklearn.metrics.pairwise import cosine_similarity

# ============================================================
# 실제 키워드 리스트
# ============================================================
RAW_QUERY = """LMS OR 학습관리시스템 OR 지원 사업 OR 챗봇 OR AI 챗봇 OR AI 상담 OR 학습 AI OR LXP OR 에듀테크 OR AI 어드바이저 OR AI 학사 OR 무전공 OR 자율전공 OR LINC3.0사업단 OR 대학 지정 OR 대학 선정 OR 대학교 선출 OR 대학교 선임 OR 대학교 인사 OR 교육부 인사 OR 대학혁신지원사업 OR 글로컬 OR 글로컬 예산 OR 교원양성기관 OR 교육 역량 강화 OR KACTL OR CTL OR 교육부 OR 대학 예산 OR 지원금 OR 사업비 OR 대학 확보 OR RISE 예산 OR RISE OR 유비온 OR 자이닉스 OR 메디오피아 OR 프리윌린 OR 엘리스그룹 OR 메이크봇 OR 로이드케이 OR 아이맥스소프트 OR 와이즈넛 OR 마인드로직 OR 국립목포해양대학교 OR 한양대학교 캠퍼스타운 OR 한림대학교 AI에듀테크 센터 OR 연세대학교 리더십센터 OR 신경주대학교 OR 카이스트 교수학습센터 OR 대구한의대학교 CTL OR 제주국제대학교 OR 대구대학교 OR 계명대학교 교수학습개발센터 OR 울산대학교 OR 배화여자대학교 OR 인하대학교 교수학습개발센터 OR 국민대학교 교수학습혁신센터 OR 건양대학교 CTL OR 상지대학교 OR 광주대학교 OR 경남도립거창대학교 OR 순천향대학교 교육혁신원 OR 남부대학교 OR 서울신학대 교육혁신원 OR 동강대학교 OR 대구한의대학교 K-MEDI디지털센터 OR 건양대 에듀테크소프트랩 OR 서울신학대 에듀테크소프트랩 OR 서울여자대학교 디지털혁신실 OR 가톨릭상지대학교 OR 서울신학대학교 국제학부 OR 한림대학교 AI에듀테크 센터 OR 서울대학교 글로벌공학교육센터 OR 한국대학교육협의회 OR 대림대학교 OR 신성대학교 OR 배재대학교 CTL OR 숭의여자대학교 OR 세한대학교 OR 동신대학교 OR 가톨릭관동대학교 교육혁신센터 OR 한림대 커뮤니티교육원 OR 연암공대"""

keywords = [kw.strip() for kw in RAW_QUERY.split(" OR ")]
# 중복 제거 (한림대 AI에듀테크 센터 두 번 등장)
seen = set()
unique_keywords = []
for kw in keywords:
    if kw not in seen:
        seen.add(kw)
        unique_keywords.append(kw)

keywords = unique_keywords
print(f"총 키워드 수: {len(keywords)}개\n")

# ============================================================
# 방법 A: K-means + Elbow Method
# ============================================================
print("=" * 60)
print("방법 A: K-means + Elbow Method (최적 k 자동 결정)")
print("=" * 60)

start = time.time()

vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), sublinear_tf=True)
X = vectorizer.fit_transform(keywords)

n = len(keywords)
max_k = max(2, int(np.sqrt(n)))
k_range = range(2, min(max_k + 1, n))

inertias, sil_scores = [], []
for k in k_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X)
    inertias.append(km.inertia_)
    sil_scores.append(silhouette_score(X, km.labels_))

# Elbow: 2차 미분 최대 지점
if len(inertias) >= 3:
    acc = np.diff(np.diff(inertias))
    elbow_idx = int(np.argmax(np.abs(acc))) + 2
    optimal_k = list(k_range)[elbow_idx]
else:
    optimal_k = list(k_range)[int(np.argmax(sil_scores))]

km_final = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
labels = km_final.fit_predict(X)

elapsed_a = (time.time() - start) / 60
print(f"\n[자동 결정] 최적 k = {optimal_k}")
print(f"[실행 시간] {elapsed_a:.4f}분 ({elapsed_a*60:.2f}초)\n")

clusters_a = {}
for kw, label in zip(keywords, labels):
    clusters_a.setdefault(int(label), []).append(kw)

for cid in sorted(clusters_a.keys()):
    members = clusters_a[cid]
    print(f"  클러스터 {cid:2d} ({len(members)}개): {', '.join(members)}")

# ============================================================
# 방법 1 (Phase 3): TF-IDF + Dendrogram 임계값 자동 결정
# ============================================================
print("\n" + "=" * 60)
print("접근 1: TF-IDF + Dendrogram (임계값+클러스터 수 자동 결정)")
print("=" * 60)

start = time.time()

sim_matrix = cosine_similarity(X)
dist_matrix = 1 - np.clip(sim_matrix, 0, 1)
np.fill_diagonal(dist_matrix, 0)
condensed = squareform(dist_matrix)
Z = linkage(condensed, method="ward")

last_merges = Z[-min(20, n-1):, 2]
if len(last_merges) >= 3:
    acc = np.diff(last_merges, 2)
    cut_idx = int(np.argmax(np.abs(acc))) + 2
    auto_k = max(1, len(last_merges) - cut_idx + 1)
    threshold = float(last_merges[-(cut_idx+1)] if cut_idx < len(last_merges) else last_merges[-1])
else:
    auto_k = max(1, n // 3)
    threshold = float(np.mean(last_merges))

auto_k = max(1, min(auto_k, n))
labels2 = fcluster(Z, auto_k, criterion="maxclust")

elapsed_1 = (time.time() - start) / 60
print(f"\n[자동 결정] 클러스터 수 = {auto_k}")
print(f"[자동 결정] 거리 임계값 = {threshold:.4f}  (유사도 = {1-threshold:.4f})")
print(f"[실행 시간] {elapsed_1:.4f}분 ({elapsed_1*60:.2f}초)\n")

clusters_1 = {}
for kw, label in zip(keywords, labels2):
    clusters_1.setdefault(int(label), []).append(kw)

for cid in sorted(clusters_1.keys()):
    members = clusters_1[cid]
    rep_idxs = [i for i, k in enumerate(keywords) if k in members]
    sub = sim_matrix[np.ix_(rep_idxs, rep_idxs)]
    rep_local = int(np.argmax(sub.mean(axis=1)))
    rep = members[rep_local]
    print(f"  클러스터 {cid:2d} ({len(members)}개) | 대표: [{rep}]")
    others = [m for m in members if m != rep]
    if others:
        print(f"           묶임: {', '.join(others)}")

# ============================================================
# 최종 요약
# ============================================================
print("\n" + "=" * 60)
print("요약")
print("=" * 60)
print(f"  입력 키워드: {len(keywords)}개")
print(f"  방법 A (K-means):    최적 k={optimal_k}개 클러스터  ({elapsed_a*60:.1f}초)")
print(f"  접근 1 (Dendrogram): 자동 {auto_k}개 클러스터       ({elapsed_1*60:.1f}초)")
