"""
검색어 설계 방법론 비교 - 4가지 방법 통합 테스트
================================================================
방법 A: K-means + Elbow Method     (알고리즘 자동)
방법 B: BERT 임베딩 + 코사인 유사도  (로컬 모델)
방법 C: 서브에이전트 시스템          (멀티에이전트 코디네이터, 로컬)
방법 D: LLM 단독 시뮬레이션         (단일 LLM 방식)

전제: API 호출 없음, 모든 처리 로컬
================================================================
"""

import time
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform

# ============================================================
# 공통 키워드 데이터
# ============================================================

RAW_QUERY = (
    "LMS OR 학습관리시스템 OR 지원 사업 OR 챗봇 OR AI 챗봇 OR AI 상담 OR 학습 AI OR LXP OR "
    "에듀테크 OR AI 어드바이저 OR AI 학사 OR 무전공 OR 자율전공 OR LINC3.0사업단 OR 대학 지정 OR "
    "대학 선정 OR 대학교 선출 OR 대학교 선임 OR 대학교 인사 OR 교육부 인사 OR 대학혁신지원사업 OR "
    "글로컬 OR 글로컬 예산 OR 교원양성기관 OR 교육 역량 강화 OR KACTL OR CTL OR 교육부 OR "
    "대학 예산 OR 지원금 OR 사업비 OR 대학 확보 OR RISE 예산 OR RISE OR 유비온 OR 자이닉스 OR "
    "메디오피아 OR 프리윌린 OR 엘리스그룹 OR 메이크봇 OR 로이드케이 OR 아이맥스소프트 OR 와이즈넛 OR "
    "마인드로직 OR 국립목포해양대학교 OR 한양대학교 캠퍼스타운 OR 한림대학교 AI에듀테크 센터 OR "
    "연세대학교 리더십센터 OR 신경주대학교 OR 카이스트 교수학습센터 OR 대구한의대학교 CTL OR "
    "제주국제대학교 OR 대구대학교 OR 계명대학교 교수학습개발센터 OR 울산대학교 OR 배화여자대학교 OR "
    "인하대학교 교수학습개발센터 OR 국민대학교 교수학습혁신센터 OR 건양대학교 CTL OR 상지대학교 OR "
    "광주대학교 OR 경남도립거창대학교 OR 순천향대학교 교육혁신원 OR 남부대학교 OR 서울신학대 교육혁신원 OR "
    "동강대학교 OR 대구한의대학교 K-MEDI디지털센터 OR 건양대 에듀테크소프트랩 OR "
    "서울신학대 에듀테크소프트랩 OR 서울여자대학교 디지털혁신실 OR 가톨릭상지대학교 OR "
    "서울신학대학교 국제학부 OR 한림대학교 AI에듀테크 센터 OR 서울대학교 글로벌공학교육센터 OR "
    "한국대학교육협의회 OR 대림대학교 OR 신성대학교 OR 배재대학교 CTL OR 숭의여자대학교 OR "
    "세한대학교 OR 동신대학교 OR 가톨릭관동대학교 교육혁신센터 OR 한림대 커뮤니티교육원 OR 연암공대"
)

# 중복 제거
seen, KEYWORDS = set(), []
for kw in [k.strip() for k in RAW_QUERY.split(" OR ")]:
    if kw not in seen:
        seen.add(kw)
        KEYWORDS.append(kw)

# ============================================================
# 공통 유틸
# ============================================================

def tfidf_matrix(keywords):
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), sublinear_tf=True)
    return vec, vec.fit_transform(keywords)


def build_cluster_dict(keywords, labels):
    d = {}
    for kw, lb in zip(keywords, labels):
        d.setdefault(int(lb), []).append(kw)
    return d


def pick_representative(keywords, members, sim_matrix):
    """클러스터 내 평균 유사도가 가장 높은 키워드를 대표로 선정"""
    idxs = [i for i, k in enumerate(keywords) if k in members]
    if len(idxs) == 1:
        return members[0]
    sub = sim_matrix[np.ix_(idxs, idxs)]
    return members[int(np.argmax(sub.mean(axis=1)))]


def print_clusters(cluster_dict, keywords=None, sim_matrix=None, show_rep=False):
    for cid in sorted(cluster_dict.keys()):
        members = cluster_dict[cid]
        if show_rep and keywords and sim_matrix is not None:
            rep = pick_representative(keywords, members, sim_matrix)
            others = [m for m in members if m != rep]
            print(f"  [{cid:2d}] ({len(members)}개) 대표: 『{rep}』", end="")
            if others:
                print(f"  ← {', '.join(others)}", end="")
            print()
        else:
            print(f"  [{cid:2d}] ({len(members)}개) {', '.join(members)}")


# ============================================================
# 방법 A: K-means + Elbow Method
# ============================================================

def method_a(keywords):
    start = time.time()

    _, X = tfidf_matrix(keywords)
    n = len(keywords)
    k_range = range(2, max(3, int(np.sqrt(n)) + 1))

    inertias, sil_scores = [], []
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(X)
        inertias.append(km.inertia_)
        sil_scores.append(silhouette_score(X, km.labels_))

    # Elbow: 2차 미분 최대
    if len(inertias) >= 3:
        acc = np.diff(np.diff(inertias))
        optimal_k = list(k_range)[int(np.argmax(np.abs(acc))) + 2]
    else:
        optimal_k = list(k_range)[int(np.argmax(sil_scores))]

    km_final = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
    labels = km_final.fit_predict(X)
    elapsed = time.time() - start

    clusters = build_cluster_dict(keywords, labels)
    return {
        "method": "A. K-means",
        "decision": "Elbow Method 자동",
        "n_clusters": optimal_k,
        "elapsed": elapsed,
        "clusters": clusters,
        "param": f"k={optimal_k} (Elbow 자동)",
        "pros": "완전 자동, 재현 가능",
        "cons": "의미론적 유사도 한계",
    }


# ============================================================
# 방법 B: BERT 임베딩 + 코사인 유사도
# ============================================================

def method_b(keywords):
    start = time.time()

    try:
        from transformers import AutoTokenizer, AutoModel
        import torch

        model_name = "snunlp/KR-ELECTRA-discriminator"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
        model.eval()

        embeddings = []
        with torch.no_grad():
            for kw in keywords:
                inp = tokenizer(kw, return_tensors="pt", truncation=True, max_length=64)
                out = model(**inp)
                embeddings.append(out.last_hidden_state[:, 0, :].squeeze().numpy())
        embeddings = np.array(embeddings)
        sim_matrix = cosine_similarity(embeddings)
        mode = "KR-ELECTRA 로컬"

    except (ImportError, Exception):
        # transformers 미설치 → TF-IDF 기반 시뮬레이션
        _, X = tfidf_matrix(keywords)
        sim_matrix = cosine_similarity(X)
        mode = "시뮬레이션(TF-IDF 대체)"

    # 유사도 분포 → 75 percentile 임계값 자동 제안
    flat = sim_matrix[np.triu_indices_from(sim_matrix, k=1)]
    threshold = float(np.percentile(flat, 75))

    # 임계값 기반 그리디 그룹핑
    assigned = set()
    groups = []
    for i in range(len(keywords)):
        if i in assigned:
            continue
        grp = [i]
        assigned.add(i)
        for j in range(i + 1, len(keywords)):
            if j not in assigned and sim_matrix[i, j] >= threshold:
                grp.append(j)
                assigned.add(j)
        groups.append(grp)

    clusters = {i: [keywords[idx] for idx in grp] for i, grp in enumerate(groups)}
    elapsed = time.time() - start

    return {
        "method": "B. BERT 임베딩",
        "decision": f"분포 분석 자동 ({mode})",
        "n_clusters": len(clusters),
        "elapsed": elapsed,
        "clusters": clusters,
        "param": f"임계값={threshold:.3f} (75 percentile 자동)",
        "pros": "의미론적 유사도 포착",
        "cons": f"모델 설치 필요 ({mode})",
    }


# ============================================================
# 방법 C: 서브에이전트 시스템 (멀티에이전트, 로컬)
# ============================================================

class CategorizerAgent:
    """Agent 1: TF-IDF 기반 초기 카테고리 분류"""

    def run(self, keywords):
        _, X = tfidf_matrix(keywords)
        sim = cosine_similarity(X)
        # Dendrogram으로 자동 클러스터 수 결정
        dist = 1 - np.clip(sim, 0, 1)
        np.fill_diagonal(dist, 0)
        Z = linkage(squareform(dist), method="ward")
        last = Z[-min(15, len(keywords)-1):, 2]
        if len(last) >= 3:
            acc = np.diff(last, 2)
            cut = int(np.argmax(np.abs(acc))) + 2
            k = max(1, len(last) - cut + 1)
        else:
            k = max(2, len(keywords) // 5)
        labels = fcluster(Z, k, criterion="maxclust")
        return build_cluster_dict(keywords, labels), sim, k


class FilterAgent:
    """Agent 2: 클러스터 내 중복/노이즈 제거"""

    def run(self, clusters, keywords, sim_matrix):
        cleaned = {}
        removed = []
        for cid, members in clusters.items():
            if len(members) <= 1:
                cleaned[cid] = members
                continue
            # 클러스터 내 유사도가 0.95 초과이면 하나만 남김 (실질적 중복)
            keep = [members[0]]
            for m in members[1:]:
                i = keywords.index(members[0])
                j = keywords.index(m)
                if sim_matrix[i, j] < 0.95:
                    keep.append(m)
                else:
                    removed.append(m)
            cleaned[cid] = keep
        return cleaned, removed


class SynthesizerAgent:
    """Agent 3: 각 클러스터 대표 키워드 선정 + 라벨링"""

    CATEGORY_HINTS = {
        "AI/챗봇": ["챗봇", "AI", "LLM", "어드바이저", "상담"],
        "플랫폼/LMS": ["LMS", "LXP", "학습관리", "에듀테크", "소프트랩"],
        "정책/사업": ["교육부", "RISE", "글로컬", "혁신지원", "LINC", "사업비", "예산", "지원금"],
        "대학/기관": ["대학교", "대학원", "대학", "카이스트", "연세", "한양", "서울대"],
        "CTL/교수학습": ["CTL", "KACTL", "교수학습", "교육혁신", "역량 강화"],
        "경쟁사": ["유비온", "자이닉스", "메디오피아", "프리윌린", "엘리스", "메이크봇",
                  "로이드케이", "아이맥스", "와이즈넛", "마인드로직"],
        "입학제도": ["무전공", "자율전공", "선출", "선임", "인사"],
    }

    def run(self, clusters, keywords, sim_matrix):
        result = []
        for cid, members in clusters.items():
            rep = pick_representative(keywords, members, sim_matrix)
            label = self._infer_label(members)
            result.append({
                "id": cid,
                "label": label,
                "representative": rep,
                "members": members,
                "size": len(members),
            })
        return result

    def _infer_label(self, members):
        text = " ".join(members)
        for label, hints in self.CATEGORY_HINTS.items():
            if any(h in text for h in hints):
                return label
        return "기타"


class SubagentCoordinator:
    """코디네이터: 3개 서브에이전트 오케스트레이션"""

    def __init__(self):
        self.categorizer = CategorizerAgent()
        self.filter_agent = FilterAgent()
        self.synthesizer = SynthesizerAgent()
        self.log = []

    def run(self, keywords):
        self.log.append("[Coordinator] 서브에이전트 파이프라인 시작")

        # Agent 1
        self.log.append("[Agent 1: Categorizer] TF-IDF + Dendrogram으로 초기 분류")
        clusters, sim_matrix, k = self.categorizer.run(keywords)
        self.log.append(f"  → {k}개 카테고리 생성")

        # Agent 2
        self.log.append("[Agent 2: Filter] 중복/노이즈 제거")
        cleaned, removed = self.filter_agent.run(clusters, keywords, sim_matrix)
        self.log.append(f"  → {len(removed)}개 중복 제거")

        # Agent 3
        self.log.append("[Agent 3: Synthesizer] 대표 키워드 선정 + 카테고리 라벨링")
        final = self.synthesizer.run(cleaned, keywords, sim_matrix)
        self.log.append(f"  → {len(final)}개 최종 그룹")

        return final, sim_matrix, cleaned


def method_c(keywords):
    start = time.time()
    coordinator = SubagentCoordinator()
    final, sim_matrix, cleaned = coordinator.run(keywords)
    elapsed = time.time() - start

    clusters = {g["id"]: g["members"] for g in final}
    labels_used = {g["label"] for g in final}

    return {
        "method": "C. 서브에이전트",
        "decision": "멀티에이전트 자율 결정",
        "n_clusters": len(final),
        "elapsed": elapsed,
        "clusters": clusters,
        "final_groups": final,
        "log": coordinator.log,
        "param": f"Categorizer→Filter→Synthesizer 3단계",
        "pros": "역할 분리, 결정 근거 투명",
        "cons": "파이프라인 설계 복잡",
    }


# ============================================================
# 방법 D: LLM 단독 시뮬레이션 (단일 패스, 규칙 기반)
# ============================================================

# 실제 LLM이 맥락 이해로 분류할 법한 카테고리 규칙
LLM_CATEGORIES = {
    "AI·챗봇 기술": {
        "keywords": ["챗봇", "AI 챗봇", "AI 상담", "학습 AI", "AI 어드바이저", "AI 학사", "LLM"],
    },
    "학습 플랫폼·에듀테크": {
        "keywords": ["LMS", "학습관리시스템", "LXP", "에듀테크",
                     "건양대 에듀테크소프트랩", "서울신학대 에듀테크소프트랩",
                     "한림대학교 AI에듀테크 센터"],
    },
    "정부 정책·사업비": {
        "keywords": ["교육부", "교육부 인사", "대학혁신지원사업", "글로컬", "글로컬 예산",
                     "RISE", "RISE 예산", "지원금", "사업비", "대학 예산", "대학 확보",
                     "대학 지정", "대학 선정", "지원 사업", "LINC3.0사업단",
                     "교원양성기관", "교육 역량 강화"],
    },
    "대학 인사·행정": {
        "keywords": ["대학교 선출", "대학교 선임", "대학교 인사", "무전공", "자율전공"],
    },
    "CTL·교수학습센터": {
        "keywords": ["CTL", "KACTL", "카이스트 교수학습센터",
                     "계명대학교 교수학습개발센터", "인하대학교 교수학습개발센터",
                     "국민대학교 교수학습혁신센터", "건양대학교 CTL",
                     "대구한의대학교 CTL", "배재대학교 CTL",
                     "순천향대학교 교육혁신원", "서울신학대 교육혁신원",
                     "가톨릭관동대학교 교육혁신센터", "대구한의대학교 K-MEDI디지털센터",
                     "서울여자대학교 디지털혁신실", "한림대 커뮤니티교육원"],
    },
    "에듀테크 경쟁사": {
        "keywords": ["유비온", "자이닉스", "메디오피아", "프리윌린", "엘리스그룹",
                     "메이크봇", "로이드케이", "아이맥스소프트", "와이즈넛", "마인드로직"],
    },
    "파트너·고객 대학교": {
        "keywords": ["국립목포해양대학교", "한양대학교 캠퍼스타운", "연세대학교 리더십센터",
                     "신경주대학교", "제주국제대학교", "대구대학교", "울산대학교",
                     "배화여자대학교", "상지대학교", "광주대학교", "경남도립거창대학교",
                     "남부대학교", "동강대학교", "가톨릭상지대학교",
                     "서울신학대학교 국제학부", "서울대학교 글로벌공학교육센터",
                     "한국대학교육협의회", "대림대학교", "신성대학교", "숭의여자대학교",
                     "세한대학교", "동신대학교", "연암공대"],
    },
}


def method_d(keywords):
    start = time.time()

    # LLM 단독: 규칙 기반 카테고리 매핑 (단일 패스)
    assigned = {}
    unassigned = list(keywords)

    for cat, info in LLM_CATEGORIES.items():
        matched = [kw for kw in unassigned if kw in info["keywords"]]
        if matched:
            assigned[cat] = matched
        unassigned = [kw for kw in unassigned if kw not in matched]

    # 미분류 키워드 → "기타" 카테고리
    if unassigned:
        assigned["기타"] = unassigned

    # 정수 키로 변환
    clusters = {i: v for i, (k, v) in enumerate(assigned.items())}
    cat_labels = list(assigned.keys())
    elapsed = time.time() - start

    return {
        "method": "D. LLM 단독",
        "decision": "LLM 의미 이해 기반 (단일 패스)",
        "n_clusters": len(assigned),
        "elapsed": elapsed,
        "clusters": clusters,
        "cat_labels": cat_labels,
        "assigned": assigned,
        "param": f"{len(LLM_CATEGORIES)}개 카테고리 규칙 (LLM 판단 시뮬레이션)",
        "pros": "맥락 이해, 라벨 명확",
        "cons": "재현성 낮음, 프롬프트 의존",
    }


# ============================================================
# 결과 비교표 출력
# ============================================================

def print_comparison_table(results):
    print("\n" + "=" * 90)
    print("  방법론 비교표")
    print("=" * 90)

    # 헤더
    fmt = "{:<18} {:<22} {:>8} {:>8} {:<22} {:<22}"
    print(fmt.format("방법", "결정 방식", "그룹 수", "시간(초)", "장점", "단점"))
    print("-" * 90)

    for r in results:
        print(fmt.format(
            r["method"],
            r["decision"][:22],
            str(r["n_clusters"]),
            f"{r['elapsed']:.2f}",
            r["pros"][:22],
            r["cons"][:22],
        ))

    print("-" * 90)
    print(fmt.format("API 비용", "$0 (전원)", "-", "-", "로컬 실행", ""))
    print("=" * 90)


def print_detail(result):
    print(f"\n{'─'*60}")
    print(f"  {result['method']} | 파라미터: {result['param']}")
    print(f"{'─'*60}")

    # 방법 C는 서브에이전트 로그 + 라벨 출력
    if "final_groups" in result:
        for g in result["final_groups"]:
            print(f"  [{g['label']:12s}] ({g['size']}개) 대표: 『{g['representative']}』")
            others = [m for m in g["members"] if m != g["representative"]]
            if others:
                print(f"               ← {', '.join(others)}")
        if result.get("log"):
            print(f"\n  [에이전트 실행 로그]")
            for line in result["log"]:
                print(f"    {line}")

    # 방법 D는 카테고리별 출력
    elif "assigned" in result:
        for cat, members in result["assigned"].items():
            print(f"  [{cat:16s}] ({len(members)}개): {', '.join(members)}")

    # 방법 A, B
    else:
        for cid in sorted(result["clusters"].keys()):
            members = result["clusters"][cid]
            print(f"  [{cid:2d}] ({len(members)}개): {', '.join(members)}")


# ============================================================
# 메인
# ============================================================

if __name__ == "__main__":
    print(f"입력 키워드: {len(KEYWORDS)}개\n")
    print("각 방법론 실행 중...\n")

    results = []

    print("[방법 A] K-means + Elbow Method 실행 중...")
    results.append(method_a(KEYWORDS))
    print(f"  → 완료: {results[-1]['n_clusters']}개 클러스터, {results[-1]['elapsed']:.2f}초")

    print("[방법 B] BERT 임베딩 + 코사인 유사도 실행 중...")
    results.append(method_b(KEYWORDS))
    print(f"  → 완료: {results[-1]['n_clusters']}개 그룹, {results[-1]['elapsed']:.2f}초")

    print("[방법 C] 서브에이전트 시스템 실행 중...")
    results.append(method_c(KEYWORDS))
    print(f"  → 완료: {results[-1]['n_clusters']}개 그룹, {results[-1]['elapsed']:.2f}초")

    print("[방법 D] LLM 단독 시뮬레이션 실행 중...")
    results.append(method_d(KEYWORDS))
    print(f"  → 완료: {results[-1]['n_clusters']}개 카테고리, {results[-1]['elapsed']:.2f}초")

    # 요약 비교표
    print_comparison_table(results)

    # 각 방법 상세 결과
    for r in results:
        print_detail(r)
