"""
Phase 6: 검색식 생성 - Subagent 기반 OR+AND 혼합 쿼리 빌더
================================================================
입력  : 클러스터링 결과 (A/B/C/D 4개 방법 모두)
출력  : 최종 검색식 후보 5개 (각 후보마다 OR·AND 혼합 구성)
기록  : ProcessLogger → query_candidates_archive.txt +
                        process_log_archive.jsonl 에 누적

서브에이전트 구성
  KeywordMergerAgent  : A/B/C/D 4개 결과 통합 + 카테고리별 합의도 계산
  QueryBuilderAgent   : 카테고리별 OR 키워드 묶음 생성 + 길이 필터
  StrategyAgent       : and_groups 리스트로 5가지 전략 생성
                        후보1 = 순수 OR (1그룹 전체)
                        후보2~5 = AND 그룹별로 OR 키워드 묶음
  QueryFormatterAgent : and_groups → 실제 검색식 문자열 포맷팅
                        (A OR B) AND (C OR D) AND ...
  ValidatorAgent      : 복잡도·유효성 검증
  QueryCoordinator    : 5개 에이전트 오케스트레이션
================================================================
"""

import json
import time
from datetime import datetime
from pathlib import Path


# ============================================================
# 프로세스 로거 (전체 파이프라인 기록)
# ============================================================

class ProcessLogger:
    """
    테스트 프로세스 전 과정을 기록하는 로거.
    - record()        : 에이전트 실행 이벤트 기록
    - record_decision(): 의사결정(파라미터 선택) 기록
    - save()          : JSON + TXT 파일 저장
    """

    VERSION = "1.1.0"

    def __init__(self, session_name: str = "keyword_search_design"):
        self.session_name = session_name
        self.started_at = datetime.now().isoformat()
        self.events: list[dict] = []
        self.decisions: list[dict] = []

    def record(self, agent: str, action: str,
               inputs: dict = None, outputs: dict = None,
               elapsed: float = None):
        self.events.append({
            "ts": datetime.now().isoformat(),
            "agent": agent,
            "action": action,
            "inputs": inputs or {},
            "outputs": outputs or {},
            "elapsed_sec": round(elapsed, 4) if elapsed is not None else None,
        })

    def record_decision(self, agent: str, parameter: str,
                        value: str, reasoning: str):
        self.decisions.append({
            "ts": datetime.now().isoformat(),
            "agent": agent,
            "parameter": parameter,
            "value": value,
            "reasoning": reasoning,
        })

    def save(self, out_dir: str = "report") -> tuple[str, str]:
        """단일 아카이브 파일에 누적 저장 (타임스탬프 파일 생성 안 함)."""
        Path(out_dir).mkdir(parents=True, exist_ok=True)

        archive_path = f"{out_dir}/process_log_archive.jsonl"
        payload = {
            "version":        self.VERSION,
            "session":        self.session_name,
            "started_at":     self.started_at,
            "completed_at":   datetime.now().isoformat(),
            "total_events":   len(self.events),
            "total_decisions": len(self.decisions),
            "events":         self.events,
            "decisions":      self.decisions,
        }
        # JSON Lines 형식으로 append (1 run = 1 line)
        with open(archive_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")

        print(f"\n  [ProcessLogger] 로그 아카이브 → {archive_path}")
        return archive_path, archive_path


# ============================================================
# 서브에이전트 0: KeywordMergerAgent
# ============================================================

class KeywordMergerAgent:
    """
    역할  : A/B/C/D 4개 방법 결과를 통합하고 카테고리별 합의도 계산
    합의도: 같은 카테고리 내 키워드 쌍이 A/B/C 방법에서도 동일 클러스터에
            공출현하는 비율 (0.0 ~ 1.0)
    출력  : assigned {카테고리명: [keywords]},
            consensus_scores {카테고리명: float}
    """

    def run(self, r_a: dict, r_b: dict, r_c: dict, r_d: dict,
            logger: ProcessLogger) -> tuple[dict, dict]:
        t0 = time.time()

        # ── 1. 각 방법의 keyword → cluster_id 매핑 ────────────────
        kw_cluster: dict[str, dict] = {"A": {}, "B": {}, "C": {}}

        for cid, members in r_a["clusters"].items():
            for kw in members:
                kw_cluster["A"][kw] = str(cid)

        for cid, members in r_b["clusters"].items():
            for kw in members:
                kw_cluster["B"][kw] = str(cid)

        for g in r_c.get("final_groups", []):
            for kw in g["members"]:
                kw_cluster["C"][kw] = g["label"]

        # ── 2. Method D 카테고리를 기준으로 합의도 계산 ────────────
        assigned: dict[str, list[str]] = r_d["assigned"].copy()
        consensus_scores: dict[str, float] = {}

        for cat, members in assigned.items():
            if len(members) <= 1:
                consensus_scores[cat] = 1.0
                continue

            agreed = 0
            total = 0
            for i, kw1 in enumerate(members):
                for kw2 in members[i + 1:]:
                    total += 1
                    # A, B, C 중 하나라도 같은 클러스터에 공출현하면 합의
                    for method in ("A", "B", "C"):
                        c1 = kw_cluster[method].get(kw1)
                        c2 = kw_cluster[method].get(kw2)
                        if c1 is not None and c2 is not None and c1 == c2:
                            agreed += 1
                            break

            consensus_scores[cat] = agreed / total if total > 0 else 0.0

        elapsed = time.time() - t0

        # ── 3. 로깅 ────────────────────────────────────────────────
        sorted_cats = sorted(consensus_scores.items(), key=lambda x: -x[1])
        logger.record(
            agent="KeywordMergerAgent",
            action="4개 방법(A/B/C/D) 결과 통합 및 합의도 계산",
            inputs={"methods": ["A", "B", "C", "D"],
                    "n_categories": len(assigned),
                    "total_keywords": sum(len(v) for v in assigned.values())},
            outputs={"consensus_scores": {k: round(v, 3) for k, v in sorted_cats},
                     "top_consensus_cat": sorted_cats[0][0] if sorted_cats else None},
            elapsed=elapsed,
        )
        for cat, score in sorted_cats:
            logger.record_decision(
                agent="KeywordMergerAgent",
                parameter=f"합의도: {cat}",
                value=f"{score:.3f}",
                reasoning=(
                    f"A/B/C 방법에서 '{cat}' 카테고리 멤버 쌍이 "
                    f"동일 클러스터에 공출현하는 비율 = {score:.3f}"
                ),
            )

        return assigned, consensus_scores


# ============================================================
# 서브에이전트 1: QueryBuilderAgent
# ============================================================

class QueryBuilderAgent:
    """
    역할  : 카테고리별 키워드를 OR 묶음으로 변환
    정책  : 단어 수 3개 이하 키워드만 포함 (긴 기관명은 검색 잡음 증가)
    출력  : {카테고리명: [키워드, ...]} OR 그룹 딕셔너리
    """

    MAX_WORDS = 3

    def run(self, assigned: dict, logger: ProcessLogger) -> dict:
        t0 = time.time()
        or_groups: dict[str, list[str]] = {}

        for cat, members in assigned.items():
            filtered = [m for m in members if len(m.split()) <= self.MAX_WORDS]
            if filtered:
                or_groups[cat] = filtered

        elapsed = time.time() - t0
        logger.record(
            agent="QueryBuilderAgent",
            action="카테고리별 OR 묶음 생성",
            inputs={"n_categories": len(assigned),
                    "total_keywords": sum(len(v) for v in assigned.values())},
            outputs={"n_or_groups": len(or_groups),
                     "included_keywords": sum(len(v) for v in or_groups.values())},
            elapsed=elapsed,
        )
        logger.record_decision(
            agent="QueryBuilderAgent",
            parameter="키워드 길이 필터",
            value=f"최대 {self.MAX_WORDS}단어",
            reasoning=(
                "빅카인즈 검색식에 긴 기관명을 포함하면 "
                "약칭·다른 표기로 작성된 기사가 누락됨. "
                "핵심 개념어 위주로 OR 묶음 구성해 재현율 유지."
            ),
        )
        return or_groups


# ============================================================
# 서브에이전트 2: TermScoringAgent
# ============================================================

class TermScoringAgent:
    """
    역할: 각 키워드의 특이도(Specificity) 점수 계산
    특이도 낮은 일반어 → OR 쿼리에서 제거, AND 그룹 하위 순위
    특이도 높은 특정어 → AND 그룹 우선 사용

    점수 구성:
      - 단어 수  (복합어일수록 구체적)
      - 문자 길이 (길수록 구체적)
      - ASCII 비율 (약어·영문 = 고유명사/특정)
      - 숫자 포함 (버전·연도 = 매우 특정)
      - 카테고리 수정자 (경쟁사=+0.20 / 인사행정=-0.20)
    """

    # 카테고리별 특이도 수정자 (도메인 지식 기반)
    CAT_MODIFIER: dict[str, float] = {
        "AI·챗봇 기술":        +0.20,
        "학습 플랫폼·에듀테크": +0.10,
        "에듀테크 경쟁사":      +0.20,   # 고유 기업명 → 특정
        "CTL·교수학습센터":     +0.00,
        "정부 정책·사업비":     -0.05,   # 정책어는 다소 일반
        "파트너·고객 대학교":   -0.15,   # 교육 뉴스에서 빈출
        "대학 인사·행정":       -0.20,   # 가장 일반적
    }

    # 명시적 일반어 (어느 카테고리든 OR에 넣으면 잡음 유발)
    GENERIC_TERMS: frozenset = frozenset({
        "무전공", "자율전공",
        "대학교 선출", "대학교 선임", "대학교 인사",
    })

    def _base_score(self, kw: str) -> float:
        words = kw.split()
        # 단어 수 (복합어일수록 특정적)
        word_s  = min(len(words) / 4,  0.35)
        # 문자 길이
        len_s   = min(len(kw) / 15,    0.30)
        # ASCII 비율 (영문/숫자 ≈ 약어·브랜드)
        ascii_n = sum(1 for c in kw if c.isascii() and c.isalnum())
        ascii_s = min(ascii_n / max(len(kw.replace(" ", "")), 1), 1.0) * 0.25
        # 숫자 포함
        num_s   = 0.10 if any(c.isdigit() for c in kw) else 0.0
        return word_s + len_s + ascii_s + num_s

    def score(self, kw: str, cat: str) -> float:
        if kw in self.GENERIC_TERMS:
            return 0.01
        base = self._base_score(kw)
        mod  = self.CAT_MODIFIER.get(cat, 0.0)
        return max(0.0, min(base + mod, 1.0))

    def run(self, or_groups: dict, logger: "ProcessLogger") -> dict:
        """반환: {카테고리: [(kw, score), ...]} desc 정렬"""
        t0     = time.time()
        scored: dict[str, list] = {}

        for cat, kws in or_groups.items():
            pairs = [(kw, round(self.score(kw, cat), 3)) for kw in kws]
            pairs.sort(key=lambda x: -x[1])
            scored[cat] = pairs

            generic  = [kw for kw, s in pairs if s < 0.30]
            specific = [kw for kw, s in pairs if s >= 0.50]
            logger.record_decision(
                agent="TermScoringAgent",
                parameter=f"[{cat}] 특이도",
                value=f"특정={len(specific)}개 / 일반={len(generic)}개",
                reasoning=(
                    f"TOP3={[kw for kw, _ in pairs[:3]]}  "
                    f"LOW3={[kw for kw, _ in pairs[-3:]]}"
                ),
            )

        logger.record(
            agent="TermScoringAgent",
            action="키워드 특이도 분석",
            inputs={"n_categories": len(or_groups),
                    "total_kw": sum(len(v) for v in or_groups.values())},
            outputs={"scored_categories": len(scored)},
            elapsed=time.time() - t0,
        )
        return scored


# ============================================================
# 서브에이전트 3: InterGroupDistanceAgent
# ============================================================

class InterGroupDistanceAgent:
    """
    역할: 키워드 집단 간 거리를 3가지 방법으로 측정

    방법1. TF-IDF 코사인 거리  — 문자 2~3-gram TF-IDF 벡터 코사인 거리
    방법2. 자카드 문자 bigram 거리 — 집단 내 bigram 집합 비교
    방법3. 도메인 우선순위 거리  — BIZ_PRIORITY 절대 차이

    종합 거리 = 방법1 × 0.40 + 방법2 × 0.40 + 방법3 × 0.20

    판정 기준:
      < 0.55 → AND 적합    (공출현 예상)
      0.55 ~ 0.72 → AND 주의  (결과 수 확인 필요)
      ≥ 0.72 → AND 비적합  (결과 극소 예상 → OR 권장)
    """

    AND_OK_THRESH   = 0.55
    AND_WARN_THRESH = 0.72

    BIZ_PRIORITY: dict[str, int] = {
        "AI·챗봇 기술":        10,
        "학습 플랫폼·에듀테크": 9,
        "CTL·교수학습센터":     8,
        "에듀테크 경쟁사":      7,
        "정부 정책·사업비":     6,
        "파트너·고객 대학교":   4,
        "대학 인사·행정":       3,
    }

    # ── 방법1: TF-IDF 코사인 거리 ───────────────────────────────
    def _tfidf_cosine(self, or_groups: dict) -> dict:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_distances

        cats = list(or_groups.keys())
        docs = [" ".join(kws) for kws in or_groups.values()]
        try:
            vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 3), min_df=1)
            mat = cosine_distances(vec.fit_transform(docs))
        except Exception:
            n   = len(cats)
            mat = [[0.0 if i == j else 1.0 for j in range(n)] for i in range(n)]

        result = {}
        for i, ci in enumerate(cats):
            for j, cj in enumerate(cats):
                if i < j:
                    result[(ci, cj)] = float(mat[i][j])
        return result

    # ── 방법2: 자카드 문자 bigram 거리 ──────────────────────────
    def _jaccard_ngram(self, or_groups: dict, n: int = 2) -> dict:
        def ngrams(kws):
            txt = " ".join(kws)
            return set(txt[i:i+n] for i in range(len(txt) - n + 1))

        cats = list(or_groups.keys())
        ng   = {cat: ngrams(kws) for cat, kws in or_groups.items()}

        result = {}
        for i, ci in enumerate(cats):
            for j, cj in enumerate(cats):
                if i < j:
                    a, b  = ng[ci], ng[cj]
                    union = len(a | b)
                    result[(ci, cj)] = 1.0 - (len(a & b) / union if union else 0.0)
        return result

    # ── 방법3: 도메인 우선순위 거리 ─────────────────────────────
    def _domain_dist(self, or_groups: dict) -> dict:
        max_p  = max(self.BIZ_PRIORITY.values())
        cats   = list(or_groups.keys())
        result = {}
        for i, ci in enumerate(cats):
            for j, cj in enumerate(cats):
                if i < j:
                    result[(ci, cj)] = abs(
                        self.BIZ_PRIORITY.get(ci, 5) -
                        self.BIZ_PRIORITY.get(cj, 5)
                    ) / max_p
        return result

    def run(self, or_groups: dict, logger: "ProcessLogger") -> dict:
        t0     = time.time()
        tfidf  = self._tfidf_cosine(or_groups)
        jac    = self._jaccard_ngram(or_groups)
        domain = self._domain_dist(or_groups)

        combined: dict[tuple, float] = {}
        for pair in tfidf:
            combined[pair] = (
                tfidf[pair]             * 0.40
                + jac.get(pair, 1.0)   * 0.40
                + domain.get(pair, 1.0)* 0.20
            )

        and_ok   = [p for p, d in combined.items() if d <  self.AND_OK_THRESH]
        and_warn = [p for p, d in combined.items()
                    if self.AND_OK_THRESH <= d < self.AND_WARN_THRESH]
        and_bad  = [p for p, d in combined.items() if d >= self.AND_WARN_THRESH]

        for pair in sorted(combined, key=combined.get):
            ci, cj = pair
            d = combined[pair]
            verdict = ("AND 적합"  if d < self.AND_OK_THRESH else
                       "AND 주의"  if d < self.AND_WARN_THRESH else
                       "AND 비적합")
            logger.record_decision(
                agent="InterGroupDistanceAgent",
                parameter=f"거리: {ci} ↔ {cj}",
                value=f"{d:.3f} [{verdict}]",
                reasoning=(
                    f"TF-IDF={tfidf[pair]:.3f}, "
                    f"Jaccard={jac.get(pair, 1.0):.3f}, "
                    f"Domain={domain.get(pair, 1.0):.3f}"
                ),
            )

        logger.record(
            agent="InterGroupDistanceAgent",
            action="집단 간 거리 측정 (3가지 방법)",
            inputs={"n_groups": len(or_groups), "n_pairs": len(tfidf)},
            outputs={"and_ok": len(and_ok), "and_warn": len(and_warn),
                     "and_bad": len(and_bad)},
            elapsed=time.time() - t0,
        )
        return {
            "tfidf":    tfidf,    "jaccard":  jac,
            "domain":   domain,   "combined": combined,
            "and_ok":   and_ok,   "and_warn": and_warn,
            "and_bad":  and_bad,
            "threshold_ok":   self.AND_OK_THRESH,
            "threshold_warn": self.AND_WARN_THRESH,
        }


# ============================================================
# 서브에이전트 4: QueryArchitectAgent
# ============================================================

class QueryArchitectAgent:
    """
    역할: 거리 분석 결과로 5개 쿼리 청사진(and_groups 리스트) 자동 설계

    데이터 드리븐 방식:
      1. BIZ 최고 카테고리를 '앵커'로 고정
      2. 앵커 ↔ 각 집단 종합 거리 오름차순으로 근접 집단 선택
      3. 거리가 가까운 쌍 → AND / 먼 쌍 → OR 통합

    5개 후보:
      후보1: 특이도 필터 OR  (일반어 제거, 1 그룹)
      후보2: 앵커 AND 최근접 집단            [2-AND, 가장 좁음]
      후보3: 앵커 AND 차근접 집단            [2-AND]
      후보4: 앵커 AND MERGE(근접1+2)         [2-AND, OR 그룹 확장]
      후보5: 앵커 AND 근접1 AND 근접2        [3-AND, 최정밀]
    """

    BIZ_PRIORITY: dict[str, int] = {
        "AI·챗봇 기술":        10,
        "학습 플랫폼·에듀테크": 9,
        "CTL·교수학습센터":     8,
        "에듀테크 경쟁사":      7,
        "정부 정책·사업비":     6,
        "파트너·고객 대학교":   4,
        "대학 인사·행정":       3,
    }

    OR_SPEC_THRESHOLD = 0.30   # OR 쿼리 특이도 하한
    AND_MAX_KW        = 6      # AND 그룹당 최대 키워드 수

    def _top_kws(self, scored: list, max_n: int) -> list[str]:
        return [kw for kw, _ in scored[:max_n]]

    def run(self, or_groups: dict, scored: dict,
            distances: dict, logger: "ProcessLogger") -> dict:
        t0       = time.time()
        combined = distances["combined"]

        # ── 앵커: BIZ 우선순위 최고 카테고리 ──────────────────────
        anchor = max(or_groups, key=lambda c: self.BIZ_PRIORITY.get(c, 0))

        # ── 앵커 기준 거리 오름차순 (다른 카테고리) ──────────────
        def dist_to_anchor(cat: str) -> float:
            key = (anchor, cat) if (anchor, cat) in combined else (cat, anchor)
            return combined.get(key, 1.0)

        others_ranked = sorted(
            [c for c in or_groups if c != anchor],
            key=dist_to_anchor,
        )

        strategies: dict[str, dict] = {}

        # ── 후보1: 특이도 필터 OR ─────────────────────────────────
        or_kws = [
            kw for cat, pairs in scored.items()
            for kw, s in pairs if s >= self.OR_SPEC_THRESHOLD
        ]
        strategies["후보1 (특이도 OR, 일반어 제거)"] = {
            "and_groups": [{"label": "전체_특이도필터",
                            "cats": list(or_groups), "keywords": or_kws}],
            "cats":        list(or_groups),
            "description": f"특이도 ≥{self.OR_SPEC_THRESHOLD} 키워드만 OR — 일반어 제거",
            "reasoning": (
                f"전체 OR에서 특이도 {self.OR_SPEC_THRESHOLD} 미만 일반어 제거. "
                f"잡음(FP) 감소 기대. 포함={len(or_kws)}개."
            ),
            "dist": None,
        }

        # ── 후보2: 앵커 AND 최근접 ────────────────────────────────
        if len(others_ranked) >= 1:
            near1 = others_ranked[0]
            d1    = dist_to_anchor(near1)
            v1    = ("AND 적합" if d1 < distances["threshold_ok"] else
                     "AND 주의" if d1 < distances["threshold_warn"] else "AND 비적합")
            strategies["후보2 (앵커 AND 최근접)"] = {
                "and_groups": [
                    {"label": anchor, "cats": [anchor],
                     "keywords": self._top_kws(scored.get(anchor, []), self.AND_MAX_KW)},
                    {"label": near1,  "cats": [near1],
                     "keywords": self._top_kws(scored.get(near1,  []), self.AND_MAX_KW)},
                ],
                "cats":        [anchor, near1],
                "description": f"({anchor} TOP{self.AND_MAX_KW}) AND ({near1} TOP{self.AND_MAX_KW})",
                "reasoning": (
                    f"종합거리={d1:.3f} [{v1}]. "
                    f"{anchor}와 가장 가까운 집단 → AND 결과 가장 많을 것으로 예상."
                ),
                "dist": d1,
            }

        # ── 후보3: 앵커 AND 차근접 ────────────────────────────────
        if len(others_ranked) >= 2:
            near2 = others_ranked[1]
            d2    = dist_to_anchor(near2)
            v2    = ("AND 적합" if d2 < distances["threshold_ok"] else
                     "AND 주의" if d2 < distances["threshold_warn"] else "AND 비적합")
            strategies["후보3 (앵커 AND 차근접)"] = {
                "and_groups": [
                    {"label": anchor, "cats": [anchor],
                     "keywords": self._top_kws(scored.get(anchor, []), self.AND_MAX_KW)},
                    {"label": near2,  "cats": [near2],
                     "keywords": self._top_kws(scored.get(near2,  []), self.AND_MAX_KW)},
                ],
                "cats":        [anchor, near2],
                "description": f"({anchor} TOP{self.AND_MAX_KW}) AND ({near2} TOP{self.AND_MAX_KW})",
                "reasoning": (
                    f"종합거리={d2:.3f} [{v2}]. "
                    f"후보2와 목적·대상 집단 다름 — 다양한 커버리지."
                ),
                "dist": d2,
            }

        # ── 후보4: 앵커 AND MERGE(근접1+2) ───────────────────────
        if len(others_ranked) >= 2:
            near1, near2 = others_ranked[0], others_ranked[1]
            half = self.AND_MAX_KW // 2
            merged_kws = (self._top_kws(scored.get(near1, []), half)
                          + self._top_kws(scored.get(near2, []), half))
            d_avg = (dist_to_anchor(near1) + dist_to_anchor(near2)) / 2
            strategies["후보4 (앵커 AND 근접집단 통합 OR)"] = {
                "and_groups": [
                    {"label": anchor,
                     "cats": [anchor],
                     "keywords": self._top_kws(scored.get(anchor, []), self.AND_MAX_KW)},
                    {"label": f"{near1}+{near2}",
                     "cats": [near1, near2],
                     "keywords": merged_kws},
                ],
                "cats":        [anchor, near1, near2],
                "description": (
                    f"({anchor} TOP{self.AND_MAX_KW}) AND "
                    f"({near1} TOP{half} OR {near2} TOP{half} 통합)"
                ),
                "reasoning": (
                    f"근접1({near1}, d={dist_to_anchor(near1):.3f})과 "
                    f"근접2({near2}, d={dist_to_anchor(near2):.3f})를 OR로 통합. "
                    f"AND 그룹 커버리지 확대 → 후보2·3보다 결과 많음. 평균거리={d_avg:.3f}."
                ),
                "dist": d_avg,
            }

        # ── 후보5: 트리플 AND ─────────────────────────────────────
        if len(others_ranked) >= 2:
            near1, near2 = others_ranked[0], others_ranked[1]
            d_max = max(dist_to_anchor(near1), dist_to_anchor(near2))
            v5    = ("AND 적합" if d_max < distances["threshold_ok"] else
                     "AND 주의" if d_max < distances["threshold_warn"] else "AND 비적합")
            strategies["후보5 (트리플 AND, 최정밀)"] = {
                "and_groups": [
                    {"label": anchor, "cats": [anchor],
                     "keywords": self._top_kws(scored.get(anchor, []), self.AND_MAX_KW)},
                    {"label": near1,  "cats": [near1],
                     "keywords": self._top_kws(scored.get(near1,  []), self.AND_MAX_KW)},
                    {"label": near2,  "cats": [near2],
                     "keywords": self._top_kws(scored.get(near2,  []), self.AND_MAX_KW)},
                ],
                "cats":        [anchor, near1, near2],
                "description": (
                    f"({anchor}) AND ({near1}) AND ({near2}) — 트리플 AND"
                ),
                "reasoning": (
                    f"3집단 동시 AND. 최대거리={d_max:.3f} [{v5}]. "
                    f"가장 정밀하지만 결과 수 가장 적음. "
                    f"0건 시 후보2·3으로 완화 권장."
                ),
                "dist": d_max,
            }

        elapsed = time.time() - t0
        logger.record(
            agent="QueryArchitectAgent",
            action="거리 기반 5개 쿼리 청사진 설계 (데이터 드리븐)",
            inputs={"anchor": anchor, "others_ranked": others_ranked,
                    "and_ok_pairs": len(distances["and_ok"])},
            outputs={"n_strategies": len(strategies), "anchor": anchor,
                     "nearest2": others_ranked[:2]},
            elapsed=elapsed,
        )
        return strategies


# ============================================================
# 서브에이전트 5: QueryRefinementAgent
# ============================================================

class QueryRefinementAgent:
    """
    역할: and_groups 청사진 → 실제 검색식 문자열 포맷팅

    포맷:
      1 그룹 → kw1 OR kw2 OR ...
      N 그룹 → (kw1 OR kw2) AND (kw3 OR kw4) AND ...
    """

    def run(self, strategies: dict, logger: "ProcessLogger") -> dict:
        t0      = time.time()
        refined: dict[str, dict] = {}

        for name, strat in strategies.items():
            groups = strat["and_groups"]

            if len(groups) == 1:
                query     = " OR ".join(groups[0]["keywords"])
                and_count = 0
            else:
                parts: list[str] = []
                for grp in groups:
                    kws = grp["keywords"]
                    if not kws:
                        continue
                    parts.append(kws[0] if len(kws) == 1
                                 else "(" + " OR ".join(kws) + ")")
                query     = " AND ".join(parts)
                and_count = len(parts) - 1

            or_count = query.count(" OR ")
            kw_count = or_count + and_count + 1 if query else 0

            refined[name] = {
                "query":       query,
                "kw_count":    kw_count,
                "and_count":   and_count,
                "or_count":    or_count,
                "n_groups":    len(groups),
                "cats_used":   strat.get("cats", []),
                "description": strat["description"],
                "reasoning":   strat["reasoning"],
                "dist":        strat.get("dist"),
            }

        elapsed = time.time() - t0
        pure_or = sum(1 for q in refined.values() if q["and_count"] == 0)
        logger.record(
            agent="QueryRefinementAgent",
            action="쿼리 문자열 포맷팅",
            inputs={"n_strategies": len(strategies)},
            outputs={"n_queries": len(refined),
                     "pure_or": pure_or, "and_mixed": len(refined) - pure_or},
            elapsed=elapsed,
        )
        return refined


# ============================================================
# 서브에이전트 6: QueryOutputAgent
# ============================================================

class QueryOutputAgent:
    """
    역할: 유효성 검증 + 거리 경고 태그 부착

    기준:
      순수 OR → kw ≤ 100  (50개 초과 시 경고)
      AND 혼합 → kw ≤ 30
      AND 거리 ≥ threshold_warn → 결과 극소 경고
    """

    MAX_KW_OR  = 100
    WARN_OR    = 50
    MAX_KW_AND = 30

    def run(self, refined: dict, distances: dict,
            logger: "ProcessLogger") -> dict:
        validated: dict[str, dict] = {}

        for name, q in refined.items():
            issues   = []
            warnings = []

            if q["kw_count"] == 0:
                issues.append("검색식이 비어 있음")
            elif q["and_count"] == 0:
                if q["kw_count"] > self.MAX_KW_OR:
                    issues.append(f"키워드 {q['kw_count']}개 > 상한 {self.MAX_KW_OR}")
                elif q["kw_count"] > self.WARN_OR:
                    warnings.append(f"키워드 {q['kw_count']}개: 결과 과다 주의")
            else:
                if q["kw_count"] > self.MAX_KW_AND:
                    warnings.append(f"AND 검색식 {q['kw_count']}개 > 권장 {self.MAX_KW_AND}")
                d = q.get("dist")
                if d is not None:
                    if d >= distances["threshold_warn"]:
                        warnings.append(
                            f"집단 간 거리={d:.3f} ≥ {distances['threshold_warn']}"
                            f" — 결과 극소 가능 (AND 비적합)"
                        )
                    elif d >= distances["threshold_ok"]:
                        warnings.append(
                            f"집단 간 거리={d:.3f} — 결과 수 확인 필요 (AND 주의)"
                        )

            status = "PASS" if not issues else "FAIL"
            validated[name] = {**q, "status": status,
                                "issues": issues, "warnings": warnings}

            and_tag = f"AND{q['and_count']}" if q["and_count"] else "순수OR"
            logger.record_decision(
                agent="QueryOutputAgent",
                parameter=f"검증: {name}",
                value=status,
                reasoning=(
                    f"[{and_tag}] kw={q['kw_count']}, OR={q['or_count']}. "
                    + ("; ".join(issues + warnings) or "이상 없음")
                ),
            )

        logger.record(
            agent="QueryOutputAgent",
            action="최종 검색식 유효성 검증",
            inputs={"n_queries": len(refined)},
            outputs={
                "passed": sum(1 for r in validated.values() if r["status"] == "PASS"),
                "failed": sum(1 for r in validated.values() if r["status"] == "FAIL"),
            },
        )
        return validated


# ============================================================
# 오케스트레이터: QueryCoordinator
# ============================================================

class QueryCoordinator:
    """7개 쿼리 전문 서브에이전트 오케스트레이션."""

    def __init__(self):
        self.merger    = KeywordMergerAgent()
        self.builder   = QueryBuilderAgent()
        self.scorer    = TermScoringAgent()
        self.distancer = InterGroupDistanceAgent()
        self.architect = QueryArchitectAgent()
        self.refiner   = QueryRefinementAgent()
        self.output    = QueryOutputAgent()

    def run(self, r_a: dict, r_b: dict, r_c: dict, r_d: dict,
            logger: "ProcessLogger") -> tuple:
        """Returns (validated_queries, distances)"""
        logger.record("QueryCoordinator", "Phase 6 v2 파이프라인 시작",
                      inputs={"methods": ["A", "B", "C", "D"]})

        print("\n  [Agent 0] KeywordMergerAgent: A/B/C/D 통합 + 합의도 계산...")
        assigned, consensus = self.merger.run(r_a, r_b, r_c, r_d, logger)
        tops = sorted(consensus.items(), key=lambda x: -x[1])
        print(f"    → {len(assigned)}개 카테고리, 합의도 1위: "
              f"{tops[0][0]}({tops[0][1]:.2f})")

        print("  [Agent 1] QueryBuilderAgent: OR 묶음 생성...")
        or_groups = self.builder.run(assigned, logger)
        print(f"    → {len(or_groups)}개 OR 그룹")

        print("  [Agent 2] TermScoringAgent: 키워드 특이도 분석...")
        scored = self.scorer.run(or_groups, logger)
        for cat, pairs in scored.items():
            n_generic  = sum(1 for _, s in pairs if s < 0.30)
            n_specific = len(pairs) - n_generic
            print(f"    {cat[:16]:16s}: 특정={n_specific}개 / 일반={n_generic}개  "
                  f"(TOP: {pairs[0][0] if pairs else '-'})")

        print("\n  [Agent 3] InterGroupDistanceAgent: 집단 간 거리 측정 (3방법)...")
        distances = self.distancer.run(or_groups, logger)
        self._print_distance_matrix(distances, or_groups)

        print("  [Agent 4] QueryArchitectAgent: 거리 기반 5개 쿼리 설계...")
        strategies = self.architect.run(or_groups, scored, distances, logger)
        print(f"    → {len(strategies)}개 쿼리 청사진 완성")

        print("  [Agent 5] QueryRefinementAgent: 검색식 포맷팅...")
        refined = self.refiner.run(strategies, logger)

        print("  [Agent 6] QueryOutputAgent: 유효성 검증...")
        validated = self.output.run(refined, distances, logger)
        passed = sum(1 for r in validated.values() if r["status"] == "PASS")
        print(f"    → {passed}/{len(validated)}개 검증 통과")

        logger.record("QueryCoordinator", "Phase 6 v2 파이프라인 완료",
                      outputs={"queries_passed": passed})
        return validated, distances

    @staticmethod
    def _print_distance_matrix(distances: dict, or_groups: dict):
        combined = distances["combined"]
        ok_t  = distances["threshold_ok"]
        wn_t  = distances["threshold_warn"]
        print(f"\n  ── 집단 간 거리 행렬 (TF-IDF / Jaccard / Domain / 종합 | 판정) ──")
        for pair in sorted(combined, key=combined.get):
            ci, cj = pair
            d  = combined[pair]
            tf = distances["tfidf"].get(pair, 1.0)
            jc = distances["jaccard"].get(pair, 1.0)
            dm = distances["domain"].get(pair, 1.0)
            verdict = ("✓ AND 적합"  if d < ok_t else
                       "△ AND 주의"  if d < wn_t else
                       "✗ AND 비적합")
            print(f"  {ci[:9]:9s} ↔ {cj[:9]:9s} : "
                  f"{tf:.3f} / {jc:.3f} / {dm:.3f} / {d:.3f} | {verdict}")
        n_ok   = len(distances["and_ok"])
        n_warn = len(distances["and_warn"])
        n_bad  = len(distances["and_bad"])
        print(f"  → AND 적합 {n_ok}쌍 / AND 주의 {n_warn}쌍 / AND 비적합 {n_bad}쌍\n")


# ============================================================
# 최종 출력 + 파일 저장
# ============================================================

def save_query_candidates(validated: dict, distances: dict,
                          out_dir: str = "report") -> str:
    """단일 아카이브 파일에 누적 저장."""
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    archive_path = f"{out_dir}/query_candidates_archive.txt"
    run_ts       = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    ok_t         = distances["threshold_ok"]
    wn_t         = distances["threshold_warn"]

    # 거리 행렬 요약
    dist_lines = [
        "  집단 간 거리 행렬 (TF-IDF / Jaccard / Domain / 종합 | 판정)",
        f"  임계값: AND 적합 < {ok_t} / AND 주의 < {wn_t} / AND 비적합 ≥ {wn_t}",
    ]
    for pair in sorted(distances["combined"], key=distances["combined"].get):
        ci, cj = pair
        d  = distances["combined"][pair]
        tf = distances["tfidf"].get(pair, 1.0)
        jc = distances["jaccard"].get(pair, 1.0)
        dm = distances["domain"].get(pair, 1.0)
        v  = ("✓ AND 적합" if d < ok_t else
              "△ AND 주의" if d < wn_t else "✗ AND 비적합")
        dist_lines.append(
            f"  {ci[:10]:10s} ↔ {cj[:10]:10s} : "
            f"{tf:.3f} / {jc:.3f} / {dm:.3f} / {d:.3f}  {v}"
        )

    lines = [
        "",
        "=" * 72,
        f"  [RUN {run_ts}]  Phase 6 v2: 쿼리 전문 서브에이전트",
        "=" * 72,
        "",
        *dist_lines,
        "",
        "-" * 72,
    ]
    for name, q in validated.items():
        warn_str  = ("  경고: " + "; ".join(q["warnings"])) if q["warnings"] else ""
        and_label = f"AND {q['and_count']}개" if q["and_count"] else "순수 OR"
        d_str     = f"  거리={q['dist']:.3f}" if q.get("dist") else ""
        lines += [
            "",
            f"【{name}】  [{and_label}]{d_str}  {q['status']}{warn_str}",
            f"  설명   : {q['description']}",
            f"  통계   : kw={q['kw_count']}개 | OR={q['or_count']} | AND={q['and_count']}",
            f"  카테고리: {' > '.join(q['cats_used'])}",
            "",
            f"  검색식 :",
            f"    {q['query']}",
            "",
            f"  근거   : {q['reasoning']}",
            "",
            "-" * 72,
        ]

    lines += [
        "",
        "[ 빅카인즈 사용 권장 ]",
        f"  AND 적합({len(distances['and_ok'])}쌍) 후보 우선 시도",
        f"  AND 주의({len(distances['and_warn'])}쌍): 결과 수 반드시 확인",
        f"  결과 0건 → 후보1(OR) 복귀 후 범위 좁혀 재시도",
    ]

    with open(archive_path, "a", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print(f"\n  [Phase 6] 검색식 아카이브 → {archive_path}")
    return archive_path


def print_query_candidates(validated: dict, distances: dict):
    ok_t = distances["threshold_ok"]
    wn_t = distances["threshold_warn"]

    print("\n" + "=" * 72)
    print("  Phase 6 v2: 쿼리 전문 서브에이전트 결과")
    print("=" * 72)

    for name, q in validated.items():
        mark  = "✓" if q["status"] == "PASS" else "✗"
        warns = f"  ⚠ {'; '.join(q['warnings'])}" if q["warnings"] else ""
        and_l = f"AND{q['and_count']}" if q["and_count"] else "순수OR"
        d     = q.get("dist")
        d_str = f"  dist={d:.3f}" if d is not None else ""
        print(f"\n【{name}】  [{and_l}]{d_str}  {mark}{warns}")
        print(f"  설명   : {q['description']}")
        print(f"  통계   : OR={q['or_count']} | AND={q['and_count']} | 총={q['kw_count']}")
        print(f"  검색식 : {q['query']}")
        print(f"  근거   : {q['reasoning'][:80]}...")

    print("\n" + "=" * 72)
    print(f"  AND 적합({len(distances['and_ok'])}쌍) | "
          f"AND 주의({len(distances['and_warn'])}쌍) | "
          f"AND 비적합({len(distances['and_bad'])}쌍)")
    print(f"  임계값: 적합 < {ok_t} / 주의 < {wn_t}")
    print("=" * 72)


# ============================================================
# 공개 인터페이스
# ============================================================

def run_phase6(r_a: dict, r_b: dict, r_c: dict, r_d: dict,
               logger: "ProcessLogger") -> dict:
    """
    Phase 6 실행 진입점.

    Parameters
    ----------
    r_a, r_b, r_c, r_d : dict   각 method_x() 반환값 (A/B/C/D 모두 필요)
    logger              : ProcessLogger   호출자가 생성한 로거 (이어서 기록)

    Returns
    -------
    validated  : dict  {전략명: {query, status, cats_used, ...}}
    distances  : dict  {pairs, and_ok, and_warn, and_bad, matrix}
    """
    coordinator = QueryCoordinator()
    validated, distances = coordinator.run(r_a, r_b, r_c, r_d, logger)
    return validated, distances
