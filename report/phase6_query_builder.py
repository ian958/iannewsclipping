"""
Phase 6: 검색식 생성 - Subagent 기반 AND/OR 쿼리 빌더
================================================================
입력  : 클러스터링 결과 (A/B/C/D 4개 방법 모두)
출력  : 최종 검색식 후보 3개 (모두 OR 기반, 범위 순)
기록  : ProcessLogger → JSON + TXT 파일 자동 저장

서브에이전트 구성
  KeywordMergerAgent  : A/B/C/D 4개 결과 통합 + 카테고리별 합의도 계산
  QueryBuilderAgent   : 카테고리별 OR 키워드 묶음 생성 + 길이 필터
  StrategyAgent       : 합의도+비즈니스 점수로 카테고리 순위 결정
                        → 전체 / 핵심5 / 정밀3 OR 전략 3개 생성
  QueryFormatterAgent : 검색식 문자열 포맷팅 (OR 전용)
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
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        Path(out_dir).mkdir(parents=True, exist_ok=True)

        json_path = f"{out_dir}/process_log_{ts}.json"
        payload = {
            "version": self.VERSION,
            "session": self.session_name,
            "started_at": self.started_at,
            "completed_at": datetime.now().isoformat(),
            "total_events": len(self.events),
            "total_decisions": len(self.decisions),
            "events": self.events,
            "decisions": self.decisions,
        }
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

        txt_path = f"{out_dir}/process_log_{ts}.txt"
        lines = [
            f"{'='*70}",
            f"  테스트 프로세스 기록",
            f"  세션: {self.session_name}  |  버전: {self.VERSION}",
            f"  시작: {self.started_at}",
            f"  완료: {payload['completed_at']}",
            f"{'='*70}",
            "",
            "[ 에이전트 실행 이벤트 ]",
        ]
        for e in self.events:
            elapsed_str = f" ({e['elapsed_sec']}s)" if e["elapsed_sec"] else ""
            lines.append(f"  {e['ts']}  [{e['agent']}]  {e['action']}{elapsed_str}")
            if e["outputs"]:
                lines.append(f"    출력: {e['outputs']}")
        lines += ["", "[ 의사결정 기록 ]"]
        for d in self.decisions:
            lines.append(f"  [{d['agent']}]  {d['parameter']} = {d['value']}")
            lines.append(f"    근거: {d['reasoning']}")
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

        print(f"\n  [ProcessLogger] JSON 저장 → {json_path}")
        print(f"  [ProcessLogger] TXT  저장 → {txt_path}")
        return json_path, txt_path


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
# 서브에이전트 2: StrategyAgent
# ============================================================

class StrategyAgent:
    """
    역할  : 합의도 + 비즈니스 관련성 점수로 카테고리 순위 결정
            → OR 3개 + AND 2개 = 총 5개 전략 생성
    전략  :
      후보1_OR_전체   : 모든 카테고리 키워드 OR (최대 수집)
      후보2_OR_핵심   : 합의도+비즈니스 상위 5개 카테고리 OR
      후보3_OR_정밀   : 합의도+비즈니스 상위 3개 카테고리 OR
      후보4_AND_균형  : (에듀테크·AI 핵심) AND (교육정책 핵심)
      후보5_AND_정밀  : (에듀테크·AI 핵심) AND (경쟁사 핵심)

    AND 후보의 0건 방지: 각 AND 그룹은 ≤2단어 단어의 핵심 키워드만 사용.
    """

    # 비즈니스 관련성 우선순위 (도메인 지식 기반, 높을수록 핵심)
    BIZ_PRIORITY: dict[str, int] = {
        "AI·챗봇 기술":        10,
        "학습 플랫폼·에듀테크": 9,
        "CTL·교수학습센터":     8,
        "에듀테크 경쟁사":      7,
        "정부 정책·사업비":     6,
        "파트너·고객 대학교":   4,
        "대학 인사·행정":       3,
    }

    # AND 전략에서 사용할 카테고리 조합 (카테고리명 직접 매칭, 인덱스 무관)
    AND_COMBOS = [
        {
            "name": "후보4_AND_균형 (에듀테크×정책)",
            "groups": [
                {
                    "label":     "에듀테크·AI",
                    "cat_names": ["AI·챗봇 기술", "학습 플랫폼·에듀테크"],
                },
                {
                    "label":     "교육정책",
                    "cat_names": ["정부 정책·사업비"],
                },
            ],
            "description": "에듀테크·AI 핵심어 AND 교육정책 핵심어",
            "reasoning": (
                "제품(AI챗봇·LMS)과 교육정책(교육부·RISE·글로컬)이 "
                "동시 언급된 기사 수집. 정책 연계 비즈니스 기회 탐색에 적합. "
                "각 AND 그룹은 ≤2단어 핵심어만 사용 → 0건 방지."
            ),
        },
        {
            "name": "후보5_AND_정밀 (에듀테크×경쟁사)",
            "groups": [
                {
                    "label":     "에듀테크·AI",
                    "cat_names": ["AI·챗봇 기술", "학습 플랫폼·에듀테크"],
                },
                {
                    "label":     "경쟁사",
                    "cat_names": ["에듀테크 경쟁사"],
                },
            ],
            "description": "에듀테크·AI 핵심어 AND 경쟁사 핵심어",
            "reasoning": (
                "제품(AI챗봇·LMS)과 경쟁사가 함께 등장하는 기사 수집. "
                "시장 비교·경쟁 동향 파악에 특화. "
                "각 AND 그룹은 ≤2단어 핵심어만 사용 → 0건 방지."
            ),
        },
    ]

    @staticmethod
    def _simplify_for_and(kws: list[str], max_words: int = 2,
                          max_n: int = 6) -> list[str]:
        """AND 그룹용 단순화: ≤max_words 단어 키워드만, 짧은 것(범용) 우선, 최대 max_n개"""
        short = [k for k in kws if len(k.split()) <= max_words]
        short.sort(key=lambda k: (len(k.split()), len(k)))   # 단어수 → 글자수 오름차순
        return short[:max_n] if short else sorted(kws, key=len)[:max_n]

    def run(self, assigned: dict, consensus_scores: dict,
            or_groups: dict, logger: ProcessLogger) -> dict:
        t0 = time.time()

        max_prio = max(self.BIZ_PRIORITY.values()) if self.BIZ_PRIORITY else 10

        # ── 카테고리별 최종 점수 = 비즈니스 우선순위 70% + 합의도 30% ──
        cat_scores: dict[str, float] = {}
        for cat in assigned:
            prio  = self.BIZ_PRIORITY.get(cat, 2)
            cons  = consensus_scores.get(cat, 0.0)
            cat_scores[cat] = (prio / max_prio) * 0.7 + cons * 0.3

        ranked = sorted(cat_scores, key=lambda c: -cat_scores[c])

        # ── OR 전략 3개 ────────────────────────────────────────────
        top5 = ranked[:5]
        top3 = ranked[:3]

        strategies: dict[str, dict] = {
            "후보1_OR_전체 (최대 수집)": {
                "type": "OR",
                "cats": ranked,
                "or_groups": {c: or_groups[c] for c in ranked if c in or_groups},
                "description": "A/B/C/D 4개 방법 통합 키워드 전체 OR — 최대 수집",
                "reasoning": (
                    "4개 방법의 모든 키워드를 포함하여 누락(FN) 최소화. "
                    "일일 뉴스 건수 파악 및 임계값 설정에 적합."
                ),
            },
            "후보2_OR_핵심 (비즈니스 핵심)": {
                "type": "OR",
                "cats": top5,
                "or_groups": {c: or_groups[c] for c in top5 if c in or_groups},
                "description": (
                    f"합의도+비즈니스 점수 상위 5개 카테고리 OR "
                    f"({' | '.join(top5)})"
                ),
                "reasoning": (
                    "카테고리 점수 = 비즈니스 관련성 70% + 합의도 30%. "
                    "하위 카테고리 제외로 잡음 감소. 일상 모니터링 권장."
                ),
            },
            "후보3_OR_정밀 (제품집중)": {
                "type": "OR",
                "cats": top3,
                "or_groups": {c: or_groups[c] for c in top3 if c in or_groups},
                "description": (
                    f"합의도+비즈니스 점수 상위 3개 카테고리 OR "
                    f"({' | '.join(top3)})"
                ),
                "reasoning": (
                    "제품 직접 관련 상위 3개 카테고리. 고관련도 기사 정밀 수집."
                ),
            },
        }

        # ── AND 전략 2개 (카테고리명 직접 매칭) ────────────────────
        for combo in self.AND_COMBOS:
            and_groups_built: list[dict] = []
            all_cats_used: list[str] = []

            for grp in combo["groups"]:
                merged_kws: list[str] = []
                cats_in_grp: list[str] = []
                # cat_names 로 or_groups 에서 직접 조회
                for cat_name in grp["cat_names"]:
                    if cat_name in or_groups:
                        cats_in_grp.append(cat_name)
                        all_cats_used.append(cat_name)
                        merged_kws.extend(or_groups[cat_name])

                simplified = self._simplify_for_and(merged_kws)
                and_groups_built.append({
                    "label":    grp["label"],
                    "cats":     cats_in_grp,
                    "keywords": simplified,
                })

            strategies[combo["name"]] = {
                "type":        "AND",
                "cats":        all_cats_used,
                "and_groups":  and_groups_built,
                "description": combo["description"],
                "reasoning":   combo["reasoning"],
            }

        elapsed = time.time() - t0

        # ── 로깅 ────────────────────────────────────────────────────
        for cat in ranked:
            logger.record_decision(
                agent="StrategyAgent",
                parameter=f"카테고리 순위: {cat}",
                value=f"{cat_scores[cat]:.3f}",
                reasoning=(
                    f"비즈니스우선순위={self.BIZ_PRIORITY.get(cat, 2)}, "
                    f"합의도={consensus_scores.get(cat, 0.0):.3f}, "
                    f"최종점수={cat_scores[cat]:.3f}"
                ),
            )

        logger.record(
            agent="StrategyAgent",
            action="OR 3개 + AND 2개 = 총 5개 전략 결정",
            inputs={"n_categories": len(assigned), "ranked_categories": ranked},
            outputs={"OR후보": 3, "AND후보": 2, "total": len(strategies)},
            elapsed=elapsed,
        )
        return strategies


# ============================================================
# 서브에이전트 3: QueryFormatterAgent
# ============================================================

class QueryFormatterAgent:
    """
    역할  : 전략 딕셔너리를 실제 검색식 문자열로 포맷팅
    지원  : OR 전용 (type=OR) / AND+OR 조합 (type=AND)
    AND 포맷: (kw1 OR kw2 ...) AND (kw3 OR kw4 ...)
    출력  : strategy_name → {query, and_count, or_count, kw_count, ...}
    """

    def run(self, strategies: dict, logger: ProcessLogger) -> dict:
        t0 = time.time()
        formatted: dict[str, dict] = {}

        for name, strat in strategies.items():
            if strat["type"] == "OR":
                # ── 순수 OR ──────────────────────────────────────
                all_kws: list[str] = []
                for kws in strat["or_groups"].values():
                    all_kws.extend(kws)
                query = " OR ".join(all_kws)
                and_count = 0

            else:
                # ── AND + 내부 OR 그룹 ────────────────────────────
                parts: list[str] = []
                for grp in strat["and_groups"]:
                    kws = grp["keywords"]
                    if not kws:
                        continue
                    if len(kws) == 1:
                        parts.append(kws[0])
                    else:
                        parts.append("(" + " OR ".join(kws) + ")")
                query     = " AND ".join(parts)
                and_count = query.count(" AND ")

            or_count = query.count(" OR ")
            kw_count = or_count + and_count + 1 if query else 0

            formatted[name] = {
                "query":       query,
                "kw_count":    kw_count,
                "and_count":   and_count,
                "or_count":    or_count,
                "type":        strat["type"],
                "cats_used":   strat["cats"],
                "description": strat["description"],
                "reasoning":   strat["reasoning"],
            }

        elapsed = time.time() - t0
        logger.record(
            agent="QueryFormatterAgent",
            action="OR/AND 혼합 검색식 문자열 포맷팅",
            inputs={"n_strategies": len(strategies)},
            outputs={"n_queries": len(formatted),
                     "or_only": sum(1 for s in strategies.values() if s["type"] == "OR"),
                     "and_mixed": sum(1 for s in strategies.values() if s["type"] == "AND")},
            elapsed=elapsed,
        )
        return formatted


# ============================================================
# 서브에이전트 4: ValidatorAgent
# ============================================================

class ValidatorAgent:
    """
    역할  : 검색식 유효성·복잡도 검증
    기준  : OR 전용 → 키워드 수 ≤ 100
            AND 포함 → AND 그룹당 키워드 ≤ 6, 전체 키워드 ≤ 30
    """

    MAX_KW_OR  = 100
    MAX_KW_AND = 30
    WARN_OR    = 50

    def run(self, queries: dict, logger: ProcessLogger) -> dict:
        validated: dict[str, dict] = {}

        for name, q in queries.items():
            issues   = []
            warnings = []
            qtype    = q.get("type", "OR")

            if q["kw_count"] == 0:
                issues.append("검색식이 비어 있음")
            elif qtype == "OR":
                if q["kw_count"] > self.MAX_KW_OR:
                    issues.append(f"키워드 {q['kw_count']}개 > 권장 {self.MAX_KW_OR}개")
                if q["kw_count"] > self.WARN_OR:
                    warnings.append(f"키워드 {q['kw_count']}개: 결과 과다 주의")
            else:  # AND
                if q["kw_count"] > self.MAX_KW_AND:
                    warnings.append(
                        f"AND 검색식 키워드 {q['kw_count']}개 > 권장 {self.MAX_KW_AND}개"
                    )
                if q["and_count"] == 0:
                    issues.append("AND 전략인데 AND 연산자 없음")

            status = "PASS" if not issues else "FAIL"
            validated[name] = {**q, "status": status,
                                "issues": issues, "warnings": warnings}

            logger.record_decision(
                agent="ValidatorAgent",
                parameter=f"검증: {name}",
                value=status,
                reasoning=(
                    f"type={qtype}, kw={q['kw_count']}, "
                    f"AND={q['and_count']}, OR={q['or_count']}. "
                    + ("; ".join(issues + warnings) or "이상 없음")
                ),
            )

        logger.record(
            agent="ValidatorAgent",
            action="검색식 유효성 검증 (OR/AND 혼합)",
            inputs={"n_queries": len(queries)},
            outputs={
                "passed": sum(1 for r in validated.values() if r["status"] == "PASS"),
                "failed": sum(1 for r in validated.values() if r["status"] == "FAIL"),
            },
        )
        return validated


# ============================================================
# QueryCoordinator (오케스트레이터)
# ============================================================

class QueryCoordinator:
    """
    5개 서브에이전트를 순서대로 실행하고 결과를 취합.
    각 단계 결과는 ProcessLogger에 자동 기록됨.
    """

    def __init__(self):
        self.merger    = KeywordMergerAgent()
        self.builder   = QueryBuilderAgent()
        self.strategist = StrategyAgent()
        self.formatter = QueryFormatterAgent()
        self.validator = ValidatorAgent()

    def run(self, r_a: dict, r_b: dict, r_c: dict, r_d: dict,
            logger: ProcessLogger) -> dict:
        logger.record("QueryCoordinator", "Phase 6 파이프라인 시작",
                      inputs={"methods": ["A", "B", "C", "D"]})

        print("\n  [Agent 0] KeywordMergerAgent: A/B/C/D 결과 통합 + 합의도 계산 중...")
        assigned, consensus_scores = self.merger.run(r_a, r_b, r_c, r_d, logger)
        cats_sorted = sorted(consensus_scores.items(), key=lambda x: -x[1])
        print(f"    → {len(assigned)}개 카테고리, 합의도 상위: "
              f"{cats_sorted[0][0]}({cats_sorted[0][1]:.2f})")

        print("  [Agent 1] QueryBuilderAgent: OR 묶음 생성 중...")
        or_groups = self.builder.run(assigned, logger)
        print(f"    → {len(or_groups)}개 OR 그룹 생성")

        print("  [Agent 2] StrategyAgent: OR 3개 + AND 2개 = 총 5개 전략 결정 중...")
        strategies = self.strategist.run(assigned, consensus_scores, or_groups, logger)
        or_n  = sum(1 for s in strategies.values() if s["type"] == "OR")
        and_n = sum(1 for s in strategies.values() if s["type"] == "AND")
        print(f"    → {len(strategies)}개 전략 확정 (OR={or_n}, AND={and_n})")

        print("  [Agent 3] QueryFormatterAgent: 검색식 문자열 포맷팅 중...")
        formatted = self.formatter.run(strategies, logger)
        print(f"    → {len(formatted)}개 검색식 생성")

        print("  [Agent 4] ValidatorAgent: 유효성 검증 중...")
        validated = self.validator.run(formatted, logger)
        passed = sum(1 for r in validated.values() if r["status"] == "PASS")
        print(f"    → {passed}/{len(validated)}개 검증 통과")

        logger.record("QueryCoordinator", "Phase 6 파이프라인 완료",
                      outputs={"queries_passed": passed})
        return validated


# ============================================================
# 최종 출력 + 파일 저장
# ============================================================

def save_query_candidates(validated: dict, out_dir: str = "report") -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    path = f"{out_dir}/phase6_query_candidates_{ts}.txt"

    lines = [
        "=" * 70,
        "  Phase 6: 최종 검색식 후보 5개 (OR 3개 + AND 2개)",
        f"  생성일시: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "=" * 70,
        "",
    ]
    for name, q in validated.items():
        warn_str = ("  경고: " + "; ".join(q["warnings"])) if q["warnings"] else ""
        qtype    = q.get("type", "OR")
        lines += [
            f"【{name}】  [{qtype}]",
            f"  설명  : {q['description']}",
            f"  상태  : {q['status']}{warn_str}",
            f"  통계  : 키워드 {q['kw_count']}개 | OR {q['or_count']}개 | AND {q['and_count']}개",
            f"  카테고리: {' > '.join(q['cats_used'])}",
            "",
            f"  검색식:",
            f"    {q['query']}",
            "",
            f"  선택 근거:",
            f"    {q['reasoning']}",
            "",
            "-" * 70,
        ]

    lines += [
        "",
        "[ 빅카인즈 사용 권장 순서 ]",
        "  [OR 후보]",
        "  1단계: 후보2_OR_핵심 으로 1주일 수집 → 일평균 건수 확인",
        "  2단계: 건수 부족(< 5건/일) → 후보1_OR_전체 전환",
        "         건수 과다(> 50건/일) → 후보3_OR_정밀 전환",
        "  [AND 후보]",
        "  정책 연계 기사 탐색 → 후보4_AND_균형",
        "  경쟁사 언급 기사 탐색 → 후보5_AND_정밀",
        "  3단계: 2주 운영 후 Phase 5 메트릭(Precision/Recall) 평가",
    ]

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"\n  [Phase 6] 검색식 후보 저장 → {path}")
    return path


def print_query_candidates(validated: dict):
    print("\n" + "=" * 70)
    print("  Phase 6 결과: 최종 검색식 후보 5개 (OR 3개 + AND 2개)")
    print("=" * 70)

    for name, q in validated.items():
        status_mark = "✓" if q["status"] == "PASS" else "✗"
        warn_str    = f"  ⚠ {'; '.join(q['warnings'])}" if q["warnings"] else ""
        qtype       = q.get("type", "OR")
        print(f"\n【{name}】  [{qtype}]  {status_mark}{warn_str}")
        print(f"  설명      : {q['description']}")
        print(f"  카테고리  : {' > '.join(q['cats_used'])}")
        print(f"  통계      : OR={q['or_count']}개 / AND={q['and_count']}개 / 총={q['kw_count']}개")
        print(f"  검색식    : {q['query']}")
        print(f"  근거      : {q['reasoning'][:90]}...")

    print("\n" + "=" * 70)
    print("  빅카인즈 사용 권장 순서")
    print("=" * 70)
    print("  [OR] 후보2_OR_핵심 → 1주일 수집, 일평균 건수 확인")
    print("       < 5건/일 → 후보1_OR_전체  |  > 50건/일 → 후보3_OR_정밀")
    print("  [AND] 후보4(정책×에듀테크) / 후보5(경쟁사×에듀테크) → 목적별 선택")
    print("  2주 운영 후 Phase 5 메트릭 평가")


# ============================================================
# 공개 인터페이스
# ============================================================

def run_phase6(r_a: dict, r_b: dict, r_c: dict, r_d: dict,
               logger: ProcessLogger) -> dict:
    """
    Phase 6 실행 진입점.

    Parameters
    ----------
    r_a, r_b, r_c, r_d : dict   각 method_x() 반환값 (A/B/C/D 모두 필요)
    logger              : ProcessLogger   호출자가 생성한 로거 (이어서 기록)

    Returns
    -------
    validated : dict  {전략명: {query, status, cats_used, ...}}
    """
    coordinator = QueryCoordinator()
    return coordinator.run(r_a, r_b, r_c, r_d, logger)
