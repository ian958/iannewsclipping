"""
Phase 6: 검색식 생성 - Subagent 기반 AND/OR 쿼리 빌더
================================================================
입력  : 클러스터링 결과 (Method D 의미 카테고리 우선)
출력  : 최종 검색식 후보 3개  (AND/OR 조합)
기록  : ProcessLogger → JSON + TXT 파일 자동 저장

서브에이전트 구성
  QueryBuilderAgent   : 그룹별 OR 검색식 생성 + 키워드 길이 필터
  StrategyAgent       : 3가지 AND/OR 전략 결정
  QueryFormatterAgent : 검색식 문자열 포맷팅
  ValidatorAgent      : 복잡도·유효성 검증
  QueryCoordinator    : 4개 에이전트 오케스트레이션
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

    VERSION = "1.0.0"

    def __init__(self, session_name: str = "keyword_search_design"):
        self.session_name = session_name
        self.started_at = datetime.now().isoformat()
        self.events: list[dict] = []      # 에이전트 실행 이벤트
        self.decisions: list[dict] = []   # 의사결정 로그

    # ── 기록 메서드 ──────────────────────────────────────────

    def record(self, agent: str, action: str,
               inputs: dict = None, outputs: dict = None,
               elapsed: float = None):
        """에이전트 실행 이벤트 기록"""
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
        """의사결정 기록 (어떤 파라미터를 왜 선택했는가)"""
        self.decisions.append({
            "ts": datetime.now().isoformat(),
            "agent": agent,
            "parameter": parameter,
            "value": value,
            "reasoning": reasoning,
        })

    # ── 저장 ────────────────────────────────────────────────

    def save(self, out_dir: str = "report") -> tuple[str, str]:
        """JSON + TXT 두 가지 형식으로 저장"""
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        Path(out_dir).mkdir(parents=True, exist_ok=True)

        # ── JSON (전체 로그) ──────────────────────────
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

        # ── TXT (사람이 읽기 쉬운 요약) ──────────────
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
        lines += [
            "",
            "[ 의사결정 기록 ]",
        ]
        for d in self.decisions:
            lines.append(f"  [{d['agent']}]  {d['parameter']} = {d['value']}")
            lines.append(f"    근거: {d['reasoning']}")
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

        print(f"\n  [ProcessLogger] JSON 저장 → {json_path}")
        print(f"  [ProcessLogger] TXT  저장 → {txt_path}")
        return json_path, txt_path


# ============================================================
# 서브에이전트 1: QueryBuilderAgent
# ============================================================

class QueryBuilderAgent:
    """
    역할  : 카테고리별 키워드를 OR 묶음으로 변환
    정책  : 단어 수 3개 이하 키워드만 포함 (긴 기관명은 검색 잡음 증가)
    출력  : {카테고리명: [키워드, ...]} OR 그룹 딕셔너리
    """

    MAX_WORDS = 3   # 키워드 내 공백 기준 단어 수 상한

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
                "빅카인즈 검색식에 '국립목포해양대학교' 같은 긴 기관명을 포함하면 "
                "다른 표기(약칭·한자병기)로 작성된 기사가 누락됨. "
                "핵심 개념어 위주로 OR 묶음을 구성해 재현율 유지."
            ),
        )
        return or_groups


# ============================================================
# 서브에이전트 2: StrategyAgent
# ============================================================

class StrategyAgent:
    """
    역할  : 3가지 AND/OR 조합 전략 정의
    출력  : strategy_name → {and_cats, or_only, description, reasoning}
    """

    # 전략 정의 (AND에 포함할 카테고리 목록)
    STRATEGIES = {
        "후보1_광범위 (Recall 최대화)": {
            "and_cats": [],          # AND 없음 → 전체 OR
            "or_extra_cats": [       # OR로 포함할 카테고리
                "AI·챗봇 기술",
                "학습 플랫폼·에듀테크",
                "정부 정책·사업비",
                "대학 인사·행정",
                "에듀테크 경쟁사",
            ],
            "description": "핵심 AI/에듀테크/정책 키워드 전체를 OR로 묶어 최대 수집",
            "reasoning": (
                "누락(FN) 최소화 목적. 광범위하게 수집 후 "
                "Phase 4 스코어링에서 관련도 필터링 예정. "
                "일일 뉴스 건수 확인 및 임계값 설정에 적합."
            ),
        },
        "후보2_균형 (Precision-Recall 균형)": {
            "and_cats": ["AI·챗봇 기술", "정부 정책·사업비"],
            "or_extra_cats": [],
            "description": "AI/챗봇 기술 AND 정부 정책·사업비 교집합",
            "reasoning": (
                "회사 제품(AI 챗봇)과 직접 연관된 정책 뉴스에 집중. "
                "AND 2개로 잡음을 줄이면서 핵심 비즈니스 컨텍스트 유지. "
                "일반적인 일일 모니터링에 권장."
            ),
        },
        "후보3_정밀 (Precision 최대화)": {
            "and_cats": ["AI·챗봇 기술", "학습 플랫폼·에듀테크", "정부 정책·사업비"],
            "or_extra_cats": [],
            "description": "AI챗봇 AND 에듀테크플랫폼 AND 정책 3중 교집합",
            "reasoning": (
                "노이즈 최소화. 세 조건을 모두 만족하는 고관련도 기사만 수집. "
                "결과 건수가 적으므로 주간·월간 심층 분석에 적합. "
                "AND 3개 이상이면 빅카인즈 검색식이 복잡해지므로 최대 권장선."
            ),
        },
    }

    def run(self, or_groups: dict, logger: ProcessLogger) -> dict:
        t0 = time.time()
        strategies: dict[str, dict] = {}

        for name, cfg in self.STRATEGIES.items():
            strategies[name] = {
                "and_cats": cfg["and_cats"],
                "or_extra_cats": cfg.get("or_extra_cats", []),
                "description": cfg["description"],
                "reasoning": cfg["reasoning"],
                # 실제 키워드 그룹 참조
                "and_groups": {
                    c: or_groups[c] for c in cfg["and_cats"] if c in or_groups
                },
                "or_groups": {
                    c: or_groups[c]
                    for c in cfg.get("or_extra_cats", [])
                    if c in or_groups
                },
            }
            logger.record_decision(
                agent="StrategyAgent",
                parameter=name,
                value=f"AND={cfg['and_cats']}",
                reasoning=cfg["reasoning"],
            )

        elapsed = time.time() - t0
        logger.record(
            agent="StrategyAgent",
            action="3가지 AND/OR 전략 결정",
            inputs={"n_or_groups": len(or_groups)},
            outputs={"n_strategies": len(strategies)},
            elapsed=elapsed,
        )
        return strategies


# ============================================================
# 서브에이전트 3: QueryFormatterAgent
# ============================================================

class QueryFormatterAgent:
    """
    역할  : 전략 딕셔너리를 실제 검색식 문자열로 포맷팅
    출력  : strategy_name → {query, stats, description, reasoning}
    """

    def run(self, strategies: dict, logger: ProcessLogger) -> dict:
        t0 = time.time()
        formatted: dict[str, dict] = {}

        for name, strat in strategies.items():
            and_groups = strat["and_groups"]
            or_groups  = strat["or_groups"]

            if and_groups:
                # AND 연결: 각 그룹을 (k1 OR k2 ...) 으로 래핑 후 AND
                and_parts = [
                    "(" + " OR ".join(kws) + ")"
                    for kws in and_groups.values() if kws
                ]
                query = " AND ".join(and_parts)
            else:
                # 전체 OR (광범위 전략)
                all_kws: list[str] = []
                for kws in or_groups.values():
                    all_kws.extend(kws)
                query = " OR ".join(all_kws)

            # 통계
            kw_count = query.count(" OR ") + query.count(" AND ") + 1
            and_count = query.count(" AND ")
            or_count  = query.count(" OR ")

            formatted[name] = {
                "query": query,
                "kw_count": kw_count,
                "and_count": and_count,
                "or_count": or_count,
                "description": strat["description"],
                "reasoning": strat["reasoning"],
            }

        elapsed = time.time() - t0
        logger.record(
            agent="QueryFormatterAgent",
            action="검색식 문자열 포맷팅",
            inputs={"n_strategies": len(strategies)},
            outputs={"n_queries": len(formatted)},
            elapsed=elapsed,
        )
        return formatted


# ============================================================
# 서브에이전트 4: ValidatorAgent
# ============================================================

class ValidatorAgent:
    """
    역할  : 검색식 유효성·복잡도 검증
    기준  : 키워드 수 ≤ 50, AND ≤ 3 (빅카인즈 실용 권장)
    """

    MAX_KW   = 50
    MAX_AND  = 3

    def run(self, queries: dict, logger: ProcessLogger) -> dict:
        validated: dict[str, dict] = {}

        for name, q in queries.items():
            issues   = []
            warnings = []

            if q["kw_count"] > self.MAX_KW:
                issues.append(f"키워드 {q['kw_count']}개 > 권장 {self.MAX_KW}개")
            if q["and_count"] > self.MAX_AND:
                warnings.append(f"AND {q['and_count']}개: 결과 과소 위험")
            if q["and_count"] == 0:
                warnings.append("AND 없음: 결과 과다·스코어링 부하 주의")

            status = "PASS" if not issues else "FAIL"
            validated[name] = {**q, "status": status,
                                "issues": issues, "warnings": warnings}

            logger.record_decision(
                agent="ValidatorAgent",
                parameter=f"검증: {name}",
                value=status,
                reasoning=(
                    f"kw={q['kw_count']}, AND={q['and_count']}, OR={q['or_count']}. "
                    + ("; ".join(issues + warnings) or "이상 없음")
                ),
            )

        logger.record(
            agent="ValidatorAgent",
            action="검색식 유효성 검증",
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
    4개 서브에이전트를 순서대로 실행하고 결과를 취합.
    각 단계 결과는 ProcessLogger에 자동 기록됨.
    """

    def __init__(self):
        self.builder   = QueryBuilderAgent()
        self.strategist = StrategyAgent()
        self.formatter = QueryFormatterAgent()
        self.validator = ValidatorAgent()

    def run(self, assigned: dict, logger: ProcessLogger) -> dict:
        logger.record("QueryCoordinator", "Phase 6 파이프라인 시작",
                      inputs={"n_categories": len(assigned)})

        print("\n  [Agent 1] QueryBuilderAgent: OR 묶음 생성 중...")
        or_groups = self.builder.run(assigned, logger)
        print(f"    → {len(or_groups)}개 OR 그룹 생성")

        print("  [Agent 2] StrategyAgent: AND/OR 전략 3개 결정 중...")
        strategies = self.strategist.run(or_groups, logger)
        print(f"    → {len(strategies)}개 전략 확정")

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
    """검색식 후보 3개를 사람이 읽기 쉬운 TXT 파일로 저장"""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    path = f"{out_dir}/phase6_query_candidates_{ts}.txt"

    lines = [
        "=" * 70,
        "  Phase 6: 최종 검색식 후보 3개",
        f"  생성일시: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "=" * 70,
        "",
    ]
    for idx, (name, q) in enumerate(validated.items(), 1):
        lines += [
            f"【{name}】",
            f"  설명  : {q['description']}",
            f"  상태  : {q['status']}"
            + (f"  경고: {'; '.join(q['warnings'])}" if q["warnings"] else ""),
            f"  통계  : 키워드 {q['kw_count']}개 | AND {q['and_count']}개 | OR {q['or_count']}개",
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
        "  1단계: 후보2(균형) 으로 1주일 수집 → 일평균 건수 확인",
        "  2단계: 건수 부족(< 5건/일) → 후보1(광범위) 전환",
        "         건수 과다(> 50건/일) → 후보3(정밀) 전환",
        "  3단계: 2주 운영 후 Phase 5 메트릭(Precision/Recall) 평가",
    ]

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"\n  [Phase 6] 검색식 후보 저장 → {path}")
    return path


def print_query_candidates(validated: dict):
    """콘솔 출력"""
    print("\n" + "=" * 70)
    print("  Phase 6 결과: 최종 검색식 후보 3개")
    print("=" * 70)

    for name, q in validated.items():
        status_mark = "✓" if q["status"] == "PASS" else "✗"
        warn_str = f"  ⚠ {'; '.join(q['warnings'])}" if q["warnings"] else ""
        print(f"\n【{name}】  {status_mark}{warn_str}")
        print(f"  설명  : {q['description']}")
        print(f"  통계  : AND={q['and_count']}개 / OR={q['or_count']}개 / 총={q['kw_count']}개")
        print(f"  검색식: {q['query']}")
        print(f"  근거  : {q['reasoning'][:80]}...")

    print("\n" + "=" * 70)
    print("  빅카인즈 사용 권장 순서")
    print("=" * 70)
    print("  1단계: 후보2(균형) → 1주일 수집, 일평균 건수 확인")
    print("  2단계: < 5건/일 → 후보1(광범위)  |  > 50건/일 → 후보3(정밀)")
    print("  3단계: 2주 운영 후 Phase 5 메트릭 평가")


# ============================================================
# 공개 인터페이스
# ============================================================

def run_phase6(d_result: dict, logger: ProcessLogger) -> dict:
    """
    Phase 6 실행 진입점.

    Parameters
    ----------
    d_result : dict   test_run.method_d() 반환값
    logger   : ProcessLogger   호출자가 생성한 로거 (이어서 기록)

    Returns
    -------
    validated : dict  {전략명: {query, status, ...}}
    """
    assigned = d_result.get("assigned", {})
    if not assigned:
        raise ValueError("d_result에 'assigned' 키가 없습니다. method_d() 반환값을 전달하세요.")

    coordinator = QueryCoordinator()
    validated   = coordinator.run(assigned, logger)
    return validated


# ============================================================
# 단독 실행 (데모)
# ============================================================

if __name__ == "__main__":
    # ── Method D 카테고리 (test_run.LLM_CATEGORIES 기반) ──
    DEMO_ASSIGNED = {
        "AI·챗봇 기술": ["챗봇", "AI 챗봇", "AI 상담", "학습 AI", "AI 어드바이저", "AI 학사"],
        "학습 플랫폼·에듀테크": ["LMS", "학습관리시스템", "LXP", "에듀테크",
                             "건양대 에듀테크소프트랩", "서울신학대 에듀테크소프트랩",
                             "한림대학교 AI에듀테크 센터"],
        "정부 정책·사업비": ["교육부", "교육부 인사", "대학혁신지원사업", "글로컬",
                         "글로컬 예산", "RISE", "RISE 예산", "지원금", "사업비",
                         "대학 예산", "대학 확보", "대학 지정", "대학 선정",
                         "지원 사업", "LINC3.0사업단", "교원양성기관", "교육 역량 강화"],
        "대학 인사·행정": ["대학교 선출", "대학교 선임", "대학교 인사", "무전공", "자율전공"],
        "CTL·교수학습센터": ["CTL", "KACTL", "카이스트 교수학습센터",
                          "계명대학교 교수학습개발센터", "인하대학교 교수학습개발센터",
                          "국민대학교 교수학습혁신센터", "건양대학교 CTL",
                          "대구한의대학교 CTL", "배재대학교 CTL",
                          "순천향대학교 교육혁신원", "서울신학대 교육혁신원",
                          "가톨릭관동대학교 교육혁신센터", "대구한의대학교 K-MEDI디지털센터",
                          "서울여자대학교 디지털혁신실", "한림대 커뮤니티교육원"],
        "에듀테크 경쟁사": ["유비온", "자이닉스", "메디오피아", "프리윌린", "엘리스그룹",
                        "메이크봇", "로이드케이", "아이맥스소프트", "와이즈넛", "마인드로직"],
        "파트너·고객 대학교": ["국립목포해양대학교", "한양대학교 캠퍼스타운", "연세대학교 리더십센터",
                           "신경주대학교", "제주국제대학교", "대구대학교", "울산대학교",
                           "배화여자대학교", "상지대학교", "광주대학교", "경남도립거창대학교",
                           "남부대학교", "동강대학교", "가톨릭상지대학교",
                           "서울신학대학교 국제학부", "서울대학교 글로벌공학교육센터",
                           "한국대학교육협의회", "대림대학교", "신성대학교", "숭의여자대학교",
                           "세한대학교", "동신대학교", "연암공대"],
    }

    logger = ProcessLogger("phase6_standalone_demo")
    logger.record("Main", "Phase 6 단독 실행 시작",
                  inputs={"n_categories": len(DEMO_ASSIGNED)})

    print("Phase 6: 검색식 생성 파이프라인 시작\n")
    coordinator = QueryCoordinator()
    validated   = coordinator.run(DEMO_ASSIGNED, logger)

    print_query_candidates(validated)
    save_query_candidates(validated)
    logger.save()
