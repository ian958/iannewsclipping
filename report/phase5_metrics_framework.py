"""
Phase 5: 메트릭 측정 프레임워크
AI 에듀테크 뉴스클리핑 자동화 시스템

ProcessMetrics 클래스: 각 Phase의 처리 시간, 비용, 품질, Subagent 결정사항 기록
"""

import json
import csv
from datetime import datetime


class ProcessMetrics:
    """각 Phase의 메트릭을 측정하는 클래스"""

    def __init__(self, process_name):
        self.process_name = process_name
        self.created_at = datetime.now().isoformat()

        # 시간 메트릭 (분 단위)
        self.human_time = None      # 사람 작업 시간 (직접 측정)
        self.auto_time = None       # Subagent 실행 시간 (직접 측정)

        # 비용 메트릭
        self.hourly_rate = 30000    # 시급 (원) - 인건비 계산 기준
        self.api_cost = 0           # API 비용 (원, $0 = 0원)

        # 처리 건수
        self.input_count = None     # 입력 데이터 건수
        self.output_count = None    # 출력 데이터 건수

        # 품질 메트릭
        self.accuracy = None        # 정확도 (샘플 평가 필요, %)

        # Subagent 결정 파라미터
        self.subagent_decisions = {}

        # 세부 시간 기록 (분 단위)
        self.time_breakdown = {}

    # ----------------------------------------------------------
    # 기본 계산
    # ----------------------------------------------------------

    def calculate_total_time(self):
        """총 소요 시간 계산 (분)"""
        if self.human_time is None or self.auto_time is None:
            return None
        return self.human_time + self.auto_time

    def calculate_cost(self):
        """총 비용 계산 (원)"""
        total_time = self.calculate_total_time()
        if total_time is None:
            return None
        human_cost = (self.human_time / 60) * self.hourly_rate
        return human_cost + self.api_cost

    def calculate_automation_rate(self):
        """자동화율 계산 (%)"""
        total = self.calculate_total_time()
        if total is None or total == 0:
            return None
        return (self.auto_time / total) * 100

    def calculate_compression_rate(self):
        """데이터 압축률 계산 (%)"""
        if self.input_count is None or self.output_count is None:
            return None
        if self.input_count == 0:
            return None
        return ((self.input_count - self.output_count) / self.input_count) * 100

    def calculate_throughput(self):
        """시간당 처리 건수 (건/시간)"""
        total = self.calculate_total_time()
        if total is None or total == 0 or self.output_count is None:
            return None
        return (self.output_count / total) * 60  # 분 → 시간 환산

    # ----------------------------------------------------------
    # 데이터 기록
    # ----------------------------------------------------------

    def add_subagent_decision(self, parameter, value, reasoning):
        """
        Subagent가 결정한 파라미터 기록

        Parameters
        ----------
        parameter : str   파라미터 이름 (예: "유사도 임계값")
        value     : any   Subagent가 결정한 값 (None이면 측정 후 기입)
        reasoning : str   Subagent가 제공한 결정 근거
        """
        self.subagent_decisions[parameter] = {
            "value": value,
            "reasoning": reasoning,
        }

    def add_time_breakdown(self, step_name, minutes):
        """
        세부 작업별 시간 기록

        Parameters
        ----------
        step_name : str    단계 이름 (예: "파일 업로드")
        minutes   : float  소요 시간 (분)
        """
        self.time_breakdown[step_name] = minutes

    def set_accuracy(self, correct_count, sample_size):
        """
        샘플 기반 정확도 계산

        Parameters
        ----------
        correct_count : int  정확한 결과 수
        sample_size   : int  전체 샘플 수
        """
        if sample_size > 0:
            self.accuracy = (correct_count / sample_size) * 100

    # ----------------------------------------------------------
    # 요약 출력
    # ----------------------------------------------------------

    def summary(self):
        """메트릭 전체 요약 딕셔너리"""
        total_time = self.calculate_total_time()
        return {
            "process": self.process_name,
            "created_at": self.created_at,
            "time": {
                "human_min": self.human_time,
                "auto_min": self.auto_time,
                "total_min": total_time,
                "breakdown": self.time_breakdown,
            },
            "cost": {
                "human_cost_krw": (self.human_time / 60 * self.hourly_rate)
                    if self.human_time is not None else None,
                "api_cost_krw": self.api_cost,
                "total_cost_krw": self.calculate_cost(),
                "hourly_rate_krw": self.hourly_rate,
            },
            "efficiency": {
                "automation_rate_pct": self.calculate_automation_rate(),
                "throughput_per_hour": self.calculate_throughput(),
            },
            "throughput": {
                "input": self.input_count,
                "output": self.output_count,
                "compression_rate_pct": self.calculate_compression_rate(),
            },
            "quality": {
                "accuracy_pct": self.accuracy,
            },
            "subagent_decisions": self.subagent_decisions,
        }

    def print_summary(self):
        """콘솔 출력용 포맷팅"""
        s = self.summary()
        print(f"\n{'='*50}")
        print(f"  [{s['process']}] 메트릭 요약")
        print(f"{'='*50}")

        t = s["time"]
        print(f"\n[시간]")
        print(f"  사람 작업: {t['human_min']}분")
        print(f"  Subagent:  {t['auto_min']}분")
        print(f"  합계:      {t['total_min']}분")
        if t["breakdown"]:
            print("  세부:")
            for step, mins in t["breakdown"].items():
                print(f"    - {step}: {mins}분")

        c = s["cost"]
        print(f"\n[비용]")
        print(f"  인건비:    {c['human_cost_krw']}원")
        print(f"  API 비용:  {c['api_cost_krw']}원 (${c['api_cost_krw']/1300:.2f})")
        print(f"  합계:      {c['total_cost_krw']}원")

        e = s["efficiency"]
        print(f"\n[효율성]")
        print(f"  자동화율:  {e['automation_rate_pct']}%")
        print(f"  처리량:    {e['throughput_per_hour']}건/시간")

        tp = s["throughput"]
        print(f"\n[처리량]")
        print(f"  입력:      {tp['input']}건")
        print(f"  출력:      {tp['output']}건")
        print(f"  압축률:    {tp['compression_rate_pct']}%")

        q = s["quality"]
        print(f"\n[품질]")
        print(f"  정확도:    {q['accuracy_pct']}%")

        if s["subagent_decisions"]:
            print(f"\n[Subagent 결정사항]")
            for param, info in s["subagent_decisions"].items():
                print(f"  {param}: {info['value']}")
                if info["reasoning"]:
                    print(f"    └ 근거: {info['reasoning']}")

    def to_json(self, filepath=None):
        """JSON 파일로 저장"""
        data = self.summary()
        json_str = json.dumps(data, ensure_ascii=False, indent=2)
        if filepath:
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(json_str)
            print(f"  메트릭 저장: {filepath}")
        return json_str

    @staticmethod
    def compare_phases(metrics_list):
        """
        여러 Phase의 메트릭을 비교 요약합니다.

        Parameters
        ----------
        metrics_list : list[ProcessMetrics]

        Returns
        -------
        comparison : dict
        """
        totals = {
            "total_human_time": 0,
            "total_auto_time": 0,
            "total_cost": 0,
            "phases": [],
        }

        for m in metrics_list:
            s = m.summary()
            totals["total_human_time"] += (s["time"]["human_min"] or 0)
            totals["total_auto_time"] += (s["time"]["auto_min"] or 0)
            totals["total_cost"] += (s["cost"]["total_cost_krw"] or 0)
            totals["phases"].append({
                "name": s["process"],
                "total_min": s["time"]["total_min"],
                "compression_rate": s["throughput"]["compression_rate_pct"],
                "automation_rate": s["efficiency"]["automation_rate_pct"],
                "subagent_decisions": list(s["subagent_decisions"].keys()),
            })

        total_time = totals["total_human_time"] + totals["total_auto_time"]
        totals["overall_automation_rate"] = (
            (totals["total_auto_time"] / total_time * 100) if total_time > 0 else None
        )

        return totals

    @staticmethod
    def save_comparison_csv(metrics_list, filepath="report/pipeline_metrics.csv"):
        """Phase 비교 CSV 저장"""
        fieldnames = [
            "phase", "human_min", "auto_min", "total_min",
            "total_cost_krw", "automation_rate_pct",
            "input_count", "output_count", "compression_rate_pct",
            "accuracy_pct", "subagent_decisions",
        ]

        with open(filepath, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for m in metrics_list:
                s = m.summary()
                writer.writerow({
                    "phase": s["process"],
                    "human_min": s["time"]["human_min"],
                    "auto_min": s["time"]["auto_min"],
                    "total_min": s["time"]["total_min"],
                    "total_cost_krw": s["cost"]["total_cost_krw"],
                    "automation_rate_pct": s["efficiency"]["automation_rate_pct"],
                    "input_count": s["throughput"]["input"],
                    "output_count": s["throughput"]["output"],
                    "compression_rate_pct": s["throughput"]["compression_rate_pct"],
                    "accuracy_pct": s["quality"]["accuracy_pct"],
                    "subagent_decisions": "|".join(s["subagent_decisions"].keys()),
                })

        print(f"  비교 CSV 저장: {filepath}")


# ============================================================
# 사용 예시 (실제 측정 후 기입)
# ============================================================

if __name__ == "__main__":
    # Phase 1: 검색어 설계
    phase1 = ProcessMetrics("1. 검색어 설계")
    phase1.human_time = None        # 측정 후 기입 (분)
    phase1.auto_time = 0            # Phase 1은 자동화 없음
    phase1.input_count = None       # seed 키워드 수
    phase1.output_count = None      # 최종 키워드 수
    phase1.api_cost = 0
    # Phase 1은 Subagent 결정사항 없음

    # Phase 2: 데이터 수집
    phase2 = ProcessMetrics("2. 데이터 수집")
    phase2.human_time = None        # 빅카인즈 검색 + 다운로드 + 전처리
    phase2.auto_time = 0            # 수동 작업
    phase2.input_count = None       # 검색 결과 건수
    phase2.output_count = None      # 전처리 후 건수
    phase2.api_cost = 0

    # Phase 3: 클러스터링
    phase3 = ProcessMetrics("3. 클러스터링")
    phase3.human_time = None        # CSV 업로드 + 프롬프트 + 검토
    phase3.auto_time = None         # Subagent 실행 시간
    phase3.input_count = None       # 전처리된 뉴스 건수
    phase3.output_count = None      # 클러스터 수 (대표 기사)
    phase3.api_cost = 0

    phase3.add_time_breakdown("CSV 업로드", None)
    phase3.add_time_breakdown("프롬프트 작성", None)
    phase3.add_time_breakdown("결과 검토", None)

    phase3.add_subagent_decision(
        parameter="유사도 임계값",
        value=None,           # Subagent가 제안한 값 기입
        reasoning=None,       # Subagent가 제공한 이유 기입
    )
    phase3.add_subagent_decision(
        parameter="클러스터 개수",
        value=None,
        reasoning=None,
    )
    phase3.add_subagent_decision(
        parameter="소규모 클러스터 처리",
        value=None,
        reasoning=None,
    )

    # Phase 4: 스코어링
    phase4 = ProcessMetrics("4. 스코어링")
    phase4.human_time = None
    phase4.auto_time = None
    phase4.input_count = None       # 클러스터링 후 건수
    phase4.output_count = None      # 임계점 적용 후 건수
    phase4.api_cost = 0

    phase4.add_time_breakdown("파일 업로드", None)
    phase4.add_time_breakdown("프롬프트 작성", None)
    phase4.add_time_breakdown("임계점 검토", None)
    phase4.add_time_breakdown("필터링 실행", None)

    phase4.add_subagent_decision(
        parameter="필터링 임계점",
        value=None,           # 예: 7
        reasoning=None,
    )
    phase4.add_subagent_decision(
        parameter="예상 출력 건수",
        value=None,
        reasoning=None,
    )

    # Phase 5: 요약/분류
    phase5 = ProcessMetrics("5. 요약/분류")
    phase5.human_time = None
    phase5.auto_time = None
    phase5.input_count = None
    phase5.output_count = None
    phase5.api_cost = 0

    phase5.add_subagent_decision(
        parameter="요약 길이",
        value=None,           # 예: "3문장"
        reasoning=None,
    )
    phase5.add_subagent_decision(
        parameter="카테고리 분류 기준",
        value=None,
        reasoning=None,
    )

    # 전체 Phase 요약 출력
    all_phases = [phase1, phase2, phase3, phase4, phase5]
    for p in all_phases:
        p.print_summary()

    # 비교 CSV 저장
    ProcessMetrics.save_comparison_csv(all_phases)

    # 전체 비교
    comparison = ProcessMetrics.compare_phases(all_phases)
    print("\n[전체 파이프라인 합계]")
    print(f"  총 사람 작업: {comparison['total_human_time']}분")
    print(f"  총 Subagent:  {comparison['total_auto_time']}분")
    print(f"  총 비용:      {comparison['total_cost']}원")
    print(f"  전체 자동화율: {comparison['overall_automation_rate']}%")
