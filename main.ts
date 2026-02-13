import { Coordinator } from "./coordinator.js";

const DEFAULT_KEYWORDS = [
  "LMS",
  "챗봇",
  "한양대",
  "이러닝",
  "e-러닝",
  "온라인 교육",
  "디지털 교육",
  "에듀테크",
  "EdTech",
  "학습관리시스템",
  "AI 교육",
  "인공지능 교육",
  "스마트 교육",
  "원격교육",
  "비대면 수업",
  "블렌디드 러닝",
  "하이브리드 수업",
  "MOOC",
  "K-MOOC",
  "OCW",
  "마이크로러닝",
  "적응형 학습",
  "개인화 학습",
  "학습 분석",
  "러닝 애널리틱스",
  "교육 데이터",
  "교육 혁신",
  "대학 혁신",
  "교육 DX",
  "디지털 전환 교육",
  "교수학습",
  "교수법",
  "플립러닝",
  "거꾸로 학습",
  "PBL",
  "프로젝트 기반 학습",
  "역량 기반 교육",
  "CBE",
  "마이크로크리덴셜",
  "디지털 배지",
  "학점은행",
  "평생교육",
  "성인학습",
  "기업교육",
  "HRD",
  "직무교육",
  "리스킬링",
  "업스킬링",
  "GPT 교육",
  "생성형 AI 교육",
  "ChatGPT 교육",
  "AI 튜터",
  "지능형 튜터링",
  "ITS",
  "교육용 AI",
  "AI 리터러시",
  "디지털 리터러시",
  "메타버스 교육",
  "VR 교육",
  "AR 교육",
  "XR 교육",
  "실감형 콘텐츠",
  "교육 콘텐츠",
  "이러닝 콘텐츠",
  "디지털 교과서",
  "전자교과서",
  "OER",
  "공개교육자원",
  "교육 플랫폼",
  "학습 플랫폼",
  "클라우드 교육",
  "SaaS 교육",
  "교육부",
  "고등교육",
  "대학교육",
  "대학 온라인",
  "사이버대학",
  "방송통신대학",
  "학사 관리",
  "수강 신청",
  "교육과정 혁신",
].join(" OR ");

async function main(): Promise<void> {
  const input = process.argv[2] || DEFAULT_KEYWORDS;

  console.log("키워드 최적화 서브에이전트 시스템");
  console.log(`입력: ${input.slice(0, 100)}...`);
  console.log();

  const coordinator = new Coordinator(input);

  try {
    const report = await coordinator.run();

    console.log("\n");
    console.log("=".repeat(60));
    console.log("  최종 최적화 리포트");
    console.log("=".repeat(60));

    console.log(`\n## 요약\n${report.summary}\n`);
    console.log(
      `입력 키워드: ${report.inputKeywordCount}개 → 출력 쿼리: ${report.outputQueryCount}개\n`,
    );

    console.log("-".repeat(60));
    for (const q of report.queries) {
      const statusIcon =
        q.validationStatus === "pass"
          ? "[PASS]"
          : q.validationStatus === "warning"
            ? "[WARN]"
            : "[FAIL]";

      console.log(`\n### ${statusIcon} 쿼리 ${q.id}: ${q.title}`);
      console.log(`전략: ${q.strategy}`);
      console.log(`쿼리:\n  ${q.query}`);
      if (q.notes) {
        console.log(`비고: ${q.notes}`);
      }
    }

    console.log("\n" + "-".repeat(60));
    console.log("\n## 권장사항");
    for (const rec of report.recommendations) {
      console.log(`  - ${rec}`);
    }

    console.log(`\n## 커버리지 분석\n${report.coverageAnalysis}`);
  } catch (error) {
    console.error(
      "파이프라인 실행 중 오류 발생:",
      error instanceof Error ? error.message : error,
    );
    process.exit(1);
  }
}

main();
