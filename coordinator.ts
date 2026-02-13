import Anthropic from "@anthropic-ai/sdk";
import { analyzerAgent } from "./agents/analyzerAgent.js";
import { strategistAgent } from "./agents/strategistAgent.js";
import { queryBuilderAgent } from "./agents/queryBuilderAgent.js";
import { validatorAgent } from "./agents/validatorAgent.js";
import { reporterAgent } from "./agents/reporterAgent.js";
import type { PipelineContext, OptimizationReport } from "./types.js";

export class Coordinator {
  private client: Anthropic;
  private context: PipelineContext;
  private totalTokens = { input: 0, output: 0 };

  constructor(rawInput: string) {
    this.client = new Anthropic();
    const rawKeywords = rawInput
      .split(/\s+OR\s+/)
      .map((k) => k.trim())
      .filter((k) => k.length > 0);

    this.context = { rawKeywords, rawInput };
  }

  private log(step: string, message: string): void {
    const timestamp = new Date().toISOString().slice(11, 19);
    console.log(`[${timestamp}] [${step}] ${message}`);
  }

  private trackTokens(usage?: { input: number; output: number }): void {
    if (usage) {
      this.totalTokens.input += usage.input;
      this.totalTokens.output += usage.output;
    }
  }

  async run(): Promise<OptimizationReport> {
    console.log("=".repeat(60));
    console.log("  키워드 최적화 에이전트 파이프라인 시작");
    console.log("=".repeat(60));
    this.log("INIT", `입력 키워드: ${this.context.rawKeywords.length}개`);

    // Step 1: Analyzer
    this.log("STEP 1/5", "Analyzer 에이전트 실행 중...");
    const analysisResult = await analyzerAgent(
      this.client,
      this.context.rawKeywords,
    );
    this.trackTokens(analysisResult.tokenUsage);

    if (!analysisResult.success) {
      throw new Error(`Analyzer 실패: ${analysisResult.error}`);
    }
    this.context.analysis = analysisResult.data;
    this.log(
      "STEP 1/5",
      `완료 - ${analysisResult.data.categories.length}개 카테고리 분류`,
    );

    // Step 2: Strategist
    this.log("STEP 2/5", "Strategist 에이전트 실행 중...");
    const strategyResult = await strategistAgent(
      this.client,
      this.context.analysis,
    );
    this.trackTokens(strategyResult.tokenUsage);

    if (!strategyResult.success) {
      throw new Error(`Strategist 실패: ${strategyResult.error}`);
    }
    this.context.strategy = strategyResult.data;
    this.log(
      "STEP 2/5",
      `완료 - ${strategyResult.data.strategies.length}개 전략 수립`,
    );

    // Step 3: QueryBuilder
    this.log("STEP 3/5", "QueryBuilder 에이전트 실행 중...");
    const queryResult = await queryBuilderAgent(
      this.client,
      this.context.analysis,
      this.context.strategy,
    );
    this.trackTokens(queryResult.tokenUsage);

    if (!queryResult.success) {
      throw new Error(`QueryBuilder 실패: ${queryResult.error}`);
    }
    this.context.queries = queryResult.data;
    this.log(
      "STEP 3/5",
      `완료 - ${queryResult.data.length}개 쿼리 생성`,
    );

    // Step 4: Validator
    this.log("STEP 4/5", "Validator 에이전트 실행 중...");
    const validationResult = await validatorAgent(
      this.client,
      this.context.queries,
      this.context.rawKeywords,
    );
    this.trackTokens(validationResult.tokenUsage);

    if (!validationResult.success) {
      throw new Error(`Validator 실패: ${validationResult.error}`);
    }
    this.context.validations = validationResult.data;

    const passCount = validationResult.data.filter((v) => v.isValid).length;
    this.log(
      "STEP 4/5",
      `완료 - ${passCount}/${validationResult.data.length}개 쿼리 검증 통과`,
    );

    // Step 5: Reporter
    this.log("STEP 5/5", "Reporter 에이전트 실행 중...");
    const reportResult = await reporterAgent(
      this.client,
      this.context.analysis,
      this.context.queries,
      this.context.validations,
    );
    this.trackTokens(reportResult.tokenUsage);

    if (!reportResult.success) {
      throw new Error(`Reporter 실패: ${reportResult.error}`);
    }
    this.context.report = reportResult.data;
    this.log("STEP 5/5", "완료 - 최종 리포트 생성");

    // Summary
    console.log("\n" + "=".repeat(60));
    console.log("  파이프라인 완료");
    console.log("=".repeat(60));
    console.log(
      `  총 토큰 사용: 입력 ${this.totalTokens.input.toLocaleString()} / 출력 ${this.totalTokens.output.toLocaleString()}`,
    );
    console.log("=".repeat(60));

    return this.context.report;
  }
}
