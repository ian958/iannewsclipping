import Anthropic from "@anthropic-ai/sdk";
import type {
  AgentResult,
  BigKindsQuery,
  OptimizationReport,
  ValidationResult,
  KeywordAnalysis,
} from "../types.js";

const SYSTEM_PROMPT = `당신은 키워드 최적화 결과 리포팅 전문가입니다.
검색 쿼리 생성 및 검증 결과를 종합하여 최종 리포트를 작성합니다.

리포트에 포함할 내용:
1. 전체 요약 (summary)
2. 각 쿼리의 최종 상태 (validationStatus: pass/warning/fail)
3. 종합 권장사항 (recommendations)
4. 커버리지 분석 (coverageAnalysis)

검증에서 문제가 발견된 쿼리는 warning 또는 fail로 표시하고,
해당 쿼리의 notes에 구체적인 문제점과 개선 방향을 기술합니다.

반드시 아래 JSON 형식으로만 응답하세요:
{
  "summary": "전체 최적화 과정 요약",
  "inputKeywordCount": 숫자,
  "outputQueryCount": 숫자,
  "queries": [
    {
      "id": 1,
      "title": "쿼리 제목",
      "query": "최종 쿼리",
      "strategy": "사용 전략",
      "validationStatus": "pass|warning|fail",
      "notes": "비고"
    }
  ],
  "recommendations": ["권장사항1", "권장사항2"],
  "coverageAnalysis": "키워드 커버리지 분석 결과"
}`;

export async function reporterAgent(
  client: Anthropic,
  analysis: KeywordAnalysis,
  queries: BigKindsQuery[],
  validations: ValidationResult[],
): Promise<AgentResult<OptimizationReport>> {
  const userMessage = `다음 키워드 최적화 결과를 종합하여 최종 리포트를 작성해주세요.

## 키워드 분석 요약
- 전체 키워드 수: ${analysis.totalKeywords}
- 카테고리 수: ${analysis.categories.length}
- 카테고리: ${analysis.categories.map((c) => `${c.name}(${c.priority})`).join(", ")}
- 중복 키워드: ${analysis.redundantKeywords.length}개
- 제거 권장: ${analysis.suggestedRemovals.length}개

## 생성된 쿼리 (${queries.length}개)
${queries
  .map(
    (q) =>
      `### 쿼리 ${q.id}: ${q.title}
\`\`\`
${q.query}
\`\`\`
전략: ${q.strategy}
사용 키워드 수: ${q.keywordsUsed.length}개`,
  )
  .join("\n\n")}

## 검증 결과
${validations
  .map(
    (v) =>
      `### 쿼리 ${v.queryId}
- 유효성: ${v.isValid ? "통과" : "실패"}
- 구문 검사: ${v.syntaxCheck ? "통과" : "실패"}
- 길이 검사: ${v.lengthCheck ? "통과" : "실패"}
- 커버리지: ${v.coverageScore}점
- 문제점: ${v.issues.length > 0 ? v.issues.join("; ") : "없음"}
- 제안: ${v.suggestions.length > 0 ? v.suggestions.join("; ") : "없음"}`,
  )
  .join("\n\n")}`;

  try {
    const response = await client.messages.create({
      model: "claude-sonnet-4-20250514",
      max_tokens: 4096,
      system: SYSTEM_PROMPT,
      messages: [{ role: "user", content: userMessage }],
    });

    const text =
      response.content[0].type === "text" ? response.content[0].text : "";
    const jsonMatch = text.match(/\{[\s\S]*\}/);
    if (!jsonMatch) {
      throw new Error("JSON 응답을 파싱할 수 없습니다.");
    }

    const data: OptimizationReport = JSON.parse(jsonMatch[0]);

    return {
      agentName: "Reporter",
      success: true,
      data,
      tokenUsage: {
        input: response.usage.input_tokens,
        output: response.usage.output_tokens,
      },
    };
  } catch (error) {
    return {
      agentName: "Reporter",
      success: false,
      data: {
        summary: "",
        inputKeywordCount: 0,
        outputQueryCount: 0,
        queries: [],
        recommendations: [],
        coverageAnalysis: "",
      },
      error: error instanceof Error ? error.message : String(error),
    };
  }
}
