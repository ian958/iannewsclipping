import Anthropic from "@anthropic-ai/sdk";
import type { AgentResult, KeywordAnalysis, SearchStrategy } from "../types.js";

const SYSTEM_PROMPT = `당신은 뉴스 검색 전략 수립 전문가입니다.
빅카인즈(BigKinds) 뉴스 검색 시스템에 최적화된 검색 전략을 수립합니다.

빅카인즈 검색 특성:
- OR 연산자로 키워드를 연결
- AND 연산자로 필수 포함 조건 설정
- 괄호()로 그룹핑 가능
- 쌍따옴표""로 정확한 구문 검색
- NOT 연산자로 제외 조건 설정
- 쿼리 길이에 제한이 있으므로 효율적으로 구성해야 함

키워드 분석 결과를 받아 5개의 서로 다른 검색 전략을 수립합니다.
각 전략은 다른 관점에서 뉴스를 검색할 수 있도록 설계합니다.

반드시 아래 JSON 형식으로만 응답하세요:
{
  "strategies": [
    {
      "name": "전략명",
      "description": "전략 설명",
      "focusCategories": ["집중할 카테고리명"],
      "approach": "접근 방식 상세 설명",
      "expectedCoverage": "예상 커버리지 설명"
    }
  ],
  "rationale": "전체 전략 수립 근거"
}`;

export async function strategistAgent(
  client: Anthropic,
  analysis: KeywordAnalysis,
): Promise<AgentResult<SearchStrategy>> {
  const userMessage = `다음 키워드 분석 결과를 바탕으로 5개의 빅카인즈 검색 전략을 수립해주세요:

카테고리 분류:
${analysis.categories
  .map(
    (c) =>
      `- ${c.name} (${c.priority}): ${c.keywords.join(", ")}\n  설명: ${c.description}`,
  )
  .join("\n")}

전체 키워드 수: ${analysis.totalKeywords}
중복 키워드: ${analysis.redundantKeywords.join(", ") || "없음"}
제거 권장: ${analysis.suggestedRemovals.join(", ") || "없음"}`;

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

    const data: SearchStrategy = JSON.parse(jsonMatch[0]);

    return {
      agentName: "Strategist",
      success: true,
      data,
      tokenUsage: {
        input: response.usage.input_tokens,
        output: response.usage.output_tokens,
      },
    };
  } catch (error) {
    return {
      agentName: "Strategist",
      success: false,
      data: { strategies: [], rationale: "" },
      error: error instanceof Error ? error.message : String(error),
    };
  }
}
