import Anthropic from "@anthropic-ai/sdk";
import type { AgentResult, BigKindsQuery, ValidationResult } from "../types.js";

const SYSTEM_PROMPT = `당신은 빅카인즈(BigKinds) 검색 쿼리 검증 전문가입니다.
주어진 검색 쿼리들을 다음 기준으로 검증합니다:

1. 구문 검증 (syntaxCheck):
   - OR, AND, NOT 연산자가 올바르게 사용되었는가
   - 괄호가 올바르게 짝이 맞는가
   - 쌍따옴표가 올바르게 닫혀있는가

2. 길이 검증 (lengthCheck):
   - 빅카인즈 쿼리 길이 제한(약 500자)을 초과하지 않는가
   - 너무 짧아서 의미 없는 검색이 되지 않는가

3. 커버리지 점수 (coverageScore, 0-100):
   - 원본 키워드를 얼마나 잘 커버하는가
   - 검색 의도에 부합하는가

4. 문제점과 개선 제안

반드시 아래 JSON 배열 형식으로만 응답하세요:
[
  {
    "queryId": 1,
    "isValid": true/false,
    "issues": ["발견된 문제점"],
    "suggestions": ["개선 제안"],
    "syntaxCheck": true/false,
    "lengthCheck": true/false,
    "coverageScore": 85
  }
]`;

export async function validatorAgent(
  client: Anthropic,
  queries: BigKindsQuery[],
  originalKeywords: string[],
): Promise<AgentResult<ValidationResult[]>> {
  const userMessage = `다음 빅카인즈 검색 쿼리들을 검증해주세요.

## 원본 키워드 목록 (${originalKeywords.length}개)
${originalKeywords.join(", ")}

## 검증 대상 쿼리 (${queries.length}개)
${queries
  .map(
    (q) =>
      `### 쿼리 ${q.id}: ${q.title}
\`\`\`
${q.query}
\`\`\`
전략: ${q.strategy}
사용 키워드: ${q.keywordsUsed.join(", ")}`,
  )
  .join("\n\n")}

각 쿼리에 대해 구문, 길이, 커버리지를 검증하고 문제점과 개선 제안을 제시해주세요.`;

  try {
    const response = await client.messages.create({
      model: "claude-sonnet-4-20250514",
      max_tokens: 4096,
      system: SYSTEM_PROMPT,
      messages: [{ role: "user", content: userMessage }],
    });

    const text =
      response.content[0].type === "text" ? response.content[0].text : "";
    const jsonMatch = text.match(/\[[\s\S]*\]/);
    if (!jsonMatch) {
      throw new Error("JSON 배열 응답을 파싱할 수 없습니다.");
    }

    const data: ValidationResult[] = JSON.parse(jsonMatch[0]);

    return {
      agentName: "Validator",
      success: true,
      data,
      tokenUsage: {
        input: response.usage.input_tokens,
        output: response.usage.output_tokens,
      },
    };
  } catch (error) {
    return {
      agentName: "Validator",
      success: false,
      data: [],
      error: error instanceof Error ? error.message : String(error),
    };
  }
}
