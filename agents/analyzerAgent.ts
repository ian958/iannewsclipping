import Anthropic from "@anthropic-ai/sdk";
import type { AgentResult, KeywordAnalysis } from "../types.js";

const SYSTEM_PROMPT = `당신은 키워드 분석 전문가입니다.
주어진 키워드 목록을 분석하여 다음을 수행합니다:

1. 키워드를 의미적 카테고리로 분류합니다.
2. 각 카테고리의 우선순위를 결정합니다 (high/medium/low).
3. 중복되거나 불필요한 키워드를 식별합니다.
4. 제거를 권장하는 키워드를 제안합니다.

반드시 아래 JSON 형식으로만 응답하세요:
{
  "categories": [
    {
      "name": "카테고리명",
      "keywords": ["키워드1", "키워드2"],
      "description": "이 카테고리의 설명",
      "priority": "high|medium|low"
    }
  ],
  "totalKeywords": 숫자,
  "redundantKeywords": ["중복키워드1"],
  "suggestedRemovals": ["제거권장키워드1"]
}`;

export async function analyzerAgent(
  client: Anthropic,
  keywords: string[],
): Promise<AgentResult<KeywordAnalysis>> {
  const userMessage = `다음 ${keywords.length}개의 키워드를 분석해주세요:\n\n${keywords.join(", ")}`;

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

    const data: KeywordAnalysis = JSON.parse(jsonMatch[0]);

    return {
      agentName: "Analyzer",
      success: true,
      data,
      tokenUsage: {
        input: response.usage.input_tokens,
        output: response.usage.output_tokens,
      },
    };
  } catch (error) {
    return {
      agentName: "Analyzer",
      success: false,
      data: {
        categories: [],
        totalKeywords: keywords.length,
        redundantKeywords: [],
        suggestedRemovals: [],
      },
      error: error instanceof Error ? error.message : String(error),
    };
  }
}
