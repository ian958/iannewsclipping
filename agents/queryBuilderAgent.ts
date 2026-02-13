import Anthropic from "@anthropic-ai/sdk";
import type {
  AgentResult,
  BigKindsQuery,
  KeywordAnalysis,
  SearchStrategy,
} from "../types.js";

const SYSTEM_PROMPT = `당신은 빅카인즈(BigKinds) 검색 쿼리 작성 전문가입니다.
주어진 전략과 키워드를 기반으로 실제 빅카인즈 검색 쿼리를 작성합니다.

빅카인즈 쿼리 작성 규칙:
1. OR: 키워드 중 하나라도 포함된 기사 검색 (예: LMS OR 챗봇)
2. AND: 모든 키워드가 포함된 기사 검색 (예: AI AND 교육)
3. 괄호(): 연산자 우선순위 그룹핑 (예: (AI OR 인공지능) AND 교육)
4. 쌍따옴표"": 정확한 구문 검색 (예: "인공지능 교육")
5. NOT: 특정 키워드 제외 (예: AI NOT 게임)

쿼리 작성 시 주의사항:
- 빅카인즈 쿼리 최대 길이를 고려하여 핵심 키워드 위주로 구성
- 너무 많은 OR 연산은 검색 정확도를 떨어뜨림
- 카테고리별로 그룹핑하여 의미 있는 검색 결과를 도출
- 각 쿼리는 서로 다른 관점의 뉴스를 커버해야 함

반드시 5개의 쿼리를 아래 JSON 배열 형식으로만 응답하세요:
[
  {
    "id": 1,
    "title": "쿼리 제목",
    "query": "실제 빅카인즈 검색 쿼리",
    "strategy": "사용된 전략명",
    "keywordsUsed": ["사용된", "키워드", "목록"],
    "estimatedRelevance": "예상 관련도 설명"
  }
]`;

export async function queryBuilderAgent(
  client: Anthropic,
  analysis: KeywordAnalysis,
  strategy: SearchStrategy,
): Promise<AgentResult<BigKindsQuery[]>> {
  const allKeywords = analysis.categories.flatMap((c) => c.keywords);
  const effectiveKeywords = allKeywords.filter(
    (k) => !analysis.suggestedRemovals.includes(k),
  );

  const userMessage = `다음 정보를 바탕으로 5개의 빅카인즈 검색 쿼리를 작성해주세요.

## 사용 가능한 키워드 (${effectiveKeywords.length}개)
${effectiveKeywords.join(", ")}

## 카테고리 분류
${analysis.categories.map((c) => `- ${c.name} (${c.priority}): ${c.keywords.join(", ")}`).join("\n")}

## 검색 전략 (5개)
${strategy.strategies
  .map(
    (s, i) =>
      `### 전략 ${i + 1}: ${s.name}
설명: ${s.description}
집중 카테고리: ${s.focusCategories.join(", ")}
접근 방식: ${s.approach}`,
  )
  .join("\n\n")}

## 전략 수립 근거
${strategy.rationale}

각 전략에 맞는 빅카인즈 검색 쿼리를 1개씩, 총 5개를 작성해주세요.`;

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

    const data: BigKindsQuery[] = JSON.parse(jsonMatch[0]);

    return {
      agentName: "QueryBuilder",
      success: true,
      data,
      tokenUsage: {
        input: response.usage.input_tokens,
        output: response.usage.output_tokens,
      },
    };
  } catch (error) {
    return {
      agentName: "QueryBuilder",
      success: false,
      data: [],
      error: error instanceof Error ? error.message : String(error),
    };
  }
}
