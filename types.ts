/** 키워드 분석 결과 */
export interface KeywordAnalysis {
  categories: KeywordCategory[];
  totalKeywords: number;
  redundantKeywords: string[];
  suggestedRemovals: string[];
}

export interface KeywordCategory {
  name: string;
  keywords: string[];
  description: string;
  priority: "high" | "medium" | "low";
}

/** 검색 전략 */
export interface SearchStrategy {
  strategies: QueryStrategy[];
  rationale: string;
}

export interface QueryStrategy {
  name: string;
  description: string;
  focusCategories: string[];
  approach: string;
  expectedCoverage: string;
}

/** 빅카인즈 검색 쿼리 */
export interface BigKindsQuery {
  id: number;
  title: string;
  query: string;
  strategy: string;
  keywordsUsed: string[];
  estimatedRelevance: string;
}

/** 검증 결과 */
export interface ValidationResult {
  queryId: number;
  isValid: boolean;
  issues: string[];
  suggestions: string[];
  syntaxCheck: boolean;
  lengthCheck: boolean;
  coverageScore: number;
}

/** 최종 리포트 */
export interface OptimizationReport {
  summary: string;
  inputKeywordCount: number;
  outputQueryCount: number;
  queries: OptimizedQuery[];
  recommendations: string[];
  coverageAnalysis: string;
}

export interface OptimizedQuery {
  id: number;
  title: string;
  query: string;
  strategy: string;
  validationStatus: "pass" | "warning" | "fail";
  notes: string;
}

/** 에이전트 기본 인터페이스 */
export interface AgentResult<T> {
  agentName: string;
  success: boolean;
  data: T;
  error?: string;
  tokenUsage?: { input: number; output: number };
}

/** 코디네이터 파이프라인 컨텍스트 */
export interface PipelineContext {
  rawKeywords: string[];
  rawInput: string;
  analysis?: KeywordAnalysis;
  strategy?: SearchStrategy;
  queries?: BigKindsQuery[];
  validations?: ValidationResult[];
  report?: OptimizationReport;
}
