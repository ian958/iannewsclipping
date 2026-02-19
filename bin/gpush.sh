#!/bin/bash
# ============================================================
# gpush.sh: 로그 아카이브 파일 자동 커밋 후 push (타임라인 누적)
# 사용법: bash bin/gpush.sh [git push 옵션]
#   예)  bash bin/gpush.sh -u origin claude/news-clipping-automation-jvDXr
#        bash bin/gpush.sh                (tracking 브랜치로 push)
# ============================================================

REPO_ROOT=$(git rev-parse --show-toplevel 2>/dev/null)
if [ -z "${REPO_ROOT}" ]; then
    echo "[gpush] ❌ git 저장소가 아닙니다."
    exit 1
fi
cd "${REPO_ROOT}" || exit 1

REPORT_DIR="report"

# ── 신규 미추적 파일 감지 (.json/.jsonl/.txt) ─────────────────
UNTRACKED=$(git ls-files --others --exclude-standard "${REPORT_DIR}/" 2>/dev/null | \
            grep -E "\.(json|jsonl|txt)$")

# ── 수정된 아카이브 파일 감지 (query_candidates_archive / process_log_archive) ─
MODIFIED_ARCHIVES=$(git diff --name-only "${REPORT_DIR}/" 2>/dev/null | \
                    grep -E "(query_candidates_archive|process_log_archive)\.(txt|jsonl)$")

TARGETS=$(printf "%s\n%s" "${UNTRACKED}" "${MODIFIED_ARCHIVES}" | sed '/^$/d' | sort -u)

if [ -n "${TARGETS}" ]; then
    echo "[gpush] 📋 로그 파일 감지 → 타임라인 커밋 추가"
    echo "${TARGETS}" | while read -r f; do echo "  + ${f}"; done

    echo "${TARGETS}" | xargs git add 2>/dev/null

    TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')
    git commit -m "chore: accumulate timeline logs (${TIMESTAMP})

https://claude.ai/code/session_01QwActfPT45EHCGkxntThX8"
    echo "[gpush] ✅ 로그 커밋 완료"
fi

# ── push ──────────────────────────────────────────────────
echo "[gpush] 🚀 push 중..."
git push "$@"
