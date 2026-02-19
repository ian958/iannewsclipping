#!/bin/bash
# ============================================================
# gpush.sh: 신규 로그 파일 자동 커밋 후 push (타임라인 누적)
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

# ── 신규 미추적 파일만 감지 (삭제·수정 파일 제외) ────────────
UNTRACKED=$(git ls-files --others --exclude-standard "${REPORT_DIR}/" 2>/dev/null | \
            grep -E "\.(json|jsonl|txt)$")

if [ -n "${UNTRACKED}" ]; then
    echo "[gpush] 📋 신규 로그 파일 감지 → 타임라인 커밋 추가"
    echo "${UNTRACKED}" | while read -r f; do echo "  + ${f}"; done

    echo "${UNTRACKED}" | xargs git add 2>/dev/null

    TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')
    git commit -m "chore: accumulate timeline logs (${TIMESTAMP})

https://claude.ai/code/session_01QwActfPT45EHCGkxntThX8"
    echo "[gpush] ✅ 로그 커밋 완료"
fi

# ── push ──────────────────────────────────────────────────
echo "[gpush] 🚀 push 중..."
git push "$@"
