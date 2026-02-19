#!/bin/bash
# ============================================================
# gpush.sh: 로그 파일 자동 커밋 후 push (타임라인 누적)
# 사용법: bin/gpush.sh [git push 옵션]
#   예)  bin/gpush.sh -u origin claude/news-clipping-automation-jvDXr
#        bin/gpush.sh                (tracking 브랜치로 push)
# ============================================================

REPO_ROOT=$(git rev-parse --show-toplevel 2>/dev/null)
if [ -z "${REPO_ROOT}" ]; then
    echo "[gpush] ❌ git 저장소가 아닙니다."
    exit 1
fi
cd "${REPO_ROOT}" || exit 1

REPORT_DIR="report"

# ── 미커밋 로그 파일 감지 ──────────────────────────────────
UNTRACKED=$(git ls-files --others --exclude-standard "${REPORT_DIR}/" 2>/dev/null | \
            grep -E "\.(json|txt)$")
MODIFIED=$(git diff --name-only "${REPORT_DIR}/" 2>/dev/null | \
           grep -E "\.(json|txt)$")
TARGETS=$(printf "%s\n%s" "${UNTRACKED}" "${MODIFIED}" | sed '/^$/d' | sort -u)

if [ -n "${TARGETS}" ]; then
    echo "[gpush] 📋 미커밋 로그 파일 감지 → 타임라인 커밋 추가"
    echo "${TARGETS}" | while read -r f; do echo "  + ${f}"; done

    git add "${REPORT_DIR}"/*.json "${REPORT_DIR}"/*.txt 2>/dev/null

    TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')
    git commit -m "chore: accumulate timeline logs (${TIMESTAMP})

https://claude.ai/code/session_01QwActfPT45EHCGkxntThX8"
    echo "[gpush] ✅ 로그 커밋 완료"
fi

# ── push ──────────────────────────────────────────────────
echo "[gpush] 🚀 push 중..."
git push "$@"
