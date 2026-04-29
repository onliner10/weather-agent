#!/usr/bin/env sh
set -eu

mode="${1:---staged}"

token_pattern='(^|[^A-Za-z0-9_-])(sk-proj-[A-Za-z0-9_-]{20,}|sk-[A-Za-z0-9_-]{20,}|gh[pousr]_[A-Za-z0-9_]{20,}|xox[baprs]-[A-Za-z0-9-]{20,}|[0-9]{8,10}:AA[A-Za-z0-9_-]{30,}|BEGIN (RSA|OPENSSH|PRIVATE) KEY)'
secret_file_pattern='(^|/)(\.env($|\.)|.*\.(pem|key|p12|pfx|kubeconfig|ovpn)$)'

fail() {
  printf '%s\n' "$1" >&2
  exit 1
}

scan_staged() {
  staged_files=$(git diff --cached --name-only --diff-filter=ACMR)
  [ -n "$staged_files" ] || return 0

  blocked_files=$(printf '%s\n' "$staged_files" | grep -E "$secret_file_pattern" | grep -v '^\.env\.example$' || true)
  if [ -n "$blocked_files" ]; then
    printf '%s\n' "Refusing to commit likely secret-bearing files:" >&2
    printf '%s\n' "$blocked_files" >&2
    fail "Move secrets to local ignored files or add a safe example file instead."
  fi

  found=0
  for file in $staged_files; do
    if git show ":$file" 2>/dev/null | LC_ALL=C grep -E -q "$token_pattern"; then
      printf '%s\n' "Potential secret token pattern in staged file: $file" >&2
      found=1
    fi
  done

  [ "$found" -eq 0 ] || fail "Commit blocked by secret scan. Rotate the secret if it was real."
}

scan_history() {
  revs=$(git rev-list --all 2>/dev/null || true)
  [ -n "$revs" ] || return 0

  matches=$(git grep -l -I -E "$token_pattern" $revs -- . 2>/dev/null || true)
  if [ -n "$matches" ]; then
    printf '%s\n' "Potential secret token pattern found in git history:" >&2
    printf '%s\n' "$matches" | sort -u >&2
    fail "Push blocked by history secret scan. Remove the secret from history and rotate it."
  fi
}

run_gitleaks_if_available() {
  command -v gitleaks >/dev/null 2>&1 || return 0

  case "$mode" in
    --staged)
      gitleaks protect --staged --redact --verbose
      ;;
    --history|--all)
      gitleaks detect --redact --verbose
      ;;
  esac
}

case "$mode" in
  --staged)
    scan_staged
    run_gitleaks_if_available
    ;;
  --history)
    scan_history
    run_gitleaks_if_available
    ;;
  --all)
    scan_staged
    scan_history
    run_gitleaks_if_available
    ;;
  *)
    printf 'Usage: %s [--staged|--history|--all]\n' "$0" >&2
    exit 2
    ;;
esac
