#!/usr/bin/env sh
set -eu

repo_root=$(git rev-parse --show-toplevel)
hooks_dir="$repo_root/.git/hooks"

cp -f "$repo_root/scripts/git-hooks/pre-commit" "$hooks_dir/pre-commit"
cp -f "$repo_root/scripts/git-hooks/pre-push" "$hooks_dir/pre-push"
chmod +x "$hooks_dir/pre-commit" "$hooks_dir/pre-push"

printf '%s\n' "Installed pre-commit and pre-push hooks."
