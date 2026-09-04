# Repository Template

Default template for sunnydachs' projects. Batteries-included with a CI/CD and security stack that is entirely on free tiers.

## What's included

### CI (`ci.yml`)
- Node.js matrix test on 18.x, 20.x, 22.x
- Optional Codecov coverage upload (Node 20.x leg only)

### Security
- **CodeQL** at `security-extended` query pack (catches more than default)
- **gitleaks** pre-commit hook + CI action on every push/PR
- **Secret scanning** (GitHub native, public repos)
- **Push protection** on public repos

### Automation
- **Dependabot** version updates for npm + github-actions (weekly, grouped minor/patch)
- **Dependabot auto-merge** for patch/minor/security updates only
- **Stale workflow** to clean up old issues/PRs weekly

### Templates
- **CODEOWNERS** for personal ownership
- **pull_request_template.md** for consistent PRs
- **dependabot.yml** with grouped minor/patch updates

## Pre-installed Apps (install per repo)

| App | Purpose | Why here |
|---|---|---|
| [CodeRabbit](https://github.com/apps/coderabbit) | AI code review | Free on public repos, Advanced tier features |
| [Codecov](https://github.com/apps/codecov) | Test coverage | Free for open source |

## New-repo Setup (5 minutes)

1. `gh repo create <name> --template sunnydachs/repo-template`
2. Install [CodeRabbit](https://github.com/apps/coderabbit) and [Codecov](https://github.com/apps/codecov) GitHub Apps
3. Set `CODECOV_TOKEN` in Secrets (never echo the value; use a file)
4. Create a Repository Ruleset on `main`:
   - Required checks: `test (18.x)`, `test (20.x)`, `test (22.x)`, `Analyze (...)`, `gitleaks`
   - Bypass: Dependabot (Integration id=29110)
   - `allow_auto_merge: true`, `delete_branch_on_merge: true`

## What /ruleset does

The template workflow `.github/workflows/` assumes a Repository Ruleset named "main protection" exists on the repo. Without it, `gh pr merge --auto` will not be accepted. See neuro-profile's ruleset (id=22273820) for a complete reference.

## Notes for Agents (Hermes / Codex / Claude Code)

See `AGENTS.md` for hard rules on secrets, commits, and workflow safety.
