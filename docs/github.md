# GitHub preparation

Git was already initialized on branch `master`, with no commits at inspection.
The index contained five `.idea` files plus `main.py` and `test.py`; the latter
also had working-tree edits. Refactoring did not change the index, history,
branches or remotes. No commit, repository creation, publication or push was
performed.

`.gitignore` excludes environments, caches, IDE state, local secrets, generated
recordings/exports and temporary outputs. Original generated files are kept on
disk. Small example figures and portable numeric snapshots are included under
`docs/`. The index had previously staged IDE files: an ignore rule does not
remove already-indexed entries. Before the first commit, review and untrack
those IDE files **from the index only**, keeping the local files:

```bash
git status --short
git rm --cached -r -- .idea
git add .
git diff --cached --stat
git diff --cached
python -m unittest discover -s tests -v
git commit -m "Organize and document overlapping brain-state research"
```

The `git rm --cached` command above is a recommended manual cleanup, not an
operation run by the refactor. Inspect the staged diff before committing;
existing staged scientific-script versions are replaced by compatibility
launchers while their implementations are preserved under `src/`.

Choose a repository visibility and project license with the research team.
After creating the intended GitHub repository yourself, inspect the remote and
add it only if not already configured:

```bash
git remote -v
git remote add origin https://github.com/YOUR_ORG/YOUR_REPOSITORY.git
git push -u origin HEAD
```

Replace the example URL with the team's repository. Do not add a second remote
if `origin` already exists. This workflow preserves the current branch name;
it does not force-push or rewrite history.

Source/documentation candidates were checked for obvious credential patterns
and hardcoded personal paths. Generated provenance can retain historical local
paths, so it is ignored rather than rewritten. This is a targeted review, not
a guarantee that arbitrary future files are safe to publish.
