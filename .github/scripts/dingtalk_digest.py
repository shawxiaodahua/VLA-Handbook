#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build the DingTalk daily digest body for the VLA-Handbook fork sync workflow.

Usage:
  python3 dingtalk_digest.py --old-sha <sha> --count <n> --repo-url <url> > body.md

The diff base is the post-merge HEAD (tree-level, rename-aware), so a
dissection added and swept into a topic subdir the same day collapses to one
entry at its final path, and next-day sweeps show up as R100 and are skipped.
File content is read via `git show HEAD:<path>`, never the worktree, so local
dry-runs and CI behave identically.

Stdlib only. Exits non-zero on git failure so the workflow's failure-notify
step fires. Prints a section summary and UNCOVERED file list to stderr for
debugging in Actions logs.
"""

from __future__ import print_function

import argparse
import datetime
import re
import subprocess
import sys

REPO_URL = ""
PER_SECTION = 10


def die(msg, code=3):
    sys.stderr.write("dingtalk_digest: %s\n" % msg)
    sys.exit(code)


def git(*args):
    """Run a git command, return stdout as text. Dies on git failure."""
    proc = subprocess.run(
        ["git"] + list(args),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if proc.returncode != 0:
        die("git %s failed: %s" % (" ".join(args[:3]),
                                   proc.stderr.decode("utf-8", "replace").strip()))
    return proc.stdout.decode("utf-8", "replace")


def git_lines(*args):
    return git(*args).splitlines()


# ---------------------------------------------------------------------------
# plumbing


def name_status(old):
    """[(status, path)] from tree diff old..HEAD, rename detection at 100%.

    R100 entries keep the post-rename path, so same-day add+sweep collapses
    to the final location and next-day sweeps are identifiable by status.
    """
    out = []
    for line in git_lines("diff", "--name-status", "-M100%", old, "HEAD"):
        parts = line.split("\t")
        if len(parts) >= 2:
            out.append((parts[0], parts[1].strip('"')))
    return out


def show_head(path, limit=40):
    """First N lines of a file at HEAD (empty list if missing)."""
    content = git("show", "HEAD:" + path)
    return content.splitlines()[:limit]


def added_lines(old, path):
    """Lines added to a modified file in old..HEAD."""
    out = []
    for line in git_lines("diff", old, "HEAD", "--", path):
        if line.startswith("+") and not line.startswith("+++"):
            out.append(line[1:])
    return out


def truncate(text, width):
    text = re.sub(r"\s+", " ", text or "").strip()
    if len(text) > width:
        return text[: width - 1] + "…"
    return text


def sanitize(text):
    """Strip characters that break DingTalk markdown link text."""
    return re.sub(r"\s+", " ", re.sub(r"[\[\]|\n\r]", " ", text or "")).strip()


def first_url(text):
    """First http(s) URL in text, preferring arxiv.org."""
    urls = re.findall(r"https?://[^\s\)\]]+", text or "")
    for u in urls:
        if "arxiv.org" in u:
            return u
    return urls[0] if urls else ""


def blob_url(path):
    return "%s/blob/main/%s" % (REPO_URL.rstrip("/"), path)


def _table_cells(line):
    return [c.strip() for c in line.replace("\\|", "·").strip("|").split("|")]


# ---------------------------------------------------------------------------
# section extractors — uniform (old, ns) signature, return [{line, cat}]


def section_dissections(old, ns):
    paths = [p for (s, p) in ns
             if s == "A" and re.match(r"^theory/.+_dissection\.md$", p)]
    if not paths:
        return []
    # newest-first: order by the add commit, matched via basename (renames
    # move files into topic subdirs, so full paths don't line up)
    order = {}
    for i, line in enumerate(git_lines(
            "log", "--diff-filter=A", "--format=%x01", "--name-only",
            old + "..HEAD", "--", "theory/")):
        p = line.strip()
        if p and p != "\x01":
            order.setdefault(p.rsplit("/", 1)[-1], i)
    paths.sort(key=lambda p: order.get(p.rsplit("/", 1)[-1], 10 ** 9))
    items = []
    for path in paths[:6]:

        title, link, tagline = "", "", ""
        for line in show_head(path, 12):
            if line.startswith("# ") and not title:
                title = line[2:].strip()
            if not link and re.match(r"^>?\s*\*\*(?:鏈接|链接|連結)\*\*", line):
                link = first_url(line)
            m = re.match(r"^>?\s*\*\*核心定位\*\*:?\s*(.+)", line)
            if m and not tagline:
                tagline = m.group(1)
        if not title:
            title = path.rsplit("/", 1)[-1].replace("_dissection.md", "")
        # H1 is "中文标题 (English Title)" — the Chinese half is enough
        if " (" in title and title.endswith(")"):
            title = title.rsplit(" (", 1)[0]
        if link:
            text = "- **[%s](%s)**" % (sanitize(truncate(title, 70)), link)
        else:
            text = "- %s" % sanitize(truncate(title, 70))
        text += " ｜ [📖 解读](%s)" % blob_url(path)
        if tagline:
            items.append({"line": text + "\n  " + truncate(tagline, 80),
                          "cat": "tagline"})
        else:
            items.append({"line": text, "cat": None})
    if len(paths) > 6:
        items.append({"line": "……（另有 %d 篇，[见仓库](%s)）" % (
            len(paths) - 6, blob_url("theory/")), "cat": "overflow"})
    return items


def section_paper_index(old, ns):
    path = "theory/foundation/paper_index.md"
    if path not in [p for (_s, p) in ns]:
        return []
    items = []
    for line in added_lines(old, path):
        if not re.match(r"^\|.+\|\s*\[link\]\(", line):
            continue  # skips separators and curated-section noise
        cells = _table_cells(line)
        if len(cells) < 3:
            continue
        title, url, note = cells[0], first_url(cells[1]), cells[2]
        marker = next((m for m in ("⚡", "🔧", "📖") if m in note), "📖")
        if url:
            text = "- %s **[%s](%s)**" % (marker,
                                           sanitize(truncate(title, 60)), url)
        else:
            text = "- %s %s" % (marker, sanitize(truncate(title, 60)))
        items.append({"line": text, "cat": "paper"})
        if len(items) >= PER_SECTION:
            break
    return items


def section_sota_release(old, ns):
    changed = [p for (_s, p) in ns]
    items = []
    for path, cat, subtitle in [
            ("theory/benchmark_tracker.md", "bench",
             "**📈 SOTA 榜变动** ｜ [详情](%s)"),
            ("deployment/release_tracker.md", "release",
             "**🔧 Release 追踪** ｜ [详情](%s)")]:
        if path not in changed:
            continue
        rows = []
        for line in added_lines(old, path):
            # daily header-date refresh is '>'-prefixed prose, not table rows
            if not re.match(r"^\|\s*\d{4}-\d{2}-\d{2}\s*\|", line):
                continue
            cells = _table_cells(line)
            if cat == "bench" and len(cells) >= 5:
                rows.append("- %s · %s = %s（%s）" % (
                    sanitize(cells[1]), sanitize(cells[2]), cells[3], cells[4]))
            elif cat == "release" and len(cells) >= 7:
                mark = cells[4] if cells[4] in ("⚡", "🔧", "📖") else ""
                rows.append("- %s %s（%s）" % (
                    mark, sanitize(truncate(cells[3], 70)), sanitize(cells[1])))
            else:
                continue
            if len(rows) >= 5:
                break
        if rows:
            # subtitle shares the rows' trim category so the guard never
            # leaves an orphan header behind
            items.append({"line": subtitle % blob_url(path), "cat": cat})
            items.extend({"line": r, "cat": cat} for r in rows)
    return items


def section_biweekly(old, ns):
    paths = [p for (s, p) in ns
             if s == "A" and re.match(r"^reports/biweekly/.+\.md$", p)
             and not p.endswith("README.md")]
    paths.sort(reverse=True)  # newest report date first
    items = []
    for path in paths[:4]:
        title = ""
        for line in show_head(path, 5):
            if line.startswith("# ") or line.startswith("🤔"):
                title = line.lstrip("# ").strip()
                break
        items.append({"line": "- **[%s](%s)**" % (
            sanitize(truncate(title or path, 70)), blob_url(path)),
            "cat": None})
    if len(paths) > 4:
        items.append({"line": "……（另有 %d 期，[见仓库](%s)）" % (
            len(paths) - 4, blob_url("reports/biweekly/")), "cat": "overflow"})
    return items


def _summary_bullets(lines):
    """Bullets under the 📊 動態摘要 heading (简体/繁体/（二级） variants)."""
    bullets, cur, in_section = [], "", False
    for line in lines:
        if re.match(r"^#{0,3}\s*📊", line):
            in_section = True
            continue
        if not in_section:
            continue
        s = line.strip()
        if s.startswith("- "):
            if cur:
                bullets.append(cur)
            cur = s[2:]
        elif cur and s and not re.match(r"^#{0,3}\s*[🔴🔭📊]", line) \
                and not s.startswith("---"):
            cur += " " + s  # continuation / source URL line
        elif not s and cur:
            bullets.append(cur)
            cur = ""
    if cur:
        bullets.append(cur)
    return bullets


def section_social_intel(old, ns):
    paths = [p for (s, p) in ns
             if s == "A" and re.match(
                 r"^memory/blog/archives/vla-social-intel/"
                 r"\d{4}-\d{2}-\d{2}\.md$", p)]
    paths.sort(reverse=True)  # newest daily file first
    sig_items, trend_items = [], []
    for path in paths[:PER_SECTION]:
        lines = show_head(path, 80)
        signals, trend = [], ""
        mode, cur = None, ""  # None | "signal" | "trend"
        for line in lines:
            s = line.strip()
            if re.match(r"^#{0,3}\s*🔴", line):
                if cur:
                    signals.append(cur)
                    cur = ""
                mode = None if re.search(r"今日无|今日無", line) else "signal"
                continue
            if re.match(r"^#{0,3}\s*[📊🔭]", line) or s.startswith("---") \
                    or s.startswith("*"):
                if mode == "trend" and trend:
                    mode = None
                elif re.match(r"^#{0,3}\s*🔭", line):
                    if cur:
                        signals.append(cur)
                        cur = ""
                    mode = "trend"
                elif mode == "signal" and s and not s.startswith("-"):
                    mode = None
                continue
            if mode == "trend":
                if s:
                    trend = truncate((trend + " " + s).strip(), 160)
            elif mode == "signal":
                if s.startswith("- "):
                    if cur:
                        signals.append(cur)
                    cur = s[2:]
                elif cur and s:
                    cur += " " + s  # continuation / source URL
        if cur:
            signals.append(cur)
        # some files have an empty 🔴 section — fall back to 📊 summary bullets
        if not signals:
            signals = _summary_bullets(lines)
        for sig in signals[:3]:
            url = first_url(sig)
            text = re.sub(r"https?://\S+", "", sig)
            text = text.replace("（無一手來源連結）", "").strip(" ：:")
            text = "- " + truncate(text, 90)
            if url:
                text += " [来源](%s)" % url
            sig_items.append({"line": text, "cat": "signal"})
        if trend:
            trend_items.append({"line": "🔭 " + trend, "cat": None})
    return sig_items[:6] + trend_items[:2]


def section_radar(old, ns):
    paths = [p for (s, p) in ns
             if s == "A" and re.match(
                 r"^memory/blog/archives/industry-radar/"
                 r"(backfill-)?\d{4}-\d{2}-\d{2}\.md$", p)]
    rank = {"⚡": 0, "🔧": 1, "📖": 2}
    rows = []
    for path in paths:
        for line in git("show", "HEAD:" + path).splitlines():
            if not re.match(r"^\|\s*(⚡|🔧|📖)\s*\|", line):
                continue
            cells = _table_cells(line)
            if len(cells) < 4:
                continue
            mark = cells[0]
            company = cells[2] if len(cells) > 2 else ""
            event = cells[3] if len(cells) > 3 else ""
            url = first_url(cells[-1])
            text = "- %s **%s**：%s" % (
                mark, sanitize(truncate(company, 25)) or "—",
                truncate(event, 80))
            if url:
                text += " [来源](%s)" % url
            rows.append((rank.get(mark, 9), mark, text))
    rows.sort(key=lambda r: r[0])
    items = [{"line": t, "cat": "radar:" + m} for (_r, m, t) in rows[:PER_SECTION]]
    if len(rows) > PER_SECTION:
        items.append({"line": "……（共 %d 条，[查看全部](%s)）" % (
            len(rows), blob_url(paths[0])), "cat": "overflow"})
    return items


# ---------------------------------------------------------------------------
# assembly


SECTIONS = [
    ("🆕 新增解读文章", section_dissections),
    ("📄 paper_index 新增", section_paper_index),
    ("📈 SOTA / Release 变动", section_sota_release),
    ("🧾 双周报", section_biweekly),
    ("📡 专家信号与趋势", section_social_intel),
    ("🏭 行业雷达", section_radar),
]

CLAIMED_PATTERNS = [
    r"^theory/.+_dissection\.md$",
    r"^theory/foundation/paper_index\.md$",
    r"^theory/benchmark_tracker\.md$",
    r"^deployment/release_tracker\.md$",
    r"^reports/biweekly/.+\.md$",
    r"^memory/blog/archives/vla-social-intel/\d{4}-\d{2}-\d{2}\.md$",
    r"^memory/blog/archives/industry-radar/(backfill-)?\d{4}-\d{2}-\d{2}\.md$",
    r"README\.md$",
]


def assemble(header, sections):
    body = [header, ""]
    for title, items in sections:
        if not items:
            continue
        count = len([i for i in items if i.get("cat") != "overflow"])
        body.append("### %s（%d）" % (title, count))
        body.extend(i["line"] for i in items)
        body.append("")
    body.append("[查看仓库](%s)" % REPO_URL.rstrip("/"))
    return "\n".join(body)


def trim_drop(sections, cat):
    dropped = False
    for _t, items in sections:
        before = len(items)
        items[:] = [i for i in items if i.get("cat") != cat]
        dropped = dropped or len(items) < before
    return dropped


def shorten_taglines(sections):
    changed = False
    for _t, items in sections:
        for i in items:
            if i.get("cat") == "tagline" and "\n  " in i["line"]:
                head, tag = i["line"].split("\n  ", 1)
                if len(tag) > 40:
                    i["line"] = head + "\n  " + truncate(tag, 40)
                    changed = True
    return changed


def main():
    global REPO_URL, PER_SECTION
    ap = argparse.ArgumentParser()
    ap.add_argument("--old-sha", required=True)
    ap.add_argument("--count", default="0")
    ap.add_argument("--repo-url",
                   default="https://github.com/shawxiaodahua/VLA-Handbook")
    ap.add_argument("--max-chars", type=int, default=3300)
    ap.add_argument("--per-section", type=int, default=10)
    args = ap.parse_args()
    REPO_URL, PER_SECTION = args.repo_url, args.per_section

    ns = name_status(args.old_sha)
    sections = [(title, fn(args.old_sha, ns)) for title, fn in SECTIONS]

    # log A-status files no extractor claims (extension point)
    for (_s, path) in ns:
        if _s == "A" and not any(re.match(p, path) for p in CLAIMED_PATTERNS):
            sys.stderr.write("UNCOVERED: %s\n" % path)

    header = "## 📚 VLA-Handbook 每日更新\n\n**%s · 新增 %s 个提交**" % (
        datetime.date.today().isoformat(), args.count)

    # fallback: content-free day (sweep-only etc.) → legacy commit list
    if not any(items for _t, items in sections):
        commits = git_lines("log", args.old_sha + "..HEAD", "--no-merges",
                            "--pretty=format:- %s")[:25]
        body = "\n".join([header, ""] + commits +
                         ["", "[查看仓库](%s)" % REPO_URL.rstrip("/")])
        sys.stderr.write("sections: (fallback commit list)\n")
        sys.stdout.write(body + "\n")
        return

    # size guard: drop low-value items in defined order. Signals and
    # dissections are the core value — trim filler before touching them.
    def trim_dissections(n):
        if len(sections[0][1]) > n:
            del sections[0][1][n:]
            return True
        return False

    guard_steps = [
        lambda: trim_drop(sections, "radar:📖"),
        lambda: trim_drop(sections, "radar:🔧"),
        lambda: trim_drop(sections, "paper"),
        lambda: shorten_taglines(sections),
        lambda: trim_dissections(4),
        lambda: trim_drop(sections, "signal"),
        lambda: trim_drop(sections, "release"),
    ]
    for step in guard_steps:
        if len(assemble(header, sections)) <= args.max_chars:
            break
        step()

    body = assemble(header, sections)
    if len(body) > args.max_chars:
        # last resort (multi-month catch-up): hard cut at a line boundary,
        # keeping the repo link so the message always ends somewhere useful
        note = "…（内容超长已截断）｜ [查看仓库](%s)" % REPO_URL.rstrip("/")
        budget = args.max_chars - len(note)
        kept, total = [], 0
        for ln in body.split("\n"):
            if total + len(ln) + 1 > budget:
                break
            kept.append(ln)
            total += len(ln) + 1
        body = "\n".join(kept) + "\n" + note
    sys.stderr.write("sections: %s | total_chars=%d\n" % (
        ",".join(t for t, items in sections if items), len(body)))
    sys.stdout.write(body + "\n")


if __name__ == "__main__":
    main()
