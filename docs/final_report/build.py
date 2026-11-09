r"""Build report.html from report.md (the source) + style.css + assets/.   Usage:  python docs/report/build.py

No dependencies: a small Markdown converter covering what the report uses.

Markdown dialect
----------------
Front matter      --- / key: value lines / ---   (title, eyebrow, author, affiliation, footer)
Headings          ## 1 Problem and motivation {#problem}    numbered section: goes in the contents bar
                  ### 2.1 Scene      #### Ports
Inline            **bold**  *italic*  `code`  [link](url)  <https://autolink>  and raw HTML such as <sub>j</sub>, <span class="c">..</span>
Maths (LaTeX)     $\lambda^2$ inline;   $$ ... $$ on its own lines for a display formula (typeset by MathJax)
Lists             - item      1. item
Tables            | a | b |  with  | --- | ---: |  (---: right-aligns that column)
Definition list   Term  then a line  : definition
Equations         ```eq  ...  ```   (monospace box; HTML like <sub> allowed inside)
Boxes             ::: abstract Abstract   ...   :::        (any class name, then an optional eyebrow title)
Figures           an image on its own line, then (optionally) its caption as the next paragraph:
                      ![alt text](assets/x.png)
                      **Figure 7.** The caption.
                  .mp4/.webm -> a video player.   !include assets/diagram.svg -> pasted in as-is.
                  Options in braces after it:  ![..](..){blueprint width=720}
                      width=N caps the figure at N px; any other word is added as a CSS class.
                  A paragraph counts as a caption when it starts with **Figure N.** or **Video N.**
"""

import html
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
FONTS = ("https://fonts.googleapis.com/css2?family=Archivo:wght@500;600;700;800&family=Newsreader:ital,opsz,wght@"
         "0,6..72,400;0,6..72,500;0,6..72,600;1,6..72,400&family=JetBrains+Mono:wght@400;500&display=swap")
MATHJAX = "https://cdnjs.cloudflare.com/ajax/libs/mathjax/3.2.2/es5/tex-svg.min.js"  # LaTeX -> SVG, no CSS needed


# ---------------------------------------------------------------------------------------------- inline
def inline(text: str) -> str:
    """Markdown inline syntax -> HTML. Raw HTML tags and backslash escapes pass through untouched."""
    keep = []  # pieces protected from the regexes below; restored at the end

    def stash(s):
        keep.append(s)
        return f"\x00{len(keep) - 1}\x00"

    # LaTeX first, so Markdown never touches its backslashes, underscores or asterisks. MathJax finds \( \).
    text = re.sub(r"(?<![\\$])\$(?!\s)([^$\n]+?)(?<!\s)\$",
                  lambda m: stash(r"\(" + html.escape(m.group(1), quote=False) + r"\)"), text)
    text = re.sub(r"`([^`]+)`", lambda m: stash(f"<code>{html.escape(m.group(1), quote=False)}</code>"), text)
    text = re.sub(r"\\([\\`*_\[\]#|<>-])", lambda m: stash(html.escape(m.group(1), quote=False)), text)
    text = re.sub(r"<(https?://[^<>\s]+)>", lambda m: stash(f'<a href="{m.group(1)}">{m.group(1)}</a>'), text)
    text = re.sub(r"</?[A-Za-z][^<>]*>", lambda m: stash(m.group(0)), text)  # raw HTML tags
    text = re.sub(r"&(?!#?\w+;)", "&amp;", text).replace("<", "&lt;").replace(">", "&gt;")
    text = re.sub(r"\[([^\]]+)\]\(([^)\s]+)\)", r'<a href="\2">\1</a>', text)
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"(?<![\w*])\*(?!\s)(.+?)(?<!\s)\*(?![\w*])", r"<em>\1</em>", text)
    while "\x00" in text:
        text = re.sub(r"\x00(\d+)\x00", lambda m: keep[int(m.group(1))], text)
    return text


# ---------------------------------------------------------------------------------------------- blocks
def split_blocks(lines: list[str]) -> list[list[str]]:
    """Group lines into blocks: blank-line separated, but ``` and ::: fences stay whole."""
    blocks, cur, fence = [], [], None
    for line in lines:
        if fence:
            cur.append(line)
            if line.strip() == fence:
                blocks.append(cur)
                cur, fence = [], None
            continue
        if line.strip() == "$$" or (line.startswith("$$") and not line.rstrip().endswith("$$")):
            if cur:
                blocks.append(cur)
            cur, fence = [line], "$$"
        elif line.startswith("```") or line.startswith(":::"):
            if cur:
                blocks.append(cur)
            cur, fence = [line], ("```" if line.startswith("```") else ":::")
        elif not line.strip():
            if cur:
                blocks.append(cur)
            cur = []
        else:
            cur.append(line)
    if cur:
        blocks.append(cur)
    return blocks


def table(lines):
    def cells(row):
        return [c.strip() for c in re.split(r"(?<!\\)\|", row.strip().strip("|"))]

    head, align, rows = cells(lines[0]), cells(lines[1]), [cells(r) for r in lines[2:]]
    right = [a.endswith(":") for a in align]
    cls = lambda i: ' class="r"' if right[i] else ""
    out = ['<div class="tablewrap"><table>',
           "<thead><tr>" + "".join(f"<th{cls(i)}>{inline(c)}</th>" for i, c in enumerate(head)) + "</tr></thead><tbody>"]
    out += ["<tr>" + "".join(f"<td{cls(i)}>{inline(c)}</td>" for i, c in enumerate(r)) + "</tr>" for r in rows]
    return "\n".join(out + ["</tbody></table></div>"])


def media(line):
    alt, src = re.match(r"!\[(.*)\]\((\S+)\)", line).groups()
    if src.lower().endswith((".mp4", ".webm", ".mov")):
        return f'<video controls muted playsinline preload="metadata" src="{src}" aria-label="{html.escape(alt)}"></video>'
    return f'<div class="plate"><img src="{src}" alt="{html.escape(alt)}"></div>'


MEDIA = re.compile(r"^(!\[.*\]\(\S+\)|!include \S+)\s*(?:\{([^}]*)\})?\s*$")
CAPTION = re.compile(r"^\*\*(Figure|Video) \d+\.\*\*")


def figure(line, caption=None):
    """An image, video or included diagram, plus its caption. Figures may be wider than the text column."""
    target, opts = MEDIA.match(line).groups()
    opts = (opts or "").split()
    classes = ["wide"] + [o for o in opts if "=" not in o]
    width = next((o.split("=")[1] for o in opts if o.startswith("width=")), None)
    style = f' style="max-width:{width}px"' if width else ""
    if target.startswith("!include"):
        path = target.split(None, 1)[1]
        content = (HERE / path).read_text().strip()
        body = f'<div class="svgwrap">{content}</div>' if path.endswith(".svg") else content
    else:
        body = media(target)
    cap = f"<figcaption>{inline(' '.join(caption))}</figcaption>" if caption else ""
    return f'<figure class="{" ".join(classes)}"{style}>\n{body}\n{cap}\n</figure>'


def render_block(block):
    """One block -> (html, is_wide). Wide figures sit outside the text column."""
    first = block[0]
    if first.startswith("$$"):  # display maths: everything between the $$ markers
        tex = "\n".join(block).strip()[2:-2].strip()
        return f'<div class="math">\\[{html.escape(tex, quote=False)}\\]</div>', False
    if first.startswith("```"):
        body = "\n".join(block[1:-1])
        if first.strip() == "```eq":
            return f'<div class="eq">{body}</div>', False
        return f"<pre><code>{html.escape(body)}</code></pre>", False
    if first.startswith(":::"):
        name = first.split()[1]
        inner = block[1:-1]
        title = first.split(None, 2)[2] if len(first.split()) > 2 else ""
        eyebrow = f'<div class="eyebrow">{inline(title)}</div>' if title else ""
        body = "\n".join(render_block(b)[0] for b in split_blocks(inner))
        return f'<div class="{name}">{eyebrow}\n{body}\n</div>', False
    if first.startswith("#"):
        level = len(first) - len(first.lstrip("#"))
        text = first[level:].strip()
        return f"<h{level}>{inline(text)}</h{level}>", False
    if first.startswith("|"):
        return table(block), False
    if re.match(r"(- |\d+\. )", first):
        ordered = bool(re.match(r"\d+\. ", first))
        items, cur = [], None
        for line in block:
            m = re.match(r"(- |\d+\. )(.*)", line)
            if m:
                items.append(m.group(2))
            else:  # a wrapped continuation line
                items[-1] += " " + line.strip()
        tag = "ol" if ordered else "ul"
        return f"<{tag}>" + "".join(f"<li>{inline(i)}</li>" for i in items) + f"</{tag}>", False
    if len(block) >= 2 and block[1].startswith(": "):
        return f"<dt>{inline(first)}</dt><dd>{inline(' '.join(l[2:] if l.startswith(': ') else l for l in block[1:]))}</dd>", False
    if first.startswith("<"):  # a raw HTML block
        return "\n".join(block), False
    return f"<p>{inline(' '.join(l.strip() for l in block))}</p>", False


# ---------------------------------------------------------------------------------------------- page
def build(src=HERE / "report.md", out=HERE / "report.html"):
    text = src.read_text()
    meta = {}
    m = re.match(r"---\n(.*?)\n---\n", text, re.S)
    if m:
        for line in m.group(1).splitlines():
            key, _, value = line.partition(":")
            meta[key.strip()] = value.strip()
        text = text[m.end():]

    sections, toc = [], []  # each section: [id, [(html, wide), ...]]
    sections.append([None, []])
    pending_dl = []
    blocks = split_blocks(text.splitlines())
    i = -1
    while i + 1 < len(blocks):
        i += 1
        block = blocks[i]
        if len(block) == 1 and MEDIA.match(block[0]):
            caption = None
            if i + 1 < len(blocks) and CAPTION.match(blocks[i + 1][0]):
                caption = blocks[i + 1]
                i += 1
            sections[-1][1].append((figure(block[0], caption), True))
            continue
        h = re.match(r"## (.+?)\s*\{#([\w-]+)\}\s*$", block[0])
        if h:
            title, sid = h.groups()
            num, _, rest = title.partition(" ")
            if not re.fullmatch(r"[0-9A-Z]{1,2}", num):
                num, rest = "", title
            sections.append([sid, []])
            toc.append(f'<a href="#{sid}"><span>{num}</span>{inline(rest)}</a>')
            label = f'<span class="num">{num}</span>' if num else ""
            sections[-1][1].append((f"<h2>{label}{inline(rest)}</h2>", False))
            continue
        html_block, wide = render_block(block)
        if html_block.startswith("<dt>"):  # consecutive definitions form one list
            pending_dl.append(html_block)
            continue
        if pending_dl:
            sections[-1][1].append(('<dl class="kv">' + "".join(pending_dl) + "</dl>", False))
            pending_dl = []
        sections[-1][1].append((html_block, wide))
    if pending_dl:
        sections[-1][1].append(('<dl class="kv">' + "".join(pending_dl) + "</dl>", False))

    body = []
    for sid, items in sections:
        if not items:
            continue
        body.append(f'<section id="{sid}">' if sid else "<section>")
        col = []
        for piece, wide in items:  # text goes in the reading column; wide figures break out of it
            if wide:
                if col:
                    body.append('<div class="col">\n' + "\n".join(col) + "\n</div>")
                    col = []
                body.append(piece)
            else:
                col.append(piece)
        if col:
            body.append('<div class="col">\n' + "\n".join(col) + "\n</div>")
        body.append("</section>")

    byline = "".join(f"<span>{'<b>' + inline(meta[k]) + '</b>' if k == 'author' else inline(meta[k])}</span>"
                     for k in ("author", "affiliation") if meta.get(k))
    page = f"""<title>{html.escape(meta.get('title', 'Report'))}</title>
<meta name="claude:viewport-fit-sides" content="cover">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="{FONTS}">
<script>window.MathJax = {{ tex: {{ inlineMath: [["\\\\(", "\\\\)"]] }}, svg: {{ fontCache: "global" }} }};</script>
<script defer src="{MATHJAX}"></script>
<style>
{(HERE / 'style.css').read_text()}
</style>
<header class="mast">
  <div class="eyebrow">{inline(meta.get('eyebrow', ''))}</div>
  <h1>{inline(meta.get('title', ''))}</h1>
  <div class="byline">{byline}</div>
</header>
<nav class="toc" aria-label="Contents">
  {chr(10).join(toc)}
</nav>
<main>
{chr(10).join(body)}
<footer>{inline(meta.get('footer', ''))}</footer>
</main>
"""
    out.write_text(page)
    print(f"wrote {out.name} ({len(page) // 1000} kB)")


if __name__ == "__main__":
    build()
