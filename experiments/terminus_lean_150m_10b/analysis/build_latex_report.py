"""Build the archival LaTeX report from the verified Markdown analysis.

The Markdown file is the text source of truth, including all 23 exact prompt
transcripts. The generated TeX and PDF are committed review artifacts.
Run from anywhere with Pandoc and XeLaTeX installed; no GPU or cloud access.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path


HERE = Path(__file__).resolve().parent
SOURCE = HERE / "FINAL_RUN_ANALYSIS.md"
TEX = HERE / "FINAL_RUN_ANALYSIS.tex"
PDF = HERE / "FINAL_RUN_ANALYSIS.pdf"

PREAMBLE = r"""\documentclass[10pt,a4paper]{article}
\usepackage[margin=21mm,headheight=14pt]{geometry}
\usepackage{fontspec}
\setmainfont{Times New Roman}
\setsansfont{Avenir Next}
\setmonofont{Menlo}[Scale=0.83]
\newfontfamily\devanagarifont{Devanagari Sangam MN}
\newfontfamily\bengalifont{Kohinoor Bangla}
\newfontfamily\kannadafont{Kannada Sangam MN}
\newfontfamily\gujaratifont{Gujarati Sangam MN}
\newfontfamily\tamilfont{Tamil Sangam MN}
\newfontfamily\telugufont{Telugu Sangam MN}
\usepackage{graphicx}
\usepackage{booktabs,longtable,array,tabularx}
\usepackage{microtype}
\usepackage{xcolor}
\usepackage{caption}
\usepackage{fancyhdr}
\usepackage{hyperref}
\hypersetup{colorlinks=true,linkcolor=blue!40!black,urlcolor=blue!40!black,
  pdftitle={Terminus Lean 150M: 10B-token dense FP8 run report},
  pdfauthor={Terminus research}}
\definecolor{TerminusBlue}{HTML}{17324D}
\definecolor{TerminusGray}{HTML}{627080}
\captionsetup{font=small,labelfont=bf}
\setlength{\parskip}{0.35em}
\setlength{\parindent}{0pt}
\setlength{\emergencystretch}{3em}
\setlength{\tabcolsep}{4pt}
\renewcommand{\arraystretch}{1.12}
\renewcommand{\familydefault}{\rmdefault}
\newcommand{\tightlist}{\setlength{\itemsep}{0pt}\setlength{\parskip}{0pt}}
\pagestyle{fancy}
\fancyhf{}
\fancyhead[L]{\sffamily\footnotesize Terminus Lean 150M / Dense FP8}
\fancyhead[R]{\sffamily\footnotesize Final-run analysis}
\fancyfoot[C]{\sffamily\footnotesize\thepage}
\renewcommand{\headrulewidth}{0.3pt}
\setcounter{secnumdepth}{2}
\setcounter{tocdepth}{2}
\begin{document}
\begin{titlepage}
\vspace*{19mm}
{\sffamily\color{TerminusBlue}\LARGE Terminus Lean 150M\\[4mm]
\Huge Dense FP8 Training Run\\[3mm]
\Large Final Checkpoint and Stability Report\par}
\vspace{14mm}
{\large 10-billion-token experiment · 2 October 2026\par}
\vspace{12mm}
\rule{\linewidth}{0.7pt}
\vspace{5mm}
\begin{tabularx}{\linewidth}{@{}lX@{}}
Model & 146,423,680 trainable parameters; 25 layers; width 640\\
Training & 9,999,998,976 tokens; 25,432 updates; one H100 NVL\\
Final held-out loss & 2.4194 mixture-weighted; 2.7597 English\\
Numerical integrity & 0 non-finite incidents; 254 clipped updates\\
Artifacts & 257 validation points; 2,546 stability probes; 23 prompt completions
\end{tabularx}
\vspace{5mm}
\rule{\linewidth}{0.7pt}
\vspace{12mm}
\textbf{Assessment.} The run completed and held-out loss improved in every
tracked source. The checkpoint remains weak on several simple factual, arithmetic,
and code-completion probes. The first attention RMSNorm gain shrank markedly and
final-layer residual scale grew, despite finite gradients and improving loss.
Those scale signals warrant a controlled continuation before QAT selection.
\par\vfill
{\small\color{TerminusGray}Run ID: \texttt{lean\_dense\_10b\_20261001}\\
Training code: \texttt{c95a951d5436489b8b0902e96b327adcaa11b524}\\
Final checkpoint SHA-256: \path{3d116e88048c93a0403ab2920b1680b066a9f29cbc07fa8ae1c2c67f2144684b}\\
Full run files: verified in Azure Blob; the H100 VM is deallocated.}
\end{titlepage}
\tableofcontents
\clearpage
"""

FIGURE_CAPTIONS = {
    "loss_overall.png": "Mixture-weighted held-out loss over all 10B training tokens, with a late-run zoom.",
    "loss_domains.png": "Held-out losses for English, Indic, code, and math across training.",
    "loss_indic.png": "Held-out loss trajectories for the seven Indic languages.",
    "lr_gradient.png": "Learning-rate schedule, gradient norms, and clipping events.",
    "norm_residual_scales.png": "RMSNorm gain and sampled residual scales across training.",
    "stability_signals.png": "Sampled attention, FP8 quantization, and update-to-weight diagnostics.",
    "final_layer_scales.png": "Scale diagnostics by layer at the final checkpoint.",
}

FONTS = {
    "hindi": r"\devanagarifont",
    "marathi": r"\devanagarifont",
    "bengali": r"\bengalifont",
    "kannada": r"\kannadafont",
    "gujarati": r"\gujaratifont",
    "tamil": r"\tamilfont",
    "telugu": r"\telugufont",
    "code": r"\ttfamily",
}


def escape_tex(value: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "{": r"\{",
        "}": r"\}",
        "#": r"\#",
        "$": r"\$",
        "%": r"\%",
        "&": r"\&",
        "_": r"\_",
        "^": r"\textasciicircum{}",
        "~": r"\textasciitilde{}",
    }
    # The Markdown uses U+2420 only to make trailing ASCII spaces visible.
    value = value.replace("␠", "[space]")
    return "".join(replacements.get(char, char) for char in value)


def typeset_transcript(value: str) -> str:
    lines = []
    for line in value.split("\n"):
        leading = len(line) - len(line.lstrip(" "))
        content = escape_tex(line[leading:])
        if leading:
            content = rf"\hspace*{{{leading * 0.5}em}}" + content
        lines.append(content if content else r"\mbox{}")
    return r"\par\noindent ".join(lines)


def prompt_appendix(markdown: str) -> str:
    pattern = re.compile(
        r"### (\d+)\. (.*?) \((.*?)\)\s+Prompt:\s+```text\n(.*?)\n```"
        r"\s+Completion \((\d+) new tokens\):\s+```text\n(.*?)\n```",
        re.S,
    )
    matches = list(pattern.finditer(markdown))
    if len(matches) != 23 or [int(m[1]) for m in matches] != list(range(1, 24)):
        raise ValueError(f"Expected the 23 ordered transcript records, found {len(matches)}")
    chunks = [
        r"\appendix\section{All fixed prompt completions}",
        r"Each completion is the decoded 48-token greedy output from the SHA-verified "
        r"final checkpoint. The visible marker \texttt{[space]} represents a trailing "
        r"ASCII space; prompts and completions are separate. These fixed examples are "
        r"qualitative probes, not a task benchmark.",
    ]
    for match in matches:
        index, name, category, prompt, count, completion = match.groups()
        font = FONTS.get(category, "")
        title = f"{index}. {escape_tex(name)} ({escape_tex(category)})"
        chunks.extend(
            [
                r"\filbreak",
                rf"\subsection*{{{title}}}",
                rf"\addcontentsline{{toc}}{{subsection}}{{{title}}}",
                r"\textbf{Prompt}\par",
                rf"\begin{{quote}}\small\raggedright {font} {typeset_transcript(prompt)}\par\end{{quote}}",
                rf"\textbf{{Completion}} ({count} new tokens)\par",
                rf"\begin{{quote}}\small\raggedright {font} {typeset_transcript(completion)}\par\end{{quote}}",
            ]
        )
    return "\n".join(chunks)


def build_tex() -> None:
    markdown = SOURCE.read_text(encoding="utf-8")
    body, appendix = markdown.split("## Appendix: all fixed prompt completions", 1)
    body = re.sub(r"^# .*?\n\n", "", body, count=1)
    # The title page carries the run metadata, so avoid a duplicate paragraph.
    body = re.sub(r"^\*\*Run:\*\*.*?(?=\n## Verdict)", "", body, flags=re.S)
    body = body.replace("␠", "[space]")
    result = subprocess.run(
        ["pandoc", "-f", "gfm", "-t", "latex", "--no-highlight", "--shift-heading-level-by=-1"],
        input=body,
        text=True,
        capture_output=True,
        check=True,
        cwd=HERE,
    )
    latex = result.stdout
    latex = latex.replace(r"\def\LTcaptype{none} % do not increment counter", "")
    latex = re.sub(r"\\texttt\{([0-9a-f]{40,64})\}", r"\\path{\1}", latex)
    for source_path in ("dense/events.jsonl", "dense/stability.jsonl", "views/manifest.dense.json"):
        latex = latex.replace(r"\texttt{" + source_path + "}", r"\path{" + source_path + "}")

    def replace_figure(match: re.Match[str]) -> str:
        path = match[2]
        caption = FIGURE_CAPTIONS.get(Path(path).name)
        if caption is None:
            raise ValueError(f"Unexpected figure: {path}")
        return (
            "\\begin{figure}[htbp]\n\\centering\n"
            f"\\includegraphics[width=0.96\\linewidth]{{{path}}}\n"
            f"\\caption{{{caption}}}\n\\end{{figure}}"
        )

    latex, figure_count = re.subn(
        r"\\pandocbounded\{\\includegraphics\[keepaspectratio,alt=\{(.*?)\}\]\{(.*?)\}\}",
        replace_figure,
        latex,
    )
    if figure_count != 7:
        raise ValueError(f"Expected 7 figures, found {figure_count}")
    # Clear figures at section boundaries so captions remain close to their analyses.
    latex = latex.replace(r"\section{Fixed prompt test", r"\clearpage\section{Fixed prompt test")
    latex = latex.replace(r"\section{Time, throughput", r"\clearpage\section{Time, throughput")
    TEX.write_text(PREAMBLE + latex + "\n\\clearpage\n" + prompt_appendix(appendix) + "\n\\end{document}\n", encoding="utf-8")


def main() -> None:
    build_tex()
    for _ in range(2):
        subprocess.run(
            ["xelatex", "-interaction=nonstopmode", "-halt-on-error", "-file-line-error", TEX.name],
            cwd=HERE,
            check=True,
            stdout=subprocess.DEVNULL,
        )
    if not PDF.is_file():
        raise RuntimeError("XeLaTeX did not produce a PDF")
    print(f"Built {TEX} and {PDF}")


if __name__ == "__main__":
    main()
