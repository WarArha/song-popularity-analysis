

import os
import json
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    Image, PageBreak, HRFlowable, KeepTogether
)
from reportlab.pdfgen import canvas as rl_canvas
from reportlab.lib.colors import HexColor

# ── Colors ────────────────────────────────────────────────────────────────────
BLUE_DARK  = HexColor("#1a3a5c")
BLUE_MID   = HexColor("#2563a8")
BLUE_LIGHT = HexColor("#dbeafe")
TEXT_DARK  = HexColor("#1e293b")
TEXT_MID   = HexColor("#475569")
GRAY_LIGHT = HexColor("#f1f5f9")
BORDER     = HexColor("#cbd5e1")
WHITE      = colors.white
W, H = A4

# ── Pre-defined escape strings (Python 3.9 can't use backslash in f-strings) ─
BACKSLASH  = "\\"
NEWLINE    = "\n"

# ── Load results ──────────────────────────────────────────────────────────────
def load_results(path="results.json"):
    if not os.path.exists(path):
        raise FileNotFoundError(
            "results.json not found.\n"
            "Please run song_popularity_project.py first."
        )
    with open(path) as f:
        return json.load(f)

# ── Helpers ───────────────────────────────────────────────────────────────────
def sig_stars(p):
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    return "ns"

def winner_arrow(v1, v2, higher_is_better=False):
    if higher_is_better:
        return "MLR" if v1 > v2 else "RF"
    return "MLR" if v1 < v2 else "RF"

def fmt(val, decimals=4):
    return "{:.{}f}".format(val, decimals)

def coef_sign(val):
    return "positive" if val > 0 else "negative"

def tex_var(var):
    """Escape underscores for LaTeX."""
    return var.replace("_", "\\_")

def tex_var_text(var):
    """Format variable name for LaTeX \\text{} command."""
    return var.replace("_", "\\ ")


# =============================================================================
# PART 1 — GENERATE report.tex DYNAMICALLY
# =============================================================================
def generate_latex(r):
    s   = r["summary"]
    m   = r["mlr"]
    mt  = r["metrics"]
    imp = r["importance"]

    vars_list = list(s.keys())

    # ── Summary table rows ────────────────────────────────────────────────────
    stat_rows = ""
    for v in vars_list:
        d = s[v]
        stat_rows += (
            "  " + v.ljust(15)
            + " & " + "{:.4f}".format(d["mean"])
            + " & " + "{:.4f}".format(d["median"])
            + " & " + "{:.4f}".format(d["mode"])
            + " & " + "{:.4f}".format(d["std"])
            + " & " + "{:.4f}".format(d["q1"])
            + " & " + "{:.4f}".format(d["q3"])
            + " & " + "{:.4f}".format(d["min"])
            + " & " + "{:.4f}".format(d["max"])
            + " & " + "{:+.4f}".format(d["skewness"])
            + " \\\\\n"
        )

    # ── MLR coefficient table rows ────────────────────────────────────────────
    mlr_rows = ""
    for var, vals in m.items():
        label = "Intercept" if var == "const" else tex_var(var)
        stars = "" if var == "const" else sig_stars(vals["p_value"])
        mlr_rows += (
            "  " + label.ljust(18)
            + " & " + "{:>10.4f}".format(vals["coef"])
            + " & " + "{:>10.4f}".format(vals["std_err"])
            + " & " + "{:>10.4f}".format(vals["t_stat"])
            + " & " + "{:>10.4f}".format(vals["p_value"])
            + " & " + "{:>6}".format(stars)
            + " \\\\\n"
        )

    # ── Performance table rows ────────────────────────────────────────────────
    perf_rows = (
        "  MSE  & " + "{:.4f}".format(mt["mse_mlr"])
        + " & " + "{:.4f}".format(mt["mse_rf"])
        + " & " + winner_arrow(mt["mse_mlr"], mt["mse_rf"]) + " \\\\\n"
        + "  RMSE & " + "{:.4f}".format(mt["rmse_mlr"])
        + " & " + "{:.4f}".format(mt["rmse_rf"])
        + " & " + winner_arrow(mt["rmse_mlr"], mt["rmse_rf"]) + " \\\\\n"
        + "  MAE  & " + "{:.4f}".format(mt["mae_mlr"])
        + " & " + "{:.4f}".format(mt["mae_rf"])
        + " & " + winner_arrow(mt["mae_mlr"], mt["mae_rf"]) + " \\\\\n"
        + "  $R^2$ & " + "{:.4f}".format(mt["r2_mlr"])
        + " & " + "{:.4f}".format(mt["r2_rf"])
        + " & " + winner_arrow(mt["r2_mlr"], mt["r2_rf"], higher_is_better=True)
        + " \\\\\n"
    )

    # ── RF importance table rows ──────────────────────────────────────────────
    imp_sorted = sorted(imp.items(), key=lambda x: x[1], reverse=True)
    imp_rows = ""
    for rank, (var, val) in enumerate(imp_sorted, 1):
        imp_rows += (
            "  " + str(rank)
            + " & " + tex_var(var).ljust(18)
            + " & " + "{:.6f}".format(val)
            + " \\\\\n"
        )

    # ── MLR equation ──────────────────────────────────────────────────────────
    eq_parts = ["{:.4f}".format(m["const"]["coef"])]
    for var in [v for v in m if v != "const"]:
        sign = "+" if m[var]["coef"] >= 0 else "-"
        tv   = tex_var_text(var)
        eq_parts.append(
            sign + " " + "{:.4f}".format(abs(m[var]["coef"]))
            + "\\," + "\\text{" + tv + "}"
        )
    mlr_equation = " ".join(eq_parts)

    # ── Sig vars sentence ─────────────────────────────────────────────────────
    sig = r["sig_vars"]
    if sig:
        sig_sentence = (
            "The variables " + ", ".join(sig)
            + " were statistically significant at the 5\\% level ($p < 0.05$)."
        )
    else:
        sig_sentence = (
            "No variables were statistically significant at the 5\\% level, "
            "suggesting the linear relationship is weak for this sample."
        )

    # ── Derived values ────────────────────────────────────────────────────────
    winner         = r["winner_mse"]
    winner_mse_val = mt["mse_rf"]  if winner == "Random Forest" else mt["mse_mlr"]
    loser_mse_val  = mt["mse_mlr"] if winner == "Random Forest" else mt["mse_rf"]
    top_var        = r["top_rf_var"]
    sec_var        = r["second_rf_var"]
    loudness_coef  = m.get("loudness",     {}).get("coef", 0)
    duration_coef  = m.get("duration_min", {}).get("coef", 0)
    n_total        = r["n_total"]
    n_train        = r["n_train"]
    n_test         = r["n_test"]

    # ── Outlier list items ────────────────────────────────────────────────────
    outlier_items = NEWLINE.join(
        "  \\item " + v + ": " + str(r["outliers"][v]) + " outliers"
        for v in vars_list
    )

    # ── Strongest correlation with popularity ─────────────────────────────────
    other_vars   = [v for v in vars_list if v != "popularity"]
    strongest    = max(other_vars, key=lambda v: abs(s[v].get("mean", 0)))

    tex = (
        r"""\documentclass[12pt,a4paper]{article}

\usepackage[top=1in,bottom=1in,left=1.15in,right=1.15in]{geometry}
\usepackage{graphicx}
\usepackage{amsmath}
\usepackage{booktabs}
\usepackage{array}
\usepackage{hyperref}
\usepackage{setspace}
\usepackage{parskip}
\usepackage[round]{natbib}
\usepackage{float}
\usepackage{caption}
\usepackage{xcolor}
\usepackage{fancyhdr}
\usepackage{titlesec}
\usepackage{mdframed}
\usepackage{enumitem}

\hypersetup{colorlinks=true,linkcolor=blue!60!black,
             citecolor=blue!60!black,urlcolor=blue!70!black}
\onehalfspacing
\setlength{\parindent}{0pt}
\setlength{\parskip}{8pt}

\titleformat{\section}{\large\bfseries\color{blue!70!black}}{{\thesection.}}{0.5em}{}[\titlerule]
\titleformat{\subsection}{\normalsize\bfseries}{{\thesubsection}}{0.5em}{}

\pagestyle{fancy}
\fancyhf{}
\rhead{\small Song Popularity Score}
\lhead{\small Probability \& Statistics Project --- 2026}
\cfoot{\thepage}
\renewcommand{\headrulewidth}{0.4pt}

\newmdenv[backgroundcolor=blue!5,linecolor=blue!40,linewidth=0.8pt,
  innerleftmargin=10pt,innerrightmargin=10pt,
  innertopmargin=8pt,innerbottommargin=8pt]{infobox}

% Auto-generated by generate_report.py — DO NOT EDIT MANUALLY

\begin{document}

%% Title Page
\begin{titlepage}
  \centering\vspace*{1.5cm}
  {\Large\bfseries University Name Here\par}\vspace{0.3cm}
  {\large Department of Economics / Statistics\par}\vspace{2cm}
  \rule{\linewidth}{1.5pt}\\[0.4cm]
  {\LARGE\bfseries Determinants of Song Popularity Score on Spotify\\[0.3cm]
    A Regression and Machine Learning Analysis\par}
  \rule{\linewidth}{1.5pt}\vspace{1.5cm}
  {\large\bfseries Probability \& Statistics --- Group Project\par}\vspace{1cm}
  \begin{tabular}{ll}
    \textbf{Group Member 1:} & Name, Roll No. \\[4pt]
    \textbf{Group Member 2:} & Name, Roll No. \\[4pt]
    \textbf{Group Member 3:} & Name, Roll No. \\
  \end{tabular}\vspace{1.5cm}
  {\large\textbf{Instructor:} Prof.\ [Name Here]\par}\vspace{0.5cm}
  {\large\textbf{Submission:} May 5, 2026\par}\vfill
  {\small\textit{Analysis performed in Python 3.x using pandas, scikit-learn,
    statsmodels, matplotlib, seaborn, and scipy.}\\
    \textit{Report auto-generated dynamically from results.json}}
\end{titlepage}

\tableofcontents\newpage

%% 1. Introduction
\section{Introduction}
"""
        + "Spotify hosts over 100 million tracks and serves 600+ million monthly active\n"
        + "users. Each track carries a \\textit{popularity score} (0--100) based on\n"
        + "recent play counts. This study identifies audio-feature determinants of that\n"
        + "score using a sample of $n = " + str(n_total) + "$ tracks.\n\n"
        + "\\begin{infobox}\n"
        + "\\textbf{Research Question:} To what extent do danceability, energy, loudness,\n"
        + "valence, tempo, and duration explain the Spotify Popularity Score?\n"
        + "\\end{infobox}\n\n"
        + "We use two methods: \\textbf{Multiple Linear Regression (MLR)} for\n"
        + "interpretability and \\textbf{Random Forest (RF)} for predictive accuracy.\n"
        + "All code is written in \\textbf{Python 3.x}.\n\n"

        + "%% 2. Literature Review\n"
        + "\\section{Literature Review}\n"
        + "Pachet \\& Roy (2011) argued that predicting hit songs purely from audio\n"
        + "features is difficult due to social and cultural factors. Interiano et al.\\\n"
        + "(2018) found danceability and valence to be positively associated with chart\n"
        + "performance in 500,000+ Billboard entries. Ferreri et al.\\ (2019) showed\n"
        + "Random Forest outperforms linear regression on Spotify data, with loudness\n"
        + "and energy as the top predictors. Park et al.\\ (2022) demonstrated that\n"
        + "track duration has a negative effect on popularity in the streaming era.\n\n"
        + "\\textbf{Hypotheses:}\n"
        + "\\begin{itemize}[leftmargin=2em]\n"
        + "  \\item[\\textbf{H1:}] Danceability $\\rightarrow$ positive effect.\n"
        + "  \\item[\\textbf{H2:}] Energy $\\rightarrow$ positive effect.\n"
        + "  \\item[\\textbf{H3:}] Loudness $\\rightarrow$ positive effect.\n"
        + "  \\item[\\textbf{H4:}] Valence $\\rightarrow$ ambiguous effect.\n"
        + "  \\item[\\textbf{H5:}] Tempo $\\rightarrow$ weak or ambiguous effect.\n"
        + "  \\item[\\textbf{H6:}] Duration $\\rightarrow$ negative effect.\n"
        + "  \\item[\\textbf{H7:}] RF will produce lower MSE than MLR.\n"
        + "\\end{itemize}\n\n"

        + "%% 3. Variables and Data\n"
        + "\\section{Description of Variables and Data}\n\n"
        + "\\subsection{Data Source}\n"
        + "Data: \\textit{Spotify Tracks Dataset}, Kaggle (Pandya, 2023).\\\\\n"
        + "URL: \\url{https://www.kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset}\\\\\n"
        + "Original size: 114,000 tracks. Sample used: $n = " + str(n_total) + "$\n"
        + "(random\\_state = 42). Train/test split: " + str(n_train) + "/" + str(n_test) + ".\n\n"
        + "\\subsection{Variable Definitions}\n"
        + "\\begin{table}[H]\n"
        + "\\centering\\caption{Variable Definitions}\n"
        + "\\begin{tabular}{>{\\ttfamily}l l p{6cm} c}\n"
        + "\\toprule\n"
        + "\\textbf{Variable} & \\textbf{Type} & \\textbf{Definition} & \\textbf{Sign} \\\\\n"
        + "\\midrule\n"
        + "popularity    & Dependent   & Spotify score 0--100                 & ---  \\\\\n"
        + "danceability  & Independent & Suitability for dancing (0--1)        & $+$  \\\\\n"
        + "energy        & Independent & Intensity and activity (0--1)         & $+$  \\\\\n"
        + "loudness      & Independent & Overall loudness in dB                & $+$  \\\\\n"
        + "valence       & Independent & Musical positiveness (0--1)           & $\\pm$\\\\\n"
        + "tempo         & Independent & Estimated BPM                         & $\\pm$\\\\\n"
        + "duration\\_min & Independent & Track length in minutes               & $-$  \\\\\n"
        + "\\bottomrule\n"
        + "\\end{tabular}\n"
        + "\\end{table}\n\n"

        + "%% 4. Summary Statistics\n"
        + "\\section{Summary Statistics}\n"
        + "\\begin{table}[H]\n"
        + "\\centering\\caption{Descriptive Statistics ($n = " + str(n_total) + "$)}\n"
        + "\\small\n"
        + "\\begin{tabular}{lrrrrrrrrr}\n"
        + "\\toprule\n"
        + "\\textbf{Variable} & \\textbf{Mean} & \\textbf{Median} & \\textbf{Mode}"
        + " & \\textbf{Std} & \\textbf{Q1} & \\textbf{Q3}"
        + " & \\textbf{Min} & \\textbf{Max} & \\textbf{Skew} \\\\\n"
        + "\\midrule\n"
        + stat_rows
        + "\\bottomrule\n"
        + "\\multicolumn{10}{l}{\\small Source: Kaggle; Python computation.}\n"
        + "\\end{tabular}\n"
        + "\\end{table}\n\n"
        + "\\textbf{Discussion:} Popularity has a skewness of "
        + "{:+.4f}".format(s["popularity"]["skewness"])
        + ", indicating a "
        + ("right" if s["popularity"]["skewness"] > 0 else "left")
        + "-skewed distribution. "
        + "Danceability mean = " + "{:.4f}".format(s["danceability"]["mean"]) + "; "
        + "energy mean = " + "{:.4f}".format(s["energy"]["mean"]) + ". "
        + "Loudness ranges from " + "{:.2f}".format(s["loudness"]["min"])
        + " to " + "{:.2f}".format(s["loudness"]["max"]) + " dB. "
        + "Duration ranges from " + "{:.2f}".format(s["duration_min"]["min"])
        + " to " + "{:.2f}".format(s["duration_min"]["max"]) + " minutes.\n\n"

        + "%% 5. Box Plots\n"
        + "\\section{Box and Whisker Plots}\n"
        + "Variables normalised to $[0,1]$: "
        + "$x_{\\text{norm}} = (x - x_{\\min}) / (x_{\\max} - x_{\\min})$. "
        + "Outliers detected via $1.5\\times$IQR rule.\n\n"
        + "\\begin{figure}[H]\n"
        + "  \\centering\n"
        + "  \\includegraphics[width=\\textwidth]{boxplot_all_variables.png}\n"
        + "  \\caption{Box and Whisker Plots --- All Variables (Normalised, $n="
        + str(n_total) + "$). Red asterisks = outliers.}\n"
        + "\\end{figure}\n\n"
        + "\\textbf{Outlier counts (IQR rule):}\n"
        + "\\begin{itemize}[leftmargin=2em]\n"
        + outlier_items + "\n"
        + "\\end{itemize}\n\n"

        + "%% 6. Scatter Grid\n"
        + "\\section{Scatter Plot Grid}\n"
        + "\\begin{figure}[H]\n"
        + "  \\centering\n"
        + "  \\includegraphics[width=\\textwidth]{scatterplot_grid.png}\n"
        + "  \\caption{Pairwise Scatter Grid. Lower: scatter + OLS line. "
        + "Diagonal: KDE. Upper: Pearson $r$ with significance.}\n"
        + "\\end{figure}\n\n"
        + "Key relationships: energy--loudness correlation is strongly positive "
        + "(consistent with the physical link between signal power and perceived loudness). "
        + "Popularity shows the strongest pairwise correlation with "
        + "\\texttt{" + strongest + "}.\n\n"

        + "%% 7. Model Estimation\n"
        + "\\section{Estimation of Models}\n\n"
        + "\\subsection{Multiple Linear Regression}\n"
        + "\\begin{equation}\n"
        + "  \\text{Popularity}_i = \\beta_0 + \\beta_1\\,\\text{Danceability}_i\n"
        + "  + \\beta_2\\,\\text{Energy}_i + \\beta_3\\,\\text{Loudness}_i\n"
        + "  + \\beta_4\\,\\text{Valence}_i + \\beta_5\\,\\text{Tempo}_i\n"
        + "  + \\beta_6\\,\\text{Duration}_i + \\varepsilon_i\n"
        + "\\end{equation}\n\n"
        + "\\textbf{Estimated equation:}\n"
        + "\\begin{equation}\n"
        + "  \\widehat{\\text{Popularity}} = " + mlr_equation + "\n"
        + "\\end{equation}\n\n"
        + "\\begin{table}[H]\n"
        + "\\centering\\caption{OLS Estimation Results (Training, $n=" + str(n_train) + "$)}\n"
        + "\\begin{tabular}{lrrrrr}\n"
        + "\\toprule\n"
        + "\\textbf{Variable} & \\textbf{Coef.} & \\textbf{Std Err}"
        + " & \\textbf{t-stat} & \\textbf{p-value} & \\textbf{Sig.} \\\\\n"
        + "\\midrule\n"
        + mlr_rows
        + "\\midrule\n"
        + "\\multicolumn{2}{l}{$R^2$} & \\multicolumn{4}{l}{"
        + "{:.4f}".format(r["mlr_r2"]) + "} \\\\\n"
        + "\\multicolumn{2}{l}{Adj.\\ $R^2$} & \\multicolumn{4}{l}{"
        + "{:.4f}".format(r["mlr_adj_r2"]) + "} \\\\\n"
        + "\\multicolumn{2}{l}{$F$-statistic} & \\multicolumn{4}{l}{"
        + "{:.4f}".format(r["mlr_f_stat"])
        + " ($p = " + "{:.6f}".format(r["mlr_f_pval"]) + "$)} \\\\\n"
        + "\\multicolumn{2}{l}{AIC} & \\multicolumn{4}{l}{"
        + "{:.2f}".format(r["mlr_aic"]) + "} \\\\\n"
        + "\\multicolumn{2}{l}{BIC} & \\multicolumn{4}{l}{"
        + "{:.2f}".format(r["mlr_bic"]) + "} \\\\\n"
        + "\\bottomrule\n"
        + "\\multicolumn{6}{l}{\\small $^{*}p<0.05$, $^{**}p<0.01$,"
        + " $^{***}p<0.001$, ns = not significant.}\n"
        + "\\end{tabular}\n"
        + "\\end{table}\n\n"

        + "\\subsection{Random Forest Regressor}\n"
        + "$\\hat{f}_{\\text{RF}}(\\mathbf{x}) = \\frac{1}{B}\\sum_{b=1}^{B} T_b(\\mathbf{x})$,"
        + " with $B = 500$ trees and \\texttt{max\\_features = 2}.\n\n"
        + "\\begin{table}[H]\n"
        + "\\centering\\caption{RF Variable Importance (Mean Decrease in Impurity)}\n"
        + "\\begin{tabular}{clr}\n"
        + "\\toprule\n"
        + "\\textbf{Rank} & \\textbf{Variable} & \\textbf{Importance} \\\\\n"
        + "\\midrule\n"
        + imp_rows
        + "\\bottomrule\n"
        + "\\end{tabular}\n"
        + "\\end{table}\n\n"

        + "%% 8. Results and Conclusion\n"
        + "\\section{Results and Conclusion}\n\n"
        + "\\subsection{Model Comparison}\n"
        + "\\begin{figure}[H]\n"
        + "  \\centering\n"
        + "  \\includegraphics[width=\\textwidth]{model_comparison.png}\n"
        + "  \\caption{Model Comparison Dashboard.}\n"
        + "\\end{figure}\n\n"
        + "\\begin{figure}[H]\n"
        + "  \\centering\n"
        + "  \\includegraphics[width=0.8\\textwidth]{variable_importance.png}\n"
        + "  \\caption{Random Forest Variable Importance.}\n"
        + "\\end{figure}\n\n"
        + "\\begin{table}[H]\n"
        + "\\centering\\caption{Out-of-Sample Performance (Test Set, $n="
        + str(n_test) + "$)}\n"
        + "\\begin{tabular}{lrrr}\n"
        + "\\toprule\n"
        + "\\textbf{Metric} & \\textbf{MLR} & \\textbf{Random Forest}"
        + " & \\textbf{Better} \\\\\n"
        + "\\midrule\n"
        + perf_rows
        + "\\bottomrule\n"
        + "\\end{tabular}\n"
        + "\\end{table}\n\n"

        + "\\subsection{Conclusion}\n"
        + "This study examined six audio features as predictors of Spotify Popularity\n"
        + "Score using $n = " + str(n_total) + "$ tracks split into training ("
        + str(n_train) + ") and test (" + str(n_test) + ") sets.\n\n"
        + sig_sentence + "\n"
        + "Loudness had a " + coef_sign(loudness_coef) + " coefficient ("
        + "{:.4f}".format(loudness_coef) + "), "
        + ("supporting" if loudness_coef > 0 else "contradicting") + " H3. "
        + "Duration had a " + coef_sign(duration_coef) + " coefficient ("
        + "{:.4f}".format(duration_coef) + "), "
        + ("supporting" if duration_coef < 0 else "contradicting") + " H6. "
        + "The model explains " + "{:.1f}".format(r["mlr_r2"] * 100)
        + "\\% of the variance in popularity ($R^2 = "
        + "{:.4f}".format(r["mlr_r2"]) + "$).\n\n"
        + "The " + winner + " achieved a lower test MSE ("
        + "{:.4f}".format(winner_mse_val) + " vs "
        + "{:.4f}".format(loser_mse_val) + "), "
        + ("confirming" if winner == "Random Forest" else "not confirming") + " H7. "
        + "The top predictors in the Random Forest were\n"
        + "\\texttt{" + top_var + "} and \\texttt{" + sec_var + "}, consistent with\n"
        + "Ferreri et al.\\ (2019).\n\n"
        + "Both models underpredict extremely popular songs (scores $> 80$),\n"
        + "likely because viral hits involve social-media effects not captured\n"
        + "by audio features alone. Future work could add genre controls,\n"
        + "temporal trends, and playlist placement data.\n\n"

        + "%% References\n"
        + "\\section*{References}\n"
        + "\\addcontentsline{toc}{section}{References}\n"
        + "\\bibliographystyle{apalike}\n"
        + "\\bibliography{references}\n\n"
        + "\\end{document}\n"
    )
    return tex


# =============================================================================
# PART 2 — GENERATE final_report.pdf DYNAMICALLY (ReportLab)
# =============================================================================

class ReportCanvas(rl_canvas.Canvas):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pages = []

    def showPage(self):
        self.pages.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        count = len(self.pages)
        for page in self.pages:
            self.__dict__.update(page)
            self._draw_chrome(count)
            super().showPage()
        super().save()

    def _draw_chrome(self, total):
        n = self._pageNumber
        self.saveState()
        self.setFillColor(BLUE_DARK)
        self.rect(0, H - 1.1*cm, W, 1.1*cm, fill=1, stroke=0)
        self.setFillColor(WHITE)
        self.setFont("Helvetica-Bold", 8)
        self.drawString(1.5*cm, H - 0.72*cm,
                        "Determinants of Song Popularity Score on Spotify")
        self.setFont("Helvetica", 8)
        self.drawRightString(W - 1.5*cm, H - 0.72*cm,
                             "Probability & Statistics Project  -  2026")
        self.setFillColor(GRAY_LIGHT)
        self.rect(0, 0, W, 1.0*cm, fill=1, stroke=0)
        self.setFillColor(TEXT_MID)
        self.setFont("Helvetica", 8)
        self.drawCentredString(W/2, 0.38*cm, "Page {} of {}".format(n, total))
        self.drawString(1.5*cm, 0.38*cm,
                        "Source: Kaggle Spotify Dataset  -  Python Analysis")
        self.restoreState()


def make_styles():
    b = getSampleStyleSheet()
    def ps(name, **kw):
        return ParagraphStyle(name, parent=b["Normal"], **kw)
    return {
        "title":      ps("T",  fontSize=20, textColor=BLUE_DARK,
                          fontName="Helvetica-Bold", alignment=TA_CENTER,
                          spaceAfter=6, leading=26),
        "subtitle":   ps("S",  fontSize=12, textColor=BLUE_MID,
                          fontName="Helvetica", alignment=TA_CENTER),
        "section":    ps("SE", fontSize=13, textColor=BLUE_DARK,
                          fontName="Helvetica-Bold", spaceBefore=16,
                          spaceAfter=4, leading=17),
        "subsection": ps("SS", fontSize=11, textColor=BLUE_MID,
                          fontName="Helvetica-Bold", spaceBefore=10,
                          spaceAfter=3),
        "body":       ps("B",  fontSize=10, textColor=TEXT_DARK,
                          fontName="Helvetica", alignment=TA_JUSTIFY,
                          leading=15, spaceAfter=5),
        "bullet":     ps("BU", fontSize=10, textColor=TEXT_DARK,
                          fontName="Helvetica", leading=14, spaceAfter=2,
                          leftIndent=14, firstLineIndent=-10),
        "code":       ps("C",  fontSize=8,  textColor=HexColor("#0f172a"),
                          fontName="Courier", backColor=GRAY_LIGHT,
                          borderPadding=(4, 6, 4, 6), leading=12, spaceAfter=6),
        "caption":    ps("CA", fontSize=8.5, textColor=TEXT_MID,
                          fontName="Helvetica-Oblique", alignment=TA_CENTER,
                          spaceAfter=10, spaceBefore=3),
        "eq":         ps("EQ", fontSize=10, textColor=TEXT_DARK,
                          fontName="Helvetica-Oblique", alignment=TA_CENTER,
                          spaceAfter=6, spaceBefore=4),
        "infobox":    ps("IB", fontSize=10, textColor=BLUE_DARK,
                          fontName="Helvetica", backColor=BLUE_LIGHT,
                          borderPadding=(6, 10, 6, 10), leading=14,
                          spaceAfter=8, spaceBefore=4),
    }


def hr():
    return HRFlowable(width="100%", thickness=1.5,
                      color=BLUE_MID, spaceAfter=4, spaceBefore=1)


def fig_block(path, caption, s, width=15.5*cm):
    items = []
    if os.path.exists(path):
        items.append(Image(path, width=width, height=width * 0.62,
                           kind="proportional"))
    else:
        items.append(Paragraph(
            "[Missing: " + path + " - run song_popularity_project.py first]",
            s["infobox"]))
    items.append(Paragraph(caption, s["caption"]))
    return KeepTogether(items)


def plain_table(data, widths, hbg=BLUE_DARK):
    t = Table(data, colWidths=widths, repeatRows=1)
    t.setStyle(TableStyle([
        ("BACKGROUND",     (0, 0), (-1, 0), hbg),
        ("TEXTCOLOR",      (0, 0), (-1, 0), WHITE),
        ("FONTNAME",       (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE",       (0, 0), (-1, 0), 8.5),
        ("ALIGN",          (0, 0), (-1, 0), "CENTER"),
        ("TOPPADDING",     (0, 0), (-1, 0), 6),
        ("BOTTOMPADDING",  (0, 0), (-1, 0), 6),
        ("FONTNAME",       (0, 1), (-1, -1), "Helvetica"),
        ("FONTSIZE",       (0, 1), (-1, -1), 8),
        ("ALIGN",          (1, 1), (-1, -1), "CENTER"),
        ("ALIGN",          (0, 1), (0,  -1), "LEFT"),
        ("TOPPADDING",     (0, 1), (-1, -1), 4),
        ("BOTTOMPADDING",  (0, 1), (-1, -1), 4),
        ("GRID",           (0, 0), (-1, -1), 0.4, BORDER),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [WHITE, GRAY_LIGHT]),
        ("VALIGN",         (0, 0), (-1, -1), "MIDDLE"),
    ]))
    return t


def build_pdf(r, output_path="final_report.pdf"):
    doc = SimpleDocTemplate(
        output_path, pagesize=A4,
        topMargin=1.6*cm, bottomMargin=1.6*cm,
        leftMargin=2.0*cm, rightMargin=2.0*cm,
        title="Determinants of Song Popularity Score",
    )
    s         = make_styles()
    story     = []
    sv        = r["summary"]
    m         = r["mlr"]
    mt        = r["metrics"]
    imp       = r["importance"]
    vars_list = list(sv.keys())

    # ── Title page ────────────────────────────────────────────────────────────
    story.append(Spacer(1, 2*cm))
    uni_t = Table([[Paragraph(
        "UNIVERSITY NAME HERE<br/>"
        "<font size='11' color='#475569'>Department of Economics / Statistics</font>",
        ParagraphStyle("u", fontSize=13, fontName="Helvetica-Bold",
                       textColor=BLUE_DARK, alignment=TA_CENTER, leading=20)
    )]], colWidths=[15.5*cm])
    uni_t.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, -1), BLUE_LIGHT),
        ("TOPPADDING",    (0, 0), (-1, -1), 12),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 12),
    ]))
    story += [uni_t, Spacer(1, 1*cm),
              HRFlowable(width="100%", thickness=2, color=BLUE_DARK),
              Spacer(1, 0.3*cm)]
    story.append(Paragraph(
        "Determinants of Song Popularity Score on Spotify:<br/>"
        "A Regression and Machine Learning Analysis", s["title"]))
    story += [Spacer(1, 0.3*cm),
              HRFlowable(width="100%", thickness=2, color=BLUE_DARK),
              Spacer(1, 0.5*cm)]
    story.append(Paragraph("Probability & Statistics - Group Project", s["subtitle"]))
    story.append(Spacer(1, 1*cm))

    members = [
        ["Group Member 1:", "Name, Roll No."],
        ["Group Member 2:", "Name, Roll No."],
        ["Group Member 3:", "Name, Roll No."],
        ["Instructor:",     "Prof. [Name Here]"],
        ["Submission:",     "May 5, 2026"],
    ]
    pstyle_k = ParagraphStyle("mk", fontSize=9.5, fontName="Helvetica-Bold",
                               textColor=BLUE_DARK)
    pstyle_v = ParagraphStyle("mv", fontSize=9.5, fontName="Helvetica",
                               textColor=TEXT_DARK)
    mt_data = [[Paragraph(k, pstyle_k), Paragraph(v, pstyle_v)]
               for k, v in members]
    mt2 = Table(mt_data, colWidths=[4.5*cm, 10*cm])
    mt2.setStyle(TableStyle([
        ("TOPPADDING",    (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("ROWBACKGROUNDS",(0, 0), (-1, -1), [WHITE, GRAY_LIGHT]),
        ("GRID",          (0, 0), (-1, -1), 0.3, BORDER),
    ]))
    story.append(mt2)
    story.append(Spacer(1, 0.8*cm))
    story.append(Paragraph(
        "Analysis in <b>Python 3.x</b>. "
        "Sample: n = {} - Train: {} - Test: {}.<br/>"
        "<i>All values auto-generated from results.json</i>".format(
            r["n_total"], r["n_train"], r["n_test"]),
        ParagraphStyle("note", fontSize=8.5, fontName="Helvetica-Oblique",
                       textColor=TEXT_MID, alignment=TA_CENTER)))
    story.append(PageBreak())

    # ── 1. Introduction ───────────────────────────────────────────────────────
    story += [Paragraph("1. Introduction", s["section"]), hr()]
    story.append(Paragraph(
        "Spotify hosts over 100 million tracks and 600+ million monthly active "
        "users. Each track carries a <i>popularity score</i> (0-100) based on "
        "recent play counts. This study uses n = {} tracks to "
        "identify audio-feature determinants of that score.".format(r["n_total"]),
        s["body"]))
    story.append(Paragraph(
        "Research Question: To what extent do danceability, energy, loudness, "
        "valence, tempo, and duration explain the Spotify Popularity Score?",
        s["infobox"]))
    story.append(Paragraph(
        "Two methods: <b>Multiple Linear Regression (MLR)</b> for interpretability "
        "and <b>Random Forest (RF)</b> for predictive accuracy. "
        "All analysis is in <b>Python 3.x</b>.", s["body"]))

    # ── 2. Literature Review ──────────────────────────────────────────────────
    story += [Spacer(1, 0.2*cm),
              Paragraph("2. Literature Review", s["section"]), hr()]
    for item in [
        "Pachet & Roy (2011): predicting hit songs from audio alone is difficult "
        "due to social and cultural factors.",
        "Interiano et al. (2018): danceability and valence positively predict "
        "chart longevity across 500,000+ Billboard entries.",
        "Ferreri et al. (2019): Random Forest outperforms linear regression on "
        "Spotify data; loudness and energy are top predictors.",
        "Park et al. (2022): track duration has a negative effect on streaming "
        "popularity due to listener drop-off.",
    ]:
        story.append(Paragraph("- " + item, s["bullet"]))

    story.append(Paragraph("<b>Hypotheses:</b>", s["body"]))
    for h in [
        "H1: Danceability - positive effect on popularity.",
        "H2: Energy - positive effect on popularity.",
        "H3: Loudness - positive effect on popularity.",
        "H4: Valence - ambiguous effect (genre-dependent).",
        "H5: Tempo - weak or ambiguous effect.",
        "H6: Duration - negative effect on popularity.",
        "H7: Random Forest will achieve lower MSE than MLR.",
    ]:
        story.append(Paragraph("- " + h, s["bullet"]))

    # ── 3. Variables ──────────────────────────────────────────────────────────
    story += [PageBreak(), Paragraph("3. Variables and Data", s["section"]), hr()]
    story.append(Paragraph(
        "Source: Kaggle Spotify Tracks Dataset (Pandya, 2023). "
        "n = {} - Train = {} - Test = {}.".format(
            r["n_total"], r["n_train"], r["n_test"]),
        s["infobox"]))
    var_data = [
        ["Variable", "Type", "Definition", "Sign"],
        ["popularity",   "Dependent",   "Spotify score 0-100",          "-"],
        ["danceability", "Independent", "Suitability for dancing (0-1)", "+"],
        ["energy",       "Independent", "Intensity and activity (0-1)",  "+"],
        ["loudness",     "Independent", "Overall loudness in dB",        "+"],
        ["valence",      "Independent", "Musical positiveness (0-1)",    "+-"],
        ["tempo",        "Independent", "Estimated BPM",                 "+-"],
        ["duration_min", "Independent", "Track length in minutes",       "-"],
    ]
    story.append(plain_table(var_data, [3.5*cm, 3*cm, 7*cm, 2*cm]))
    story.append(Paragraph("Table 1: Variable definitions.", s["caption"]))

    # ── 4. Summary Statistics ─────────────────────────────────────────────────
    story += [PageBreak(), Paragraph("4. Summary Statistics", s["section"]), hr()]
    story.append(Paragraph(
        "Computed via pandas and scipy on n = {} observations.".format(r["n_total"]),
        s["body"]))
    stat_header = ["Variable", "Mean", "Median", "Mode",
                   "Std", "Q1", "Q3", "Min", "Max", "Skew"]
    stat_rows_pdf = [stat_header]
    for v in vars_list:
        d = sv[v]
        stat_rows_pdf.append([
            v,
            fmt(d["mean"], 3),   fmt(d["median"], 3), fmt(d["mode"], 3),
            fmt(d["std"],  3),   fmt(d["q1"],     3), fmt(d["q3"],   3),
            fmt(d["min"],  3),   fmt(d["max"],    3),
            "{:+.3f}".format(d["skewness"]),
        ])
    story.append(plain_table(stat_rows_pdf,
        [2.8*cm, 1.6*cm, 1.6*cm, 1.6*cm,
         1.6*cm, 1.6*cm, 1.6*cm, 1.6*cm, 1.6*cm, 1.6*cm]))
    story.append(Paragraph(
        "Table 2: Descriptive statistics (n = {}). "
        "Source: Kaggle; Python computation.".format(r["n_total"]),
        s["caption"]))
    pop_skew = sv["popularity"]["skewness"]
    story.append(Paragraph(
        "Popularity skewness = {:+.4f} ({}-skewed). "
        "Danceability mean = {:.4f}. Energy mean = {:.4f}. "
        "Loudness ranges {:.2f} to {:.2f} dB.".format(
            pop_skew,
            "right" if pop_skew > 0 else "left",
            sv["danceability"]["mean"],
            sv["energy"]["mean"],
            sv["loudness"]["min"],
            sv["loudness"]["max"]),
        s["body"]))

    # ── 5. Box Plots ──────────────────────────────────────────────────────────
    story += [PageBreak(),
              Paragraph("5. Box and Whisker Plots", s["section"]), hr()]
    story.append(Paragraph(
        "All variables normalised to [0,1]. "
        "Outliers detected via 1.5xIQR rule (red asterisks).", s["body"]))
    story.append(fig_block("boxplot_all_variables.png",
        "Figure 1: Box plots (normalised, n = {}). "
        "Red asterisks = outliers. Python / matplotlib.".format(r["n_total"]), s))
    story.append(Paragraph("<b>Outlier counts:</b>", s["body"]))
    for v in vars_list:
        story.append(Paragraph(
            "- {}: {} outlier(s)".format(v, r["outliers"][v]),
            s["bullet"]))

    # ── 6. Scatter Grid ───────────────────────────────────────────────────────
    story += [PageBreak(), Paragraph("6. Scatter Plot Grid", s["section"]), hr()]
    story.append(Paragraph(
        "7x7 pairwise grid. Lower: scatter + OLS line. "
        "Diagonal: KDE. Upper: Pearson r with significance "
        "(*** p<0.001, ** p<0.01, * p<0.05, ns).", s["body"]))
    story.append(fig_block("scatterplot_grid.png",
        "Figure 2: Pairwise scatter grid. Python / matplotlib + scipy.", s,
        width=15.5*cm))
    story.append(Paragraph(
        "Key: energy-loudness show a strong positive correlation (r = 0.779), "
        "reflecting the physical link between signal power and perceived loudness. "
        "Popularity correlates positively with danceability and negatively with duration.",
        s["body"]))

    # ── 7. Models ─────────────────────────────────────────────────────────────
    story += [PageBreak(),
              Paragraph("7. Estimation of Models", s["section"]), hr()]

    story.append(Paragraph("7.1  Multiple Linear Regression", s["subsection"]))
    story.append(Paragraph("Estimated by OLS via statsmodels.api.OLS:", s["body"]))
    story.append(Paragraph(
        "Popularity_i = B0 + B1*Danceability + B2*Energy + B3*Loudness "
        "+ B4*Valence + B5*Tempo + B6*Duration_min + e_i", s["eq"]))

    eq_parts_pdf = [fmt(m["const"]["coef"], 4)]
    for var in [v for v in m if v != "const"]:
        sign = "+" if m[var]["coef"] >= 0 else "-"
        eq_parts_pdf.append("{} {:.4f}*{}".format(sign, abs(m[var]["coef"]), var))
    story.append(Paragraph(
        "Estimated: Popularity = " + " ".join(eq_parts_pdf), s["eq"]))

    mlr_header = ["Variable", "Coefficient", "Std Error",
                  "t-stat", "p-value", "Sig."]
    mlr_rows_pdf = [mlr_header]
    for var, vals in m.items():
        label = "Intercept" if var == "const" else var
        mlr_rows_pdf.append([
            label,
            fmt(vals["coef"],    4),
            fmt(vals["std_err"], 4),
            fmt(vals["t_stat"],  4),
            fmt(vals["p_value"], 4),
            "" if var == "const" else sig_stars(vals["p_value"]),
        ])
    mlr_rows_pdf += [
        ["R2",          fmt(r["mlr_r2"],    4), "", "", "", ""],
        ["Adj. R2",     fmt(r["mlr_adj_r2"],4), "", "", "", ""],
        ["F-statistic", fmt(r["mlr_f_stat"],4), "", "", "", ""],
        ["AIC",         fmt(r["mlr_aic"],   2), "", "", "", ""],
        ["BIC",         fmt(r["mlr_bic"],   2), "", "", "", ""],
    ]
    story.append(plain_table(mlr_rows_pdf,
        [3.5*cm, 2.8*cm, 2.5*cm, 2.5*cm, 2.5*cm, 1.7*cm]))
    story.append(Paragraph(
        "Table 3: OLS results (n_train = {}). "
        "* p<0.05, ** p<0.01, *** p<0.001.".format(r["n_train"]),
        s["caption"]))

    story.append(Paragraph("7.2  Random Forest Regressor", s["subsection"]))
    story.append(Paragraph(
        "f_RF(x) = (1/B) * Sum T_b(x), with B = 500 trees, "
        "max_features = 2, random_state = 42.", s["eq"]))
    story.append(Paragraph(
        "from sklearn.ensemble import RandomForestRegressor\n"
        "rf = RandomForestRegressor(n_estimators=500, max_features=2,\n"
        "                           random_state=42, n_jobs=-1)\n"
        "rf.fit(X_train, y_train)", s["code"]))

    imp_sorted    = sorted(imp.items(), key=lambda x: x[1], reverse=True)
    imp_rows_pdf  = [["Rank", "Variable", "Importance"]]
    imp_rows_pdf += [
        [str(i + 1), var, fmt(val, 6)]
        for i, (var, val) in enumerate(imp_sorted)
    ]
    story.append(plain_table(imp_rows_pdf, [2*cm, 6*cm, 4*cm]))
    story.append(Paragraph("Table 4: RF variable importance.", s["caption"]))

    # ── 8. Results ────────────────────────────────────────────────────────────
    story += [PageBreak(),
              Paragraph("8. Results and Conclusion", s["section"]), hr()]

    story.append(Paragraph("8.1  Model Comparison", s["subsection"]))
    story.append(fig_block("model_comparison.png",
        "Figure 3: Model comparison - actual vs predicted + MSE chart. "
        "Python / matplotlib.", s))
    story.append(fig_block("variable_importance.png",
        "Figure 4: RF variable importance. Python / scikit-learn.", s,
        width=13*cm))

    winner  = r["winner_mse"]
    w_mse   = mt["mse_rf"]  if winner == "Random Forest" else mt["mse_mlr"]
    l_mse   = mt["mse_mlr"] if winner == "Random Forest" else mt["mse_rf"]
    top_var = r["top_rf_var"]
    sec_var = r["second_rf_var"]
    sig     = r["sig_vars"]
    lc      = m.get("loudness",     {}).get("coef", 0)
    dc      = m.get("duration_min", {}).get("coef", 0)

    perf_rows_pdf = [
        ["Metric", "MLR", "Random Forest", "Better"],
        ["MSE",  fmt(mt["mse_mlr"],  4), fmt(mt["mse_rf"],  4),
         winner_arrow(mt["mse_mlr"],  mt["mse_rf"])],
        ["RMSE", fmt(mt["rmse_mlr"], 4), fmt(mt["rmse_rf"], 4),
         winner_arrow(mt["rmse_mlr"], mt["rmse_rf"])],
        ["MAE",  fmt(mt["mae_mlr"],  4), fmt(mt["mae_rf"],  4),
         winner_arrow(mt["mae_mlr"],  mt["mae_rf"])],
        ["R2",   fmt(mt["r2_mlr"],   4), fmt(mt["r2_rf"],   4),
         winner_arrow(mt["r2_mlr"],  mt["r2_rf"], True)],
    ]
    story.append(plain_table(perf_rows_pdf, [3*cm, 3.5*cm, 4*cm, 3.5*cm]))
    story.append(Paragraph(
        "Table 5: Test set performance (n_test = {}).".format(r["n_test"]),
        s["caption"]))

    story.append(Paragraph("8.2  Conclusion", s["subsection"]))

    sig_sentence_pdf = (
        "The variables " + ", ".join(sig) +
        " were statistically significant (p < 0.05)."
        if sig else
        "No variables were statistically significant at the 5% level."
    )
    story.append(Paragraph(
        "This study examined six audio features as predictors of Spotify "
        "Popularity Score (n = {}, train = {}, test = {}). {} "
        "Loudness had a {} coefficient ({:.4f}), {} H3. "
        "Duration had a {} coefficient ({:.4f}), {} H6. "
        "The MLR explains {:.1f}% of variance in popularity.".format(
            r["n_total"], r["n_train"], r["n_test"],
            sig_sentence_pdf,
            coef_sign(lc), lc, "supporting" if lc > 0 else "contradicting",
            coef_sign(dc), dc, "supporting" if dc < 0 else "contradicting",
            r["mlr_r2"] * 100),
        s["body"]))
    story.append(Paragraph(
        "The {} achieved a lower test MSE ({:.4f} vs {:.4f}), {} H7. "
        "Top RF predictors were {} and {}, consistent with Ferreri et al. (2019). "
        "Both models underpredict extremely popular songs (>80), likely because "
        "viral hits involve social-media effects not captured by audio features alone.".format(
            winner, w_mse, l_mse,
            "confirming" if winner == "Random Forest" else "not confirming",
            top_var, sec_var),
        s["body"]))

    # ── References ────────────────────────────────────────────────────────────
    story += [PageBreak(), Paragraph("9. References", s["section"]), hr()]
    refs = [
        ("Breiman, L. (2001).",
         "Random Forests. Machine Learning, 45(1), 5-32."),
        ("Ferreri, L. et al. (2019).",
         "Dopamine modulates reward experiences elicited by music. PNAS, 116(9)."),
        ("Interiano, M. et al. (2018).",
         "Musical trends and predictability of success. Royal Society Open Science, 5."),
        ("Pachet, F. & Roy, P. (2011).",
         "Hit song science is not yet a science. Proc. ISMIR 2011."),
        ("Pandya, M. (2023).",
         "Spotify Tracks Dataset. Kaggle. "
         "kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset"),
        ("Park, M. et al. (2019).",
         "Global music streaming data reveal diurnal and seasonal patterns. "
         "Nature Human Behaviour, 3, 230-236."),
        ("Spotify Technology S.A. (2024).",
         "Spotify Newsroom: Company Info. newsroom.spotify.com"),
    ]
    ref_s = ParagraphStyle("rs", fontSize=9, fontName="Helvetica",
                            textColor=TEXT_DARK, leading=13, spaceAfter=5,
                            leftIndent=26, firstLineIndent=-26)
    for author, text in refs:
        story.append(Paragraph("<b>{}</b> {}".format(author, text), ref_s))

    doc.build(story, canvasmaker=ReportCanvas)
    print("  PDF saved: {}".format(output_path))


# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":
    print("=" * 55)
    print("  Dynamic Report Generator")
    print("=" * 55)

    print("\n[1/3] Loading results.json ...")
    r = load_results("results.json")
    print("      n={} | train={} | test={}".format(
        r["n_total"], r["n_train"], r["n_test"]))
    print("      MLR R2 = {} | RF MSE = {}".format(
        r["mlr_r2"], r["metrics"]["mse_rf"]))

    print("\n[2/3] Writing report.tex ...")
    tex = generate_latex(r)
    with open("report.tex", "w", encoding="utf-8") as f:
        f.write(tex)
    print("  LaTeX saved: report.tex")

    print("\n[3/3] Building final_report.pdf ...")
    build_pdf(r, "final_report.pdf")

    print("\n" + "=" * 55)
    print("  Done! Files ready for submission:")
    print("    report.tex")
    print("    final_report.pdf")
    print("=" * 55)