"""
ViroSeq Analyzer - Herramienta de análisis de secuencias virales
Desarrollada con Streamlit + BioPython + Plotly
"""

import streamlit as st
import pandas as pd
import numpy as np
import io
import os
import sys
import subprocess
import tempfile
import warnings
import time
import datetime
import textwrap
warnings.filterwarnings("ignore")

try:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib import colors as rl_colors
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import cm, mm
    from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table,
                                     TableStyle, Image as RLImage, PageBreak,
                                     HRFlowable, KeepTogether)
    from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
    from reportlab.graphics.shapes import Drawing, Rect, String
    REPORTLAB_OK = True
except ImportError:
    REPORTLAB_OK = False

# ── Configuración de página ──────────────────────────────────────────────────
st.set_page_config(
    page_title="ViroSeq Analyzer",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS personalizado · Tema claro científico ────────────────────────────────
st.markdown("""
<style>
  /* ── Fuente base ── */
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
  html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

  /* ── Fondo general: blanco limpio ── */
  .stApp {
    background: #f8fafc;
    color: #1e293b;
  }

  /* ── Sidebar: azul muy suave ── */
  [data-testid="stSidebar"] {
    background: linear-gradient(180deg, #eef2ff 0%, #f0f9ff 100%);
    border-right: 1px solid #cbd5e1;
  }
  [data-testid="stSidebar"] * { color: #1e293b !important; }
  [data-testid="stSidebar"] h1,
  [data-testid="stSidebar"] h2,
  [data-testid="stSidebar"] h3 { color: #1d4ed8 !important; }

  /* ── Tarjetas métricas ── */
  [data-testid="metric-container"] {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    padding: 16px;
    box-shadow: 0 1px 4px rgba(0,0,0,0.06);
  }
  [data-testid="metric-container"] label { color: #64748b !important; font-weight: 500; }
  [data-testid="metric-container"] [data-testid="stMetricValue"] {
    color: #1d4ed8 !important;
    font-weight: 700;
  }

  /* ── Tabs ── */
  .stTabs [data-baseweb="tab-list"] {
    background: #e2e8f0;
    border-radius: 12px;
    gap: 4px;
    padding: 5px;
  }
  .stTabs [data-baseweb="tab"] {
    color: #475569;
    border-radius: 8px;
    font-weight: 600;
    font-size: 0.85rem;
  }
  .stTabs [aria-selected="true"] {
    background: #1d4ed8 !important;
    color: #ffffff !important;
    box-shadow: 0 2px 8px rgba(29,78,216,0.3);
  }

  /* ── Botones primarios ── */
  .stButton > button {
    background: linear-gradient(135deg, #1d4ed8, #2563eb);
    color: #ffffff !important;
    border: none;
    border-radius: 8px;
    font-weight: 600;
    padding: 0.5rem 1.2rem;
    transition: all 0.2s;
    box-shadow: 0 2px 6px rgba(29,78,216,0.25);
  }
  .stButton > button:hover {
    background: linear-gradient(135deg, #1e40af, #1d4ed8);
    transform: translateY(-1px);
    box-shadow: 0 4px 14px rgba(29,78,216,0.35);
  }
  .stButton > button:active { transform: translateY(0); }

  /* ── Inputs / Selects / Text areas ── */
  .stTextInput > div > div > input,
  .stTextArea > div > div > textarea,
  .stSelectbox > div > div {
    background: #ffffff;
    border: 1px solid #cbd5e1;
    border-radius: 8px;
    color: #1e293b !important;
  }
  .stTextInput > div > div > input:focus,
  .stTextArea > div > div > textarea:focus {
    border-color: #2563eb;
    box-shadow: 0 0 0 3px rgba(37,99,235,0.15);
  }

  /* ── Labels de inputs ── */
  label, .stSelectbox label, .stTextInput label,
  .stTextArea label, .stNumberInput label,
  .stRadio label, .stCheckbox label,
  .stSlider label { color: #374151 !important; font-weight: 500; }

  /* ── Radio buttons y checkboxes ── */
  .stRadio > div, .stCheckbox > div { color: #374151 !important; }

  /* ── DataFrames ── */
  [data-testid="stDataFrame"] {
    border-radius: 10px;
    overflow: hidden;
    border: 1px solid #e2e8f0;
    box-shadow: 0 1px 4px rgba(0,0,0,0.05);
  }

  /* ── Alertas ── */
  .stAlert { border-radius: 10px; }
  div[data-baseweb="notification"] { color: #1e293b !important; }

  /* ── Headers ── */
  h1 { color: #1d4ed8 !important; font-weight: 700; }
  h2 { color: #1e40af !important; font-weight: 700; }
  h3 { color: #2563eb !important; font-weight: 600; }
  p, li, span { color: #1e293b; }

  /* ── Texto general en la app ── */
  .stMarkdown p, .stMarkdown li { color: #1e293b !important; }
  .stCaption, small, .caption { color: #64748b !important; }

  /* ── Código alineamiento ── */
  .aln-block {
    font-family: 'Courier New', monospace;
    font-size: 12px;
    background: #f1f5f9;
    border: 1px solid #cbd5e1;
    border-radius: 8px;
    padding: 14px;
    overflow-x: auto;
    white-space: pre;
    color: #1e293b;
  }

  /* ── Expanders ── */
  .streamlit-expanderHeader {
    background: #f1f5f9 !important;
    border-radius: 8px !important;
    color: #1e293b !important;
    font-weight: 600;
    border: 1px solid #e2e8f0;
  }
  .streamlit-expanderContent {
    background: #ffffff !important;
    border: 1px solid #e2e8f0;
    border-top: none;
    border-radius: 0 0 8px 8px;
  }

  /* ── Sliders ── */
  .stSlider > div > div > div > div { background: #1d4ed8 !important; }

  /* ── Progress bar ── */
  .stProgress > div > div > div { background: #1d4ed8 !important; }

  /* ── Separador ── */
  hr { border-color: #e2e8f0; }

  /* ── Contenedores con fondo blanco tipo card ── */
  .block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
  }

  /* ── Logo texto sidebar ── */
  .logo-text {
    font-size: 1.8rem;
    font-weight: 800;
    color: #1d4ed8;
    text-align: center;
  }
  .logo-sub {
    text-align: center;
    color: #64748b;
    font-size: 0.85rem;
    margin-top: -6px;
    margin-bottom: 16px;
  }

  /* ── Download buttons ── */
  .stDownloadButton > button {
    background: #ffffff;
    color: #1d4ed8 !important;
    border: 2px solid #1d4ed8;
    border-radius: 8px;
    font-weight: 600;
    transition: all 0.2s;
  }
  .stDownloadButton > button:hover {
    background: #eff6ff;
    box-shadow: 0 2px 8px rgba(29,78,216,0.2);
  }

  /* ── Tooltips ── */
  .stTooltipIcon { color: #64748b !important; }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# IMPORTS con manejo de errores
# ══════════════════════════════════════════════════════════════════════════════
missing_libs = []

try:
    from Bio import SeqIO, AlignIO, Phylo
    from Bio.SeqRecord import SeqRecord
    from Bio.Seq import Seq
    from Bio.Align import MultipleSeqAlignment
    from Bio.Phylo.TreeConstruction import DistanceCalculator, DistanceTreeConstructor
    from Bio.SeqUtils.ProtParam import ProteinAnalysis
    BIOPYTHON_OK = True
except ImportError as e:
    missing_libs.append(f"biopython: {e}")
    BIOPYTHON_OK = False

try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    PLOTLY_OK = True
except ImportError as e:
    missing_libs.append(f"plotly: {e}")
    PLOTLY_OK = False

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    MATPLOTLIB_OK = True
except ImportError as e:
    missing_libs.append(f"matplotlib: {e}")
    MATPLOTLIB_OK = False

try:
    import seaborn as sns
    SEABORN_OK = True
except ImportError as e:
    missing_libs.append(f"seaborn: {e}")
    SEABORN_OK = False

try:
    from scipy.cluster.hierarchy import dendrogram, linkage
    from scipy.spatial.distance import squareform
    SCIPY_OK = True
except ImportError as e:
    missing_libs.append(f"scipy: {e}")
    SCIPY_OK = False

try:
    import dendropy
    DENDROPY_OK = True
except ImportError as e:
    missing_libs.append(f"dendropy: {e}")
    DENDROPY_OK = False

# ══════════════════════════════════════════════════════════════════════════════
# FUNCIONES UTILITARIAS
# ══════════════════════════════════════════════════════════════════════════════

def parse_fasta_text(text: str) -> list:
    """Parsea texto FASTA y retorna lista de SeqRecord."""
    records = []
    try:
        handle = io.StringIO(text.strip())
        records = list(SeqIO.parse(handle, "fasta"))
    except Exception as e:
        st.error(f"Error parseando FASTA: {e}")
    return records


def parse_fastq_to_fasta(content: bytes) -> list:
    """Convierte FASTQ bytes a lista de SeqRecord."""
    records = []
    try:
        handle = io.StringIO(content.decode("utf-8", errors="replace"))
        for rec in SeqIO.parse(handle, "fastq"):
            rec.letter_annotations = {}  # quitar calidades para FASTA
            records.append(rec)
    except Exception as e:
        st.error(f"Error convirtiendo FASTQ: {e}")
    return records


def calc_gc(seq: str) -> float:
    """Calcula GC content en porcentaje."""
    seq = seq.upper()
    gc = seq.count("G") + seq.count("C")
    total = sum(seq.count(b) for b in "ACGT")
    return round(gc / total * 100, 2) if total > 0 else 0.0


def summary_table(records: list) -> pd.DataFrame:
    """Genera tabla resumen de secuencias."""
    rows = []
    for r in records:
        s = str(r.seq).upper()
        rows.append({
            "ID": r.id,
            "Descripción": r.description[:60] if len(r.description) > 60 else r.description,
            "Longitud (bp)": len(s),
            "GC (%)": calc_gc(s),
            "A (%)": round(s.count("A") / len(s) * 100, 1) if len(s) > 0 else 0,
            "T (%)": round(s.count("T") / len(s) * 100, 1) if len(s) > 0 else 0,
            "G (%)": round(s.count("G") / len(s) * 100, 1) if len(s) > 0 else 0,
            "C (%)": round(s.count("C") / len(s) * 100, 1) if len(s) > 0 else 0,
            "N ambiguos": s.count("N"),
        })
    return pd.DataFrame(rows)


def align_sequences_biopython(records: list) -> MultipleSeqAlignment | None:
    """Alineamiento pairwise progresivo simple con BioPython (sin herramienta externa)."""
    try:
        from Bio.Align import PairwiseAligner
        if len(records) < 2:
            st.warning("Se necesitan al menos 2 secuencias para alinear.")
            return None
        
        # Normalizar longitudes con gaps para alineamiento simple
        max_len = max(len(r.seq) for r in records)
        aligned = []
        for r in records:
            padded = str(r.seq) + "-" * (max_len - len(r.seq))
            aligned.append(SeqRecord(Seq(padded), id=r.id, description=r.description))
        return MultipleSeqAlignment(aligned)
    except Exception as e:
        st.error(f"Error en alineamiento: {e}")
        return None


def align_with_muscle(records: list) -> MultipleSeqAlignment | None:
    """Intenta alinear con MUSCLE si está disponible."""
    try:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".fasta", delete=False) as tmp_in:
            SeqIO.write(records, tmp_in, "fasta")
            tmp_in_path = tmp_in.name
        tmp_out_path = tmp_in_path + "_aln.fasta"
        
        result = subprocess.run(
            ["muscle", "-align", tmp_in_path, "-output", tmp_out_path],
            capture_output=True, text=True, timeout=60
        )
        if result.returncode == 0 and os.path.exists(tmp_out_path):
            aln = AlignIO.read(tmp_out_path, "fasta")
            os.unlink(tmp_in_path)
            os.unlink(tmp_out_path)
            return aln
        
        # Intentar versión antigua de MUSCLE
        result = subprocess.run(
            ["muscle", "-in", tmp_in_path, "-out", tmp_out_path],
            capture_output=True, text=True, timeout=60
        )
        if result.returncode == 0 and os.path.exists(tmp_out_path):
            aln = AlignIO.read(tmp_out_path, "fasta")
            os.unlink(tmp_in_path)
            os.unlink(tmp_out_path)
            return aln
    except FileNotFoundError:
        pass
    except Exception as e:
        st.warning(f"MUSCLE no disponible: {e}")
    return None


def align_with_mafft(records: list) -> MultipleSeqAlignment | None:
    """Intenta alinear con MAFFT si está disponible."""
    try:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".fasta", delete=False) as tmp_in:
            SeqIO.write(records, tmp_in, "fasta")
            tmp_in_path = tmp_in.name
        
        result = subprocess.run(
            ["mafft", "--auto", "--quiet", tmp_in_path],
            capture_output=True, text=True, timeout=120
        )
        if result.returncode == 0 and result.stdout:
            handle = io.StringIO(result.stdout)
            aln = AlignIO.read(handle, "fasta")
            os.unlink(tmp_in_path)
            return aln
    except FileNotFoundError:
        pass
    except Exception as e:
        st.warning(f"MAFFT no disponible: {e}")
    return None


def colored_alignment_html(alignment: MultipleSeqAlignment, max_cols: int = 100) -> str:
    """Genera HTML con colores por nucleótido para el alineamiento."""
    colors = {
        "A": "#3fb950",  # verde
        "T": "#f85149",  # rojo
        "G": "#d29922",  # amarillo
        "C": "#58a6ff",  # azul
        "-": "#cbd5e1",  # gris claro gap
        "N": "#94a3b8",  # gris neutro
    }
    
    html = ['<div class="aln-block">']
    for record in alignment:
        seq = str(record.seq)[:max_cols]
        label = record.id[:20].ljust(20)
        html.append(f'<span style="color:#64748b">{label} </span>')
        for nt in seq:
            color = colors.get(nt.upper(), "#1e293b")
            html.append(f'<span style="color:{color};font-weight:600">{nt}</span>')
        html.append("\n")
    html.append("</div>")
    return "".join(html)


def compute_conservation(alignment: MultipleSeqAlignment) -> pd.DataFrame:
    """Calcula conservación y variabilidad por posición."""
    aln_array = np.array([list(str(rec.seq).upper()) for rec in alignment])
    n_seqs, n_cols = aln_array.shape
    
    rows = []
    for i in range(n_cols):
        col = aln_array[:, i]
        col_no_gap = [c for c in col if c != "-"]
        if not col_no_gap:
            continue
        counts = {}
        for nt in col_no_gap:
            counts[nt] = counts.get(nt, 0) + 1
        most_common = max(counts, key=counts.get)
        freq_most = counts[most_common] / len(col_no_gap)
        entropy = -sum((v / len(col_no_gap)) * np.log2(v / len(col_no_gap))
                       for v in counts.values() if v > 0)
        rows.append({
            "Posición": i + 1,
            "Conservación (%)": round(freq_most * 100, 1),
            "Entropía (bits)": round(entropy, 3),
            "NT más frecuente": most_common,
            "Variantes": len(counts),
            "Gaps": int(np.sum(col == "-")),
        })
    return pd.DataFrame(rows)


def compute_distance_matrix(alignment: MultipleSeqAlignment) -> tuple[np.ndarray, list]:
    """Calcula matriz de distancias p entre secuencias."""
    calculator = DistanceCalculator("identity")
    dm = calculator.get_distance(alignment)
    labels = dm.names
    n = len(labels)
    matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            matrix[i][j] = dm[labels[i], labels[j]]
    return matrix, labels


def build_tree_nj(alignment: MultipleSeqAlignment):
    """Construye árbol Neighbor-Joining."""
    calculator = DistanceCalculator("identity")
    constructor = DistanceTreeConstructor(calculator, "nj")
    tree = constructor.build_tree(alignment)
    return tree


def build_tree_upgma(alignment: MultipleSeqAlignment):
    """Construye árbol UPGMA."""
    calculator = DistanceCalculator("identity")
    constructor = DistanceTreeConstructor(calculator, "upgma")
    tree = constructor.build_tree(alignment)
    return tree


def tree_to_newick(tree) -> str:
    """Convierte árbol BioPython a formato Newick."""
    buf = io.StringIO()
    Phylo.write(tree, buf, "newick")
    return buf.getvalue()


def kyte_doolittle(sequence: str, window: int = 9) -> list[float]:
    """Cálculo de hidropatía Kyte-Doolittle."""
    kd = {
        "A": 1.8, "R": -4.5, "N": -3.5, "D": -3.5, "C": 2.5,
        "Q": -3.5, "E": -3.5, "G": -0.4, "H": -3.2, "I": 4.5,
        "L": 3.8, "K": -3.9, "M": 1.9, "F": 2.8, "P": -1.6,
        "S": -0.8, "T": -0.7, "W": -0.9, "Y": -1.3, "V": 4.2,
    }
    seq = sequence.upper()
    scores = []
    half = window // 2
    for i in range(len(seq)):
        start = max(0, i - half)
        end = min(len(seq), i + half + 1)
        window_seq = seq[start:end]
        vals = [kd.get(aa, 0) for aa in window_seq]
        scores.append(sum(vals) / len(vals) if vals else 0)
    return scores


def parker_antigenicity(sequence: str, window: int = 7) -> list[float]:
    """Índice de antigenicidad Parker et al."""
    parker = {
        "A": 0.00, "R": 1.40, "N": 0.50, "D": 1.30, "C": -1.00,
        "Q": 0.80, "E": 1.10, "G": -0.50, "H": 0.50, "I": -1.20,
        "L": -1.10, "K": 1.50, "M": -0.40, "F": -2.50, "P": 0.00,
        "S": 0.10, "T": 0.10, "W": -3.40, "Y": -2.30, "V": -1.00,
    }
    seq = sequence.upper()
    scores = []
    half = window // 2
    for i in range(len(seq)):
        start = max(0, i - half)
        end = min(len(seq), i + half + 1)
        window_seq = seq[start:end]
        vals = [parker.get(aa, 0) for aa in window_seq]
        scores.append(sum(vals) / len(vals) if vals else 0)
    return scores


def records_to_fasta_bytes(records: list) -> bytes:
    """Convierte lista de SeqRecord a bytes FASTA."""
    buf = io.StringIO()
    SeqIO.write(records, buf, "fasta")
    return buf.getvalue().encode()


def alignment_to_bytes(alignment: MultipleSeqAlignment, fmt: str = "fasta") -> bytes:
    """Convierte alineamiento a bytes."""
    buf = io.StringIO()
    AlignIO.write(alignment, buf, fmt)
    return buf.getvalue().encode()


def fig_to_bytes(fig) -> bytes:
    """Convierte figura matplotlib a PNG bytes."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    buf.seek(0)
    return buf.read()


def df_to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown('<div class="logo-text">🧬 ViroSeq</div>', unsafe_allow_html=True)
    st.markdown('<div class="logo-sub">Analizador de Secuencias Virales</div>', unsafe_allow_html=True)
    st.markdown("---")
    
    st.markdown("### ℹ️ Acerca de")
    st.markdown("""
    **ViroSeq Analyzer** es una plataforma de bioinformática para investigadores en virología.
    
    Permite analizar secuencias de virus desde carga hasta árbol filogenético.
    """)
    st.markdown("---")
    
    st.markdown("### 🔧 Estado de librerías")
    libs_status = {
        "BioPython": BIOPYTHON_OK,
        "Plotly": PLOTLY_OK,
        "Matplotlib": MATPLOTLIB_OK,
        "Seaborn": SEABORN_OK,
        "SciPy": SCIPY_OK,
        "DendroPy": DENDROPY_OK,
        "ReportLab (PDF)": REPORTLAB_OK,
    }
    for lib, ok in libs_status.items():
        icon = "✅" if ok else "❌"
        st.markdown(f"{icon} {lib}")
    
    if missing_libs:
        st.error("Instala las librerías faltantes:\n```\npip install " + 
                 " ".join(l.split(":")[0] for l in missing_libs) + "\n```")
    
    st.markdown("---")
    st.markdown("### 📋 Secuencias cargadas")
    if "records" in st.session_state and st.session_state.records:
        n = len(st.session_state.records)
        st.success(f"**{n}** secuencias listas")
        for r in st.session_state.records[:8]:
            st.caption(f"• {r.id[:30]}")
        if n > 8:
            st.caption(f"... y {n-8} más")
    else:
        st.info("Sin secuencias cargadas")
    
    st.markdown("---")
    st.caption("v1.0.0 · Bioinformática Viral · 2025")


# ══════════════════════════════════════════════════════════════════════════════
# INICIALIZAR SESSION STATE
# ══════════════════════════════════════════════════════════════════════════════
if "records" not in st.session_state:
    st.session_state.records = []
if "alignment" not in st.session_state:
    st.session_state.alignment = None
if "conservation_df" not in st.session_state:
    st.session_state.conservation_df = None
if "tree" not in st.session_state:
    st.session_state.tree = None
if "dist_matrix" not in st.session_state:
    st.session_state.dist_matrix = None
if "dist_labels" not in st.session_state:
    st.session_state.dist_labels = []
if "ncbi_results" not in st.session_state:
    st.session_state.ncbi_results = []
if "pipeline_log" not in st.session_state:
    st.session_state.pipeline_log = []
if "pipeline_pdf" not in st.session_state:
    st.session_state.pipeline_pdf = None

# ══════════════════════════════════════════════════════════════════════════════
# TÍTULO PRINCIPAL
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
# 🧬 ViroSeq Analyzer
### Plataforma de Análisis de Secuencias Virales
""")
st.markdown("---")

if not BIOPYTHON_OK:
    st.error("⛔ BioPython no está instalado. Ejecuta: `pip install biopython`")
    st.stop()

# ══════════════════════════════════════════════════════════════════════════════
# TABS PRINCIPALES
# ══════════════════════════════════════════════════════════════════════════════
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9 = st.tabs([
    "📂 Carga de Secuencias",
    "🔗 Alineamiento Múltiple",
    "🔍 Mutaciones Conservadas",
    "🌳 Árbol Filogenético",
    "🔬 Predicción de Epítopos",
    "📊 Dashboard & Exportación",
    "🌐 Buscar en NCBI/GenBank",
    "⚡ Pipeline Automático + PDF",
    "🔎 Identificar Virus (BLAST)",
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 · CARGA DE SECUENCIAS
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown("## 📂 Carga de Secuencias")
    st.markdown("Sube archivos FASTA o FASTQ, o pega tus secuencias directamente.")
    
    col_left, col_right = st.columns([1, 1], gap="large")
    
    with col_left:
        st.markdown("### 📁 Subir archivos")
        uploaded_files = st.file_uploader(
            "Arrastra o selecciona archivos FASTA / FASTQ",
            type=["fasta", "fa", "fna", "ffn", "faa", "frn", "fastq", "fq"],
            accept_multiple_files=True,
            help="Formatos aceptados: .fasta, .fa, .fna, .fastq, .fq"
        )
    
    with col_right:
        st.markdown("### 📝 Pegar secuencias")
        pasted = st.text_area(
            "Pega tus secuencias en formato FASTA",
            height=180,
            placeholder=">Seq1\nATGCATGCATGC...\n>Seq2\nGCTAGCTAGCTA...",
            help="Formato FASTA estándar: >ID\\nSECUENCIA"
        )
    
    # Secuencias de ejemplo
    with st.expander("💡 Cargar secuencias de ejemplo (SARS-CoV-2 simuladas)"):
        if st.button("🧪 Cargar ejemplo demo", key="load_demo"):
            demo_fasta = """>SARS-CoV-2_Alpha|EPI_ISL_1234567|2021-01-15|UK
ATGGAGAGCCTTGTCCCTGGTTTCAACGAGAAAACACACGTCCAACTCAGTTTGCCTGTTTTACAGGTTCGCGACGT
GCTTCGCGATCGTTTGAGTTTTAGTGAGATCGTTGAGCGGTTGATGGCTTATTTCTTTTGCGGCAATGAAACGACTG
TTTAATTGTGCGAATATTTTGCTGTAATCCTGCCACTTTCATCACAGGACGATGGCAACAGTTGCACATGTACTGGT
>SARS-CoV-2_Delta|EPI_ISL_2345678|2021-06-20|India
ATGGAGAGCCTTGTCCCTGGTTTCAACGAGAAAACACACGTCCAACTCAGTTTGCCTGTTTTACAGGTTCGCGACGT
GCTTCGCGATCGTTTGAGTTTTAGTGAGATCGTTGAGCGGTTGATGGCTTATTTCTTTTGCGGCAATGAAACGACAG
TTTAATTGTGCGAATATTTTGCTGTAATCCTGCCACTTTCATCACAGGACGATGGCAACAGTTGCACATGGACTGGT
>SARS-CoV-2_Omicron|EPI_ISL_3456789|2021-11-25|South_Africa
ATGGAGAGCCTTGTCCCTGGTTTCAACGAGAAAACACACGTCCAACTCAGTTTGCCTGTTTTACAGGTTCGCGACGT
GCTTCGCGATCGTTTGAGTTTTAGTGAGATCGTTGAGCGGTTGATGGCTTATTTCTTTTGCGGCAATGAAACGACAG
TTTAATTGTGCGAATATTTTGCTGTAATCCTGCCACTTTCATCACAGGACGATGGCAACAGTTGCACATGGACTGGT
>SARS-CoV-2_BA2|EPI_ISL_4567890|2022-02-10|Denmark
ATGGAGAGCCTTGTCCCTGGTTTCAACGAGAAAACACACGTCCAACTCAGTTTGCCTGTTTTACAGGTTCGCGACGT
GCTTCGCGATCGTTTGAGTTTCAGTGAGATCGTTGAGCGGTTGATGGCTTATTTCTTTTGCGGCAATGAAACGACAG
TTTAATTGTGCGAATATTTTGCTGTAATCCTGCCACTTTCATCACAGGACGATGGCAACAGTTGCACATGGACTGGT
>SARS-CoV-2_XBB|EPI_ISL_5678901|2022-10-05|Singapore
ATGGAGAGCCTTGTCCCTGGTTTCAACGAGAAAACACACGTCCAACTCAGTTTGCCTGTTTTACAGGTTCGCGACGT
GCTTCGCGATCGTTTGAGTTTCAGTGAGATCGTTGAGCGGTTGATGGCTTATTTCTTTTGCGGCAATGAAACGACTG
TTTAATTGTGCGAATATTTTGCTGTAATCCTGCCACTTTCATCACAGGACGATGGCAACAGTTGCACATGTACTGGT"""
            demo_records = parse_fasta_text(demo_fasta)
            st.session_state.records = demo_records
            st.success(f"✅ {len(demo_records)} secuencias de ejemplo cargadas.")
    
    # Botón de carga
    st.markdown("")
    if st.button("⚡ Procesar y cargar secuencias", type="primary", key="load_seqs"):
        all_records = []
        
        # Desde archivos subidos
        if uploaded_files:
            for f in uploaded_files:
                name = f.name.lower()
                content = f.read()
                try:
                    if name.endswith((".fastq", ".fq")):
                        recs = parse_fastq_to_fasta(content)
                        st.info(f"🔄 {f.name}: convertido de FASTQ → FASTA ({len(recs)} seqs)")
                    else:
                        handle = io.StringIO(content.decode("utf-8", errors="replace"))
                        recs = list(SeqIO.parse(handle, "fasta"))
                    
                    if not recs:
                        st.warning(f"⚠️ {f.name}: sin secuencias válidas.")
                    else:
                        all_records.extend(recs)
                        st.success(f"✅ {f.name}: {len(recs)} secuencias")
                except Exception as e:
                    st.error(f"❌ Error en {f.name}: {e}")
        
        # Desde texto pegado
        if pasted.strip():
            recs = parse_fasta_text(pasted)
            if recs:
                all_records.extend(recs)
                st.success(f"✅ Texto: {len(recs)} secuencias parseadas")
            else:
                st.error("❌ No se encontraron secuencias válidas en el texto pegado.")
        
        if all_records:
            # Deduplicar IDs
            seen = {}
            unique = []
            for r in all_records:
                if r.id not in seen:
                    seen[r.id] = 0
                seen[r.id] += 1
                if seen[r.id] > 1:
                    r.id = f"{r.id}_{seen[r.id]}"
                unique.append(r)
            st.session_state.records = unique
            st.session_state.alignment = None  # Reset alineamiento
            st.session_state.conservation_df = None
            st.session_state.tree = None
            st.balloons()
        else:
            if not uploaded_files and not pasted.strip():
                st.warning("⚠️ No se proporcionaron secuencias. Sube archivos o pega texto.")
    
    # Mostrar tabla resumen
    if st.session_state.records:
        records = st.session_state.records
        st.markdown("---")
        st.markdown("### 📊 Resumen de secuencias cargadas")
        
        # Métricas rápidas
        df_sum = summary_table(records)
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("🔢 Secuencias", len(records))
        c2.metric("📏 Long. media (bp)", f"{df_sum['Longitud (bp)'].mean():.0f}")
        c3.metric("📐 Long. mín.", df_sum['Longitud (bp)'].min())
        c4.metric("📐 Long. máx.", df_sum['Longitud (bp)'].max())
        c5.metric("🧪 GC% medio", f"{df_sum['GC (%)'].mean():.1f}%")
        
        st.markdown("")
        st.dataframe(df_sum, use_container_width=True, hide_index=True)
        
        # Gráfico de distribución
        if PLOTLY_OK and len(records) > 1:
            st.markdown("### 📈 Distribución de longitudes y GC%")
            fig = make_subplots(rows=1, cols=2,
                                subplot_titles=("Distribución de Longitudes", "GC Content por Secuencia"))
            
            fig.add_trace(go.Histogram(
                x=df_sum["Longitud (bp)"], nbinsx=20,
                marker_color="#1f6feb", opacity=0.8, name="Longitud"
            ), row=1, col=1)
            
            fig.add_trace(go.Bar(
                x=df_sum["ID"], y=df_sum["GC (%)"],
                marker_color="#3fb950", name="GC%"
            ), row=1, col=2)
            
            fig.update_layout(
                paper_bgcolor="#ffffff", plot_bgcolor="#f8fafc",
                font=dict(color="#1e293b"), showlegend=False, height=380,
            )
            fig.update_xaxes(gridcolor="#e2e8f0")
            fig.update_yaxes(gridcolor="#e2e8f0")
            st.plotly_chart(fig, use_container_width=True)
        
        # Descarga FASTA
        st.download_button(
            "💾 Descargar todas las secuencias (FASTA)",
            data=records_to_fasta_bytes(records),
            file_name="secuencias_cargadas.fasta",
            mime="text/plain",
        )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 · ALINEAMIENTO MÚLTIPLE
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("## 🔗 Alineamiento Múltiple de Secuencias")
    
    if not st.session_state.records:
        st.warning("⚠️ Primero carga secuencias en la pestaña **📂 Carga de Secuencias**.")
    else:
        records = st.session_state.records
        
        col_opts, col_info = st.columns([1, 1])
        with col_opts:
            st.markdown("### ⚙️ Opciones de alineamiento")
            aligner_choice = st.radio(
                "Herramienta de alineamiento",
                options=["MUSCLE (externo)", "MAFFT (externo)", "Alineamiento interno (BioPython)"],
                help="MUSCLE y MAFFT requieren instalación externa. El interno usa padding simple.",
                index=2
            )
            max_display = st.slider("Columnas a mostrar", 50, 300, 120, 10)
        
        with col_info:
            st.markdown("### ℹ️ Información")
            st.info(f"""
            **Secuencias a alinear:** {len(records)}
            
            - **MUSCLE / MAFFT:** Requieren instalación en el sistema (`brew install muscle` o `apt install mafft`)
            - **Interno:** Padding simple, suficiente para visualización básica
            """)
        
        if st.button("▶️ Ejecutar alineamiento", type="primary", key="run_align"):
            with st.spinner("Alineando secuencias..."):
                aln = None
                
                if "MUSCLE" in aligner_choice:
                    aln = align_with_muscle(records)
                    if aln is None:
                        st.warning("MUSCLE no encontrado. Usando alineamiento interno.")
                        aln = align_sequences_biopython(records)
                elif "MAFFT" in aligner_choice:
                    aln = align_with_mafft(records)
                    if aln is None:
                        st.warning("MAFFT no encontrado. Usando alineamiento interno.")
                        aln = align_sequences_biopython(records)
                else:
                    aln = align_sequences_biopython(records)
                
                if aln:
                    st.session_state.alignment = aln
                    st.session_state.conservation_df = None  # Reset
                    st.success(f"✅ Alineamiento completado: {len(aln)} secuencias × {aln.get_alignment_length()} posiciones")
        
        # Mostrar alineamiento
        if st.session_state.alignment:
            aln = st.session_state.alignment
            
            st.markdown("---")
            st.markdown(f"### 🎨 Visualización del alineamiento ({len(aln)} seqs × {aln.get_alignment_length()} cols)")
            
            # Leyenda de colores
            st.markdown("""
            <div style="display:flex;gap:16px;margin-bottom:12px;flex-wrap:wrap">
                <span style="color:#3fb950;font-weight:700">■ A (Adenina)</span>
                <span style="color:#f85149;font-weight:700">■ T (Timina)</span>
                <span style="color:#d29922;font-weight:700">■ G (Guanina)</span>
                <span style="color:#58a6ff;font-weight:700">■ C (Citosina)</span>
                <span style="color:#30363d;background:#8b949e;font-weight:700">■ - (Gap)</span>
                <span style="color:#64748b;font-weight:700">■ N (Ambiguo)</span>
            </div>
            """, unsafe_allow_html=True)
            
            aln_html = colored_alignment_html(aln, max_display)
            st.markdown(aln_html, unsafe_allow_html=True)
            
            if aln.get_alignment_length() > max_display:
                st.caption(f"⚠️ Mostrando las primeras {max_display} de {aln.get_alignment_length()} columnas.")
            
            # Botones de descarga
            st.markdown("### 💾 Descargar alineamiento")
            col_dl1, col_dl2, col_dl3 = st.columns(3)
            
            with col_dl1:
                st.download_button(
                    "📥 Descargar .fasta",
                    data=alignment_to_bytes(aln, "fasta"),
                    file_name="alineamiento.fasta",
                    mime="text/plain",
                )
            with col_dl2:
                st.download_button(
                    "📥 Descargar .clustal",
                    data=alignment_to_bytes(aln, "clustal"),
                    file_name="alineamiento.aln",
                    mime="text/plain",
                )
            with col_dl3:
                newick_str = ""
                if st.session_state.tree:
                    newick_str = tree_to_newick(st.session_state.tree)
                st.download_button(
                    "📥 Alineamiento como PHYLIP",
                    data=alignment_to_bytes(aln, "phylip-relaxed"),
                    file_name="alineamiento.phy",
                    mime="text/plain",
                )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 · MUTACIONES CONSERVADAS
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown("## 🔍 Análisis de Mutaciones y Conservación")
    
    if st.session_state.alignment is None:
        st.warning("⚠️ Primero ejecuta el **alineamiento** en la pestaña correspondiente.")
    else:
        aln = st.session_state.alignment
        
        # Calcular conservación
        if st.session_state.conservation_df is None:
            with st.spinner("Calculando conservación..."):
                st.session_state.conservation_df = compute_conservation(aln)
        
        cons_df = st.session_state.conservation_df
        
        # Métricas
        n_conserved = (cons_df["Conservación (%)"] == 100).sum()
        n_variable = (cons_df["Conservación (%)"] < 100).sum()
        n_highly_var = (cons_df["Conservación (%)"] < 80).sum()
        mean_conservation = cons_df["Conservación (%)"].mean()
        
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("✅ Posiciones conservadas (100%)", n_conserved)
        c2.metric("⚠️ Posiciones variables", n_variable)
        c3.metric("🔴 Muy variables (<80%)", n_highly_var)
        c4.metric("📊 Conservación media", f"{mean_conservation:.1f}%")
        
        # Filtro opcional
        st.markdown("---")
        col_filter, col_table = st.columns([1, 2])
        
        with col_filter:
            st.markdown("### 🔎 Filtros")
            min_cons = st.slider("Conservación mínima (%)", 0, 100, 0, 5)
            max_cons = st.slider("Conservación máxima (%)", 0, 100, 100, 5)
            
            st.markdown("### 🧬 Filtrar por región génica")
            region_start = st.number_input("Posición inicio", min_value=1, 
                                            max_value=len(cons_df), value=1)
            region_end = st.number_input("Posición fin", min_value=1,
                                          max_value=len(cons_df), value=len(cons_df))
            gene_name = st.text_input("Nombre del gen (opcional)", 
                                       placeholder="Ej: Spike, RdRp, N")
        
        filtered_df = cons_df[
            (cons_df["Conservación (%)"] >= min_cons) &
            (cons_df["Conservación (%)"] <= max_cons) &
            (cons_df["Posición"] >= region_start) &
            (cons_df["Posición"] <= region_end)
        ]
        
        with col_table:
            st.markdown(f"### 📋 Tabla de conservación ({len(filtered_df)} posiciones)")
            if gene_name:
                st.caption(f"🧬 Región: **{gene_name}** (pos. {region_start}–{region_end})")
            st.dataframe(
                filtered_df.style.background_gradient(
                    subset=["Conservación (%)"], cmap="RdYlGn"
                ).background_gradient(
                    subset=["Entropía (bits)"], cmap="YlOrRd"
                ),
                use_container_width=True, height=320, hide_index=True
            )
        
        # Gráficos
        st.markdown("---")
        st.markdown("### 📈 Visualización de conservación y variabilidad")
        
        if PLOTLY_OK:
            fig = make_subplots(
                rows=2, cols=1,
                subplot_titles=(
                    "Conservación por posición (%)",
                    "Entropía de Shannon (bits) – Variabilidad"
                ),
                shared_xaxes=True, vertical_spacing=0.12
            )
            
            # Conservación
            fig.add_trace(go.Scatter(
                x=filtered_df["Posición"], y=filtered_df["Conservación (%)"],
                mode="lines", line=dict(color="#3fb950", width=1.5),
                fill="tozeroy", fillcolor="rgba(63,185,80,0.15)",
                name="Conservación"
            ), row=1, col=1)
            
            # Línea 100%
            fig.add_hline(y=100, line_dash="dot", line_color="#58a6ff", 
                          annotation_text="100% conservado", row=1, col=1)
            fig.add_hline(y=80, line_dash="dot", line_color="#d29922",
                          annotation_text="80%", row=1, col=1)
            
            # Entropía
            fig.add_trace(go.Bar(
                x=filtered_df["Posición"], y=filtered_df["Entropía (bits)"],
                marker_color=filtered_df["Entropía (bits)"],
                marker_colorscale="Reds", name="Entropía",
                showlegend=False
            ), row=2, col=1)
            
            fig.update_layout(
                paper_bgcolor="#ffffff", plot_bgcolor="#f8fafc",
                font=dict(color="#1e293b"), height=550, showlegend=True,
            )
            fig.update_xaxes(gridcolor="#e2e8f0", title_text="Posición (bp)")
            fig.update_yaxes(gridcolor="#e2e8f0")
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Descargar
        st.download_button(
            "💾 Descargar tabla de mutaciones (CSV)",
            data=df_to_csv_bytes(filtered_df),
            file_name=f"mutaciones{'_'+gene_name if gene_name else ''}.csv",
            mime="text/csv",
        )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 · ÁRBOL FILOGENÉTICO
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown("## 🌳 Árbol Filogenético")
    
    if st.session_state.alignment is None:
        st.warning("⚠️ Primero ejecuta el **alineamiento** en la pestaña correspondiente.")
    else:
        aln = st.session_state.alignment
        
        col_opts, col_meta = st.columns([1, 1])
        
        with col_opts:
            st.markdown("### ⚙️ Opciones")
            tree_method = st.radio(
                "Método de construcción",
                ["Neighbor-Joining (NJ)", "UPGMA"],
                help="NJ es más preciso; UPGMA asume tasa de evolución constante."
            )
            tree_style = st.radio("Estilo visual", ["Rectangular", "Radial"])
        
        with col_meta:
            st.markdown("### 🗂️ Metadata (opcional)")
            st.markdown("Sube un CSV con columnas `ID` y `Grupo` para colorear ramas.")
            meta_file = st.file_uploader("Metadata CSV", type=["csv"], key="meta_csv")
            
            meta_df = None
            if meta_file:
                try:
                    meta_df = pd.read_csv(meta_file)
                    st.success(f"✅ Metadata cargada: {len(meta_df)} registros")
                    st.dataframe(meta_df.head(), use_container_width=True)
                except Exception as e:
                    st.error(f"Error leyendo metadata: {e}")
            
            st.markdown("**O ingresa metadata manualmente:**")
            meta_text = st.text_area(
                "ID,Grupo (una por línea)",
                placeholder="Seq1,Variante_Alpha\nSeq2,Variante_Delta",
                height=100
            )
            if meta_text.strip() and meta_df is None:
                try:
                    meta_df = pd.read_csv(io.StringIO("ID,Grupo\n" + meta_text))
                except:
                    pass
        
        if st.button("🌳 Construir árbol filogenético", type="primary", key="build_tree"):
            with st.spinner("Construyendo árbol..."):
                try:
                    if "NJ" in tree_method:
                        tree = build_tree_nj(aln)
                    else:
                        tree = build_tree_upgma(aln)
                    st.session_state.tree = tree
                    
                    # Calcular matrix de distancias
                    dm_matrix, dm_labels = compute_distance_matrix(aln)
                    st.session_state.dist_matrix = dm_matrix
                    st.session_state.dist_labels = dm_labels
                    
                    st.success(f"✅ Árbol {tree_method} construido correctamente.")
                except Exception as e:
                    st.error(f"❌ Error construyendo árbol: {e}")
        
        # Mostrar árbol
        if st.session_state.tree and PLOTLY_OK:
            tree = st.session_state.tree
            
            st.markdown("---")
            st.markdown("### 🌳 Visualización del árbol")
            
            # Extraer clados para Plotly
            def get_clade_coords(clade, x=0, y_counter=[0], coords=None, edges=None, labels=None):
                if coords is None: coords = []
                if edges is None: edges = []
                if labels is None: labels = []
                
                branch_length = clade.branch_length or 0.01
                x_new = x + branch_length
                
                if clade.is_terminal():
                    y = y_counter[0]
                    y_counter[0] += 1
                    coords.append((x_new, y, clade.name or ""))
                    labels.append(clade.name or "")
                    return x_new, y, coords, edges, labels
                else:
                    child_ys = []
                    for child in clade:
                        cx, cy, coords, edges, labels = get_clade_coords(
                            child, x_new, y_counter, coords, edges, labels)
                        edges.append((x, (child_ys[-1] if child_ys else cy), x_new, cy))
                        child_ys.append(cy)
                    
                    y_mid = (child_ys[0] + child_ys[-1]) / 2
                    # Línea vertical conectando hijos
                    edges.append((x_new, child_ys[0], x_new, child_ys[-1]))
                    return x_new, y_mid, coords, edges, labels
            
            try:
                root = tree.root
                coords_list = []
                edges_list = []
                labels_list = []
                y_ctr = [0]
                get_clade_coords(root, 0, y_ctr, coords_list, edges_list, labels_list)
                
                # Colores por grupo si hay metadata
                color_map = {}
                if meta_df is not None:
                    groups = meta_df["Grupo"].unique().tolist()
                    palette = px.colors.qualitative.Bold + px.colors.qualitative.Pastel
                    for idx, grp in enumerate(groups):
                        color_map[grp] = palette[idx % len(palette)]
                    
                    id_to_group = dict(zip(meta_df["ID"].astype(str), meta_df["Grupo"]))
                
                fig = go.Figure()
                
                # Dibujar ramas
                for ex1, ey1, ex2, ey2 in edges_list:
                    fig.add_shape(type="line",
                                  x0=ex1, y0=ey1, x1=ex2, y1=ey2,
                                  line=dict(color="#2563eb", width=2))
                
                # Dibujar nodos terminales
                for (cx, cy, name) in coords_list:
                    grp = id_to_group.get(name, "Sin grupo") if meta_df is not None else None
                    color = color_map.get(grp, "#3fb950") if grp else "#3fb950"
                    
                    fig.add_trace(go.Scatter(
                        x=[cx], y=[cy],
                        mode="markers+text",
                        marker=dict(size=10, color=color, 
                                    line=dict(color="white", width=1)),
                        text=[f"  {name[:30]}"],
                        textposition="middle right",
                        textfont=dict(size=10, color="#1e293b"),
                        name=grp or name,
                        hovertext=f"{name}<br>Grupo: {grp}" if grp else name,
                        hoverinfo="text",
                    ))
                
                # Leyenda de grupos
                if meta_df is not None:
                    for grp, col in color_map.items():
                        fig.add_trace(go.Scatter(
                            x=[None], y=[None], mode="markers",
                            marker=dict(size=12, color=col),
                            name=grp, showlegend=True
                        ))
                
                fig.update_layout(
                    paper_bgcolor="#ffffff", plot_bgcolor="#f8fafc",
                    font=dict(color="#1e293b"),
                    showlegend=(meta_df is not None),
                    height=max(400, len(labels_list) * 28),
                    xaxis=dict(title="Distancia evolutiva", gridcolor="#e2e8f0",
                               showgrid=True),
                    yaxis=dict(visible=False),
                    margin=dict(l=20, r=250, t=40, b=40),
                    title=dict(text=f"Árbol Filogenético ({tree_method})",
                               font=dict(color="#1d4ed8")),
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            except Exception as e:
                st.warning(f"Visualización interactiva no disponible: {e}")
                
                # Fallback matplotlib
                if MATPLOTLIB_OK:
                    fig_mpl, ax = plt.subplots(figsize=(10, max(5, len(aln) * 0.6)),
                                               facecolor="white")
                    ax.set_facecolor("#f8fafc")
                    Phylo.draw(tree, axes=ax, do_show=False,
                               label_colors={"Clade": "#374151"})
                    ax.tick_params(colors="#374151")
                    for spine in ax.spines.values():
                        spine.set_edgecolor("#e2e8f0")
                    st.pyplot(fig_mpl)
            
            # Descargar árbol
            st.markdown("### 💾 Descargar árbol")
            col_dl1, col_dl2 = st.columns(2)
            with col_dl1:
                newick = tree_to_newick(tree)
                st.download_button(
                    "📥 Descargar árbol (Newick)",
                    data=newick.encode(),
                    file_name="arbol_filogenetico.nwk",
                    mime="text/plain",
                )
            with col_dl2:
                buf = io.StringIO()
                Phylo.write(tree, buf, "nexus")
                st.download_button(
                    "📥 Descargar árbol (NEXUS)",
                    data=buf.getvalue().encode(),
                    file_name="arbol_filogenetico.nexus",
                    mime="text/plain",
                )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 · PREDICCIÓN DE EPÍTOPOS
# ══════════════════════════════════════════════════════════════════════════════
with tab5:
    st.markdown("## 🔬 Predicción Básica de Epítopos")
    st.markdown("""
    Analiza propiedades fisicoquímicas de secuencias proteicas para identificar 
    posibles regiones antigénicas usando los índices de **Kyte-Doolittle** (hidropatía) 
    y **Parker** (antigenicidad).
    """)
    
    if not st.session_state.records:
        st.warning("⚠️ Primero carga secuencias en la pestaña **📂 Carga de Secuencias**.")
    else:
        records = st.session_state.records
        
        col_sel, col_param = st.columns([1, 1])
        
        with col_sel:
            st.markdown("### 🧬 Seleccionar secuencia")
            seq_ids = [r.id for r in records]
            selected_id = st.selectbox("Secuencia a analizar", seq_ids)
            seq_type = st.radio("Tipo de secuencia", 
                                ["Proteína (aa)", "Nucleótido (traducir automáticamente)"])
        
        with col_param:
            st.markdown("### ⚙️ Parámetros")
            window_kd = st.slider("Ventana Kyte-Doolittle", 3, 21, 9, 2)
            window_pk = st.slider("Ventana Parker", 3, 15, 7, 2)
            threshold_ant = st.slider("Umbral antigenicidad (Parker)", -3.0, 3.0, 0.5, 0.1)
        
        if st.button("🔬 Analizar epítopos", type="primary", key="analyze_epitopes"):
            selected_rec = next(r for r in records if r.id == selected_id)
            seq_str = str(selected_rec.seq).upper()
            
            # Traducir si es nucleótido
            if "Nucleótido" in seq_type:
                try:
                    prot_seq = str(Seq(seq_str[:len(seq_str)//3*3]).translate(to_stop=True))
                    if len(prot_seq) < 10:
                        st.error("La traducción produjo una secuencia proteica muy corta. ¿Es realmente un nucleótido?")
                        st.stop()
                    seq_str = prot_seq
                    st.info(f"🔄 Traducido a proteína: {len(seq_str)} aminoácidos")
                except Exception as e:
                    st.error(f"Error en traducción: {e}")
                    st.stop()
            
            # Filtrar solo aminoácidos válidos
            valid_aa = set("ACDEFGHIKLMNPQRSTVWY")
            seq_clean = "".join(c for c in seq_str if c in valid_aa)
            
            if len(seq_clean) < 10:
                st.error("Secuencia proteica demasiado corta o inválida.")
            else:
                # Cálculos
                kd_scores = kyte_doolittle(seq_clean, window_kd)
                pk_scores = parker_antigenicity(seq_clean, window_pk)
                positions = list(range(1, len(seq_clean) + 1))
                
                # Propiedades globales con BioPython
                try:
                    prot_analysis = ProteinAnalysis(seq_clean)
                    mw = prot_analysis.molecular_weight()
                    pi = prot_analysis.isoelectric_point()
                    instability = prot_analysis.instability_index()
                    aromaticity = prot_analysis.aromaticity()
                    
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("⚖️ Peso molecular (Da)", f"{mw:,.0f}")
                    c2.metric("⚡ Punto isoeléctrico (pI)", f"{pi:.2f}")
                    c3.metric("💧 Índice inestabilidad", f"{instability:.1f}")
                    c4.metric("💍 Aromaticidad", f"{aromaticity:.3f}")
                except Exception as e:
                    st.warning(f"Análisis proteico parcial: {e}")
                
                # Identificar epítopos potenciales (Parker > threshold)
                epitopes = []
                in_epitope = False
                ep_start = 0
                for i, score in enumerate(pk_scores):
                    if score >= threshold_ant and not in_epitope:
                        in_epitope = True
                        ep_start = i
                    elif score < threshold_ant and in_epitope:
                        if i - ep_start >= 5:  # mín 5 aa
                            epitopes.append({
                                "Inicio": ep_start + 1,
                                "Fin": i,
                                "Longitud": i - ep_start,
                                "Secuencia": seq_clean[ep_start:i],
                                "Parker medio": round(np.mean(pk_scores[ep_start:i]), 3),
                                "KD medio": round(np.mean(kd_scores[ep_start:i]), 3),
                            })
                        in_epitope = False
                
                # Gráfico
                if PLOTLY_OK:
                    fig = make_subplots(
                        rows=2, cols=1,
                        subplot_titles=(
                            f"Hidropatía Kyte-Doolittle (ventana={window_kd})",
                            f"Antigenicidad Parker (ventana={window_pk})"
                        ),
                        shared_xaxes=True, vertical_spacing=0.12
                    )
                    
                    # KD
                    fig.add_trace(go.Scatter(
                        x=positions, y=kd_scores,
                        mode="lines", line=dict(color="#58a6ff", width=1.5),
                        fill="tozeroy", fillcolor="rgba(88,166,255,0.1)",
                        name="Hidropatía"
                    ), row=1, col=1)
                    fig.add_hline(y=1.6, line_dash="dot", line_color="#d29922",
                                  annotation_text="Umbral TM (1.6)", row=1, col=1)
                    fig.add_hline(y=0, line_dash="solid", line_color="#cbd5e1", row=1, col=1)
                    
                    # Parker
                    colors_bar = ["#16a34a" if s >= threshold_ant else "#cbd5e1" 
                                  for s in pk_scores]
                    fig.add_trace(go.Bar(
                        x=positions, y=pk_scores,
                        marker_color=colors_bar, name="Antigenicidad",
                    ), row=2, col=1)
                    fig.add_hline(y=threshold_ant, line_dash="dot", line_color="#f85149",
                                  annotation_text=f"Umbral ({threshold_ant})", row=2, col=1)
                    
                    # Marcar epítopos
                    for ep in epitopes:
                        fig.add_vrect(
                            x0=ep["Inicio"], x1=ep["Fin"],
                            fillcolor="rgba(63,185,80,0.15)",
                            layer="below", line_width=0,
                            row=2, col=1
                        )
                    
                    fig.update_layout(
                        paper_bgcolor="#ffffff", plot_bgcolor="#f8fafc",
                        font=dict(color="#1e293b"), height=600,
                        title=dict(text=f"Análisis de epítopos: {selected_id}",
                                   font=dict(color="#1d4ed8")),
                    )
                    fig.update_xaxes(gridcolor="#e2e8f0", title_text="Posición (aa)")
                    fig.update_yaxes(gridcolor="#e2e8f0")
                    
                    st.plotly_chart(fig, use_container_width=True)
                
                # Tabla de epítopos predichos
                st.markdown(f"### 🎯 Epítopos predichos: **{len(epitopes)}** regiones")
                if epitopes:
                    ep_df = pd.DataFrame(epitopes)
                    st.dataframe(ep_df, use_container_width=True, hide_index=True)
                    st.download_button(
                        "💾 Descargar epítopos (CSV)",
                        data=df_to_csv_bytes(ep_df),
                        file_name=f"epitopos_{selected_id}.csv",
                        mime="text/csv",
                    )
                else:
                    st.info("No se encontraron regiones con antigenicidad por encima del umbral. Prueba reduciendo el umbral.")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 6 · DASHBOARD & EXPORTACIÓN
# ══════════════════════════════════════════════════════════════════════════════
with tab6:
    st.markdown("## 📊 Dashboard General & Exportación")
    
    if not st.session_state.records:
        st.warning("⚠️ Carga secuencias primero para ver el dashboard.")
    else:
        records = st.session_state.records
        df_sum = summary_table(records)
        
        st.markdown("### 🗺️ Panel de composición nucleotídica")
        
        # Heatmap de composición
        if PLOTLY_OK and len(records) > 0:
            nuc_cols = ["A (%)", "T (%)", "G (%)", "C (%)"]
            heat_data = df_sum[nuc_cols].values
            
            fig_heat = go.Figure(data=go.Heatmap(
                z=heat_data,
                x=nuc_cols,
                y=df_sum["ID"].tolist(),
                colorscale="Blues",
                text=heat_data.round(1),
                texttemplate="%{text}%",
                colorbar=dict(title="Frecuencia (%)")
            ))
            fig_heat.update_layout(
                title="Composición nucleotídica por secuencia",
                paper_bgcolor="#ffffff", plot_bgcolor="#f8fafc",
                font=dict(color="#1e293b"), height=max(300, len(records) * 30 + 100),
            )
            st.plotly_chart(fig_heat, use_container_width=True)
        
        # Matriz de distancias (heatmap)
        if st.session_state.dist_matrix is not None and PLOTLY_OK:
            st.markdown("### 🧩 Matriz de distancias genéticas")
            dm = st.session_state.dist_matrix
            labels = st.session_state.dist_labels
            
            # Acortar labels
            short_labels = [l[:25] for l in labels]
            
            fig_dm = go.Figure(data=go.Heatmap(
                z=dm,
                x=short_labels,
                y=short_labels,
                colorscale="Viridis",
                text=np.round(dm, 3),
                texttemplate="%{text}",
                colorbar=dict(title="Distancia p")
            ))
            fig_dm.update_layout(
                title="Heatmap de distancias genéticas (p-distance)",
                paper_bgcolor="#ffffff", plot_bgcolor="#f8fafc",
                font=dict(color="#1e293b"),
                height=max(400, len(labels) * 35 + 100),
                xaxis=dict(tickangle=-45),
            )
            st.plotly_chart(fig_dm, use_container_width=True)
            
            # Descargar matrix
            dm_df = pd.DataFrame(dm, index=labels, columns=labels)
            st.download_button(
                "💾 Descargar matriz de distancias (CSV)",
                data=dm_df.to_csv().encode(),
                file_name="matriz_distancias.csv",
                mime="text/csv",
            )
        elif st.session_state.alignment is None:
            st.info("💡 Ejecuta el alineamiento y construye el árbol para ver la matriz de distancias.")
        
        # GC vs Longitud scatter
        if PLOTLY_OK and len(records) > 1:
            st.markdown("### 📉 Correlación GC% vs Longitud")
            fig_sc = px.scatter(
                df_sum, x="Longitud (bp)", y="GC (%)",
                text="ID", color="GC (%)",
                color_continuous_scale="Viridis",
                title="GC% vs Longitud de secuencia"
            )
            fig_sc.update_traces(textposition="top center", textfont_size=9)
            fig_sc.update_layout(
                paper_bgcolor="#ffffff", plot_bgcolor="#f8fafc",
                font=dict(color="#1e293b"), height=420,
            )
            fig_sc.update_xaxes(gridcolor="#30363d")
            fig_sc.update_yaxes(gridcolor="#30363d")
            st.plotly_chart(fig_sc, use_container_width=True)
        
        # ── Panel de exportación ──────────────────────────────────────────────
        st.markdown("---")
        st.markdown("### 💾 Exportación de resultados")
        
        col_e1, col_e2, col_e3 = st.columns(3)
        
        with col_e1:
            st.markdown("**📝 Tablas CSV**")
            st.download_button(
                "📥 Resumen de secuencias",
                data=df_to_csv_bytes(df_sum),
                file_name="resumen_secuencias.csv",
                mime="text/csv",
                key="dl_summary"
            )
            if st.session_state.conservation_df is not None:
                st.download_button(
                    "📥 Tabla de conservación",
                    data=df_to_csv_bytes(st.session_state.conservation_df),
                    file_name="conservacion.csv",
                    mime="text/csv",
                    key="dl_cons"
                )
            if st.session_state.dist_matrix is not None:
                dm_df = pd.DataFrame(
                    st.session_state.dist_matrix,
                    index=st.session_state.dist_labels,
                    columns=st.session_state.dist_labels
                )
                st.download_button(
                    "📥 Matriz de distancias",
                    data=dm_df.to_csv().encode(),
                    file_name="matriz_distancias.csv",
                    mime="text/csv",
                    key="dl_dm"
                )
        
        with col_e2:
            st.markdown("**🧬 Secuencias**")
            st.download_button(
                "📥 Secuencias (FASTA)",
                data=records_to_fasta_bytes(records),
                file_name="secuencias.fasta",
                mime="text/plain",
                key="dl_fasta"
            )
            if st.session_state.alignment is not None:
                st.download_button(
                    "📥 Alineamiento (FASTA)",
                    data=alignment_to_bytes(st.session_state.alignment, "fasta"),
                    file_name="alineamiento.fasta",
                    mime="text/plain",
                    key="dl_aln"
                )
        
        with col_e3:
            st.markdown("**🌳 Árbol filogenético**")
            if st.session_state.tree is not None:
                newick = tree_to_newick(st.session_state.tree)
                st.download_button(
                    "📥 Árbol (Newick)",
                    data=newick.encode(),
                    file_name="arbol.nwk",
                    mime="text/plain",
                    key="dl_nwk"
                )
                buf = io.StringIO()
                Phylo.write(st.session_state.tree, buf, "nexus")
                st.download_button(
                    "📥 Árbol (NEXUS)",
                    data=buf.getvalue().encode(),
                    file_name="arbol.nexus",
                    mime="text/plain",
                    key="dl_nex"
                )
            else:
                st.info("Construye el árbol para habilitar la exportación.")
        
        # Informe de texto
        st.markdown("---")
        st.markdown("### 📄 Informe resumen")
        
        report = f"""# Informe ViroSeq Analyzer
Generado automáticamente

## Secuencias analizadas
- Total: {len(records)} secuencias
- Longitud media: {df_sum['Longitud (bp)'].mean():.0f} bp
- GC% medio: {df_sum['GC (%)'].mean():.1f}%
- Longitud mínima: {df_sum['Longitud (bp)'].min()} bp
- Longitud máxima: {df_sum['Longitud (bp)'].max()} bp

## IDs de secuencias
{chr(10).join('- ' + r.id for r in records)}

## Alineamiento
{"Ejecutado: " + str(len(st.session_state.alignment)) + " secuencias × " + str(st.session_state.alignment.get_alignment_length()) + " posiciones" if st.session_state.alignment else "No ejecutado"}

## Conservación
{f"Posiciones conservadas (100%): {(st.session_state.conservation_df['Conservación (%)'] == 100).sum()}" if st.session_state.conservation_df is not None else "No calculada"}
{f"Conservación media: {st.session_state.conservation_df['Conservación (%)'].mean():.1f}%" if st.session_state.conservation_df is not None else ""}

## Árbol filogenético
{"Construido (ver archivo .nwk)" if st.session_state.tree else "No construido"}
"""
        
        st.download_button(
            "📥 Descargar informe completo (TXT)",
            data=report.encode("utf-8"),
            file_name="informe_viroseq.txt",
            mime="text/plain",
            key="dl_report"
        )

# ══════════════════════════════════════════════════════════════════════════════
# TAB 7 · NCBI / GENBANK
# ══════════════════════════════════════════════════════════════════════════════
with tab7:
    st.markdown("## 🌐 Buscar y Descargar Secuencias de NCBI/GenBank")
    st.markdown("""
    Busca secuencias directamente desde **NCBI Nucleotide / GenBank** sin salir de la app.
    Necesitas proporcionar tu email (requerido por NCBI para la API Entrez).
    """)

    if not BIOPYTHON_OK:
        st.error("BioPython es necesario para usar Entrez.")
        st.stop()

    from Bio import Entrez

    # ── Configuración Entrez ──────────────────────────────────────────────────
    with st.expander("⚙️ Configuración Entrez (NCBI)", expanded=True):
        col_email, col_api = st.columns(2)
        with col_email:
            entrez_email = st.text_input(
                "📧 Tu email (requerido por NCBI)",
                placeholder="investigador@universidad.edu",
                help="NCBI requiere un email válido para el uso de su API."
            )
        with col_api:
            entrez_apikey = st.text_input(
                "🔑 API Key de NCBI (opcional, aumenta límite a 10 req/s)",
                type="password",
                placeholder="xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",
                help="Obtén tu API key en: https://www.ncbi.nlm.nih.gov/account/"
            )
        if entrez_email:
            Entrez.email = entrez_email
        if entrez_apikey:
            Entrez.api_key = entrez_apikey

    st.markdown("---")

    # ── Búsqueda ──────────────────────────────────────────────────────────────
    st.markdown("### 🔎 Búsqueda de secuencias")

    col_q, col_db = st.columns([3, 1])
    with col_q:
        search_query = st.text_input(
            "Término de búsqueda",
            placeholder='SARS-CoV-2[Organism] AND spike[Gene] AND 2023[PDAT]',
            help="Usa la sintaxis de NCBI. Ejemplos:\n"
                 "- `SARS-CoV-2[Organism] AND complete genome[Title]`\n"
                 "- `influenza A virus H1N1[Organism] AND HA[Gene]`\n"
                 "- `NC_045512` (accession directo)"
        )
    with col_db:
        ncbi_db = st.selectbox("Base de datos", ["nucleotide", "protein", "gene"])

    col_maxr, col_sort, col_minlen, col_maxlen = st.columns(4)
    with col_maxr:
        max_results = st.number_input("Máx. resultados", 5, 500, 20, 5)
    with col_sort:
        sort_by = st.selectbox("Ordenar por", ["relevance", "pub_date", "Length"])
    with col_minlen:
        min_len = st.number_input("Long. mínima (bp)", 0, 1000000, 0, 100)
    with col_maxlen:
        max_len = st.number_input("Long. máxima (bp)", 0, 10000000, 0, 1000,
                                   help="0 = sin límite superior")

    # Añadir filtro de longitud a la query
    def build_ncbi_query(base_query, min_l, max_l):
        q = base_query.strip()
        if min_l > 0 or max_l > 0:
            lo = min_l if min_l > 0 else 1
            hi = max_l if max_l > 0 else 99999999
            q += f" AND {lo}:{hi}[SLEN]"
        return q

    col_search_btn, col_accession_btn = st.columns([1, 1])

    with col_search_btn:
        do_search = st.button("🔍 Buscar en NCBI", type="primary", key="ncbi_search")
    with col_accession_btn:
        accession_input = st.text_input(
            "O descarga por accession directo (separados por coma)",
            placeholder="NC_045512.2, MN908947.3, AY278741.1"
        )
        do_accession = st.button("⬇️ Descargar por accession", key="ncbi_accession")

    # ── Ejecutar búsqueda ─────────────────────────────────────────────────────
    if do_search:
        if not entrez_email:
            st.error("⚠️ Debes introducir tu email antes de buscar.")
        elif not search_query.strip():
            st.error("⚠️ Escribe un término de búsqueda.")
        else:
            with st.spinner(f"Buscando en NCBI ({ncbi_db})..."):
                try:
                    final_query = build_ncbi_query(search_query, min_len, max_len)
                    handle = Entrez.esearch(
                        db=ncbi_db,
                        term=final_query,
                        retmax=max_results,
                        sort=sort_by,
                        usehistory="y"
                    )
                    search_record = Entrez.read(handle)
                    handle.close()

                    total_found = int(search_record["Count"])
                    ids = search_record["IdList"]

                    if not ids:
                        st.warning(f"No se encontraron resultados para: `{final_query}`")
                        st.session_state.ncbi_results = []
                    else:
                        st.info(f"🔢 **{total_found}** registros totales encontrados. Obteniendo resumen de {len(ids)}...")

                        # Obtener resumen (esummary) para mostrar tabla
                        handle = Entrez.esummary(db=ncbi_db, id=",".join(ids), retmode="xml")
                        summaries = Entrez.read(handle)
                        handle.close()

                        rows = []
                        for s in summaries:
                            try:
                                rows.append({
                                    "✅ Seleccionar": False,
                                    "Accession": s.get("Caption", s.get("AccessionVersion", "N/A")),
                                    "Título": s.get("Title", "N/A")[:80],
                                    "Organismo": s.get("Organism", "N/A"),
                                    "Longitud (bp)": int(s.get("Length", 0)),
                                    "Fecha": s.get("CreateDate", s.get("UpdateDate", "N/A")),
                                    "UID": str(s.get("Id", s.get("Gi", ""))),
                                })
                            except Exception:
                                pass

                        st.session_state.ncbi_results = rows
                        st.success(f"✅ Resumen de {len(rows)} secuencias obtenido.")

                except Exception as e:
                    st.error(f"❌ Error en búsqueda NCBI: {e}")
                    st.caption("Comprueba tu conexión a internet y que el email sea válido.")

    # ── Descarga por accession directo ────────────────────────────────────────
    if do_accession:
        if not entrez_email:
            st.error("⚠️ Introduce tu email primero.")
        elif not accession_input.strip():
            st.error("⚠️ Introduce al menos un accession.")
        else:
            accessions = [a.strip() for a in accession_input.split(",") if a.strip()]
            with st.spinner(f"Descargando {len(accessions)} secuencia(s) de GenBank..."):
                try:
                    handle = Entrez.efetch(
                        db="nucleotide",
                        id=",".join(accessions),
                        rettype="fasta",
                        retmode="text"
                    )
                    fasta_text = handle.read()
                    handle.close()

                    new_records = parse_fasta_text(fasta_text)
                    if new_records:
                        st.session_state.records.extend(new_records)
                        st.success(f"✅ {len(new_records)} secuencias descargadas y añadidas al análisis.")
                        st.balloons()
                    else:
                        st.error("No se obtuvieron secuencias válidas.")
                except Exception as e:
                    st.error(f"❌ Error descargando por accession: {e}")

    # ── Mostrar resultados y selección ────────────────────────────────────────
    if st.session_state.ncbi_results:
        st.markdown("---")
        st.markdown(f"### 📋 Resultados de búsqueda ({len(st.session_state.ncbi_results)} secuencias)")

        results_df = pd.DataFrame(st.session_state.ncbi_results)

        # Editor interactivo con checkboxes
        edited_df = st.data_editor(
            results_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "✅ Seleccionar": st.column_config.CheckboxColumn(
                    "Seleccionar", help="Marca las secuencias que quieres descargar", default=False
                ),
                "Longitud (bp)": st.column_config.NumberColumn("Longitud (bp)", format="%d"),
                "UID": None,  # ocultar
            },
            disabled=["Accession", "Título", "Organismo", "Longitud (bp)", "Fecha"],
            key="ncbi_editor"
        )

        selected_rows = edited_df[edited_df["✅ Seleccionar"] == True]
        n_selected = len(selected_rows)

        col_sel_info, col_dl_btn = st.columns([2, 1])
        with col_sel_info:
            if n_selected > 0:
                st.info(f"✅ **{n_selected}** secuencias seleccionadas · "
                        f"Longitud total estimada: {selected_rows['Longitud (bp)'].sum():,} bp")
            else:
                st.caption("☝️ Marca las secuencias que deseas descargar")

        with col_dl_btn:
            if n_selected > 0:
                if st.button(f"⬇️ Descargar {n_selected} secuencias seleccionadas",
                             type="primary", key="dl_selected"):
                    if not entrez_email:
                        st.error("Introduce tu email primero.")
                    else:
                        uids = selected_rows["UID"].tolist()
                        with st.spinner(f"Descargando {len(uids)} secuencias de GenBank..."):
                            try:
                                # Descargar en lotes de 20
                                all_new = []
                                batch_size = 20
                                progress = st.progress(0)
                                for i in range(0, len(uids), batch_size):
                                    batch = uids[i:i + batch_size]
                                    handle = Entrez.efetch(
                                        db=ncbi_db,
                                        id=",".join(batch),
                                        rettype="fasta",
                                        retmode="text"
                                    )
                                    fasta_batch = handle.read()
                                    handle.close()
                                    recs = parse_fasta_text(fasta_batch)
                                    all_new.extend(recs)
                                    progress.progress(min((i + batch_size) / len(uids), 1.0))
                                    time.sleep(0.35)  # respetar límite NCBI

                                if all_new:
                                    st.session_state.records.extend(all_new)
                                    st.session_state.alignment = None
                                    st.session_state.conservation_df = None
                                    st.session_state.tree = None
                                    st.success(f"✅ {len(all_new)} secuencias añadidas al análisis. "
                                               f"Ve a **📂 Carga de Secuencias** para verificarlas.")
                                    st.balloons()
                                else:
                                    st.error("No se obtuvieron secuencias válidas.")
                            except Exception as e:
                                st.error(f"❌ Error descargando: {e}")

        # Histograma de longitudes de resultados
        if PLOTLY_OK and len(results_df) > 1:
            st.markdown("### 📊 Distribución de longitudes en resultados")
            fig_len = px.histogram(
                results_df, x="Longitud (bp)", nbins=25,
                color_discrete_sequence=["#2563eb"],
                title="Distribución de longitudes en resultados NCBI"
            )
            fig_len.update_layout(
                paper_bgcolor="#ffffff", plot_bgcolor="#f8fafc",
                font=dict(color="#1e293b"), height=300,
            )
            fig_len.update_xaxes(gridcolor="#30363d")
            fig_len.update_yaxes(gridcolor="#30363d")
            st.plotly_chart(fig_len, use_container_width=True)

    # ── Tips de búsqueda ─────────────────────────────────────────────────────
    with st.expander("💡 Consejos de búsqueda en NCBI"):
        st.markdown("""
        **Operadores de búsqueda NCBI:**
        
        | Filtro | Sintaxis | Ejemplo |
        |--------|----------|---------|
        | Organismo | `[Organism]` | `SARS-CoV-2[Organism]` |
        | Gen | `[Gene]` | `spike[Gene]` |
        | Fecha publicación | `[PDAT]` | `2023/01/01:2024/01/01[PDAT]` |
        | Longitud | `[SLEN]` | `1000:30000[SLEN]` |
        | Título | `[Title]` | `complete genome[Title]` |
        | País | `[Country]` | `Mexico[Country]` |
        | Tipo molécula | `[Molecule Type]` | `RNA[Molecule Type]` |
        
        **Ejemplos completos:**
        ```
        SARS-CoV-2[Organism] AND spike[Gene] AND 2023[PDAT]
        influenza A virus[Organism] AND complete genome[Title] AND H1N1[Title]
        dengue virus[Organism] AND envelope[Gene] AND Mexico[Country]
        monkeypox virus[Organism] AND 2022[PDAT] AND complete genome[Title]
        ```
        
        **Nota:** Sin API key, NCBI permite 3 peticiones/segundo. Con API key, 10/segundo.
        Obtén tu API key gratis en: https://www.ncbi.nlm.nih.gov/account/
        """)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 8 · PIPELINE AUTOMÁTICO + PDF
# ══════════════════════════════════════════════════════════════════════════════
with tab8:
    st.markdown("## ⚡ Pipeline Automático de Análisis + Informe PDF")
    st.markdown("""
    Ejecuta **todo el flujo de análisis en un solo clic** y genera un informe PDF científico 
    completo listo para compartir con tu equipo o adjuntar a una publicación.
    """)

    if not st.session_state.records:
        st.warning("⚠️ Primero carga secuencias (pestaña **📂 Carga**) o búscalas en **🌐 NCBI**.")
        st.stop()

    records = st.session_state.records

    # ── Configuración del pipeline ────────────────────────────────────────────
    st.markdown("### ⚙️ Configuración del pipeline")

    col_p1, col_p2, col_p3 = st.columns(3)
    with col_p1:
        st.markdown("**🔗 Alineamiento**")
        pipe_aligner = st.selectbox(
            "Herramienta", ["Interno (BioPython)", "MUSCLE", "MAFFT"],
            key="pipe_aligner"
        )
        pipe_tree_method = st.selectbox(
            "Árbol filogenético", ["Neighbor-Joining (NJ)", "UPGMA"],
            key="pipe_tree"
        )

    with col_p2:
        st.markdown("**📄 Informe PDF**")
        report_title = st.text_input("Título del informe",
                                      value="Análisis Filogenético Viral",
                                      key="report_title")
        report_author = st.text_input("Autor(es)",
                                       placeholder="Dr. García et al.",
                                       key="report_author")
        report_institution = st.text_input("Institución",
                                            placeholder="Universidad Nacional de Virología",
                                            key="report_inst")

    with col_p3:
        st.markdown("**🔬 Módulos a incluir**")
        run_alignment   = st.checkbox("Alineamiento múltiple", value=True)
        run_conservation = st.checkbox("Análisis de conservación", value=True)
        run_tree        = st.checkbox("Árbol filogenético", value=True)
        run_epitopes    = st.checkbox("Predicción de epítopos", value=True)
        run_dashboard   = st.checkbox("Dashboard de composición", value=True)

    # Seleccionar secuencia para epítopos
    if run_epitopes:
        pipe_epitope_seq = st.selectbox(
            "Secuencia para análisis de epítopos",
            [r.id for r in records],
            key="pipe_epi_seq"
        )
        pipe_epitope_type = st.radio("Tipo de secuencia",
                                      ["Proteína (aa)", "Nucleótido (traducir)"],
                                      horizontal=True, key="pipe_epi_type")

    st.markdown("---")

    # ── Botón de inicio ───────────────────────────────────────────────────────
    col_run, col_status = st.columns([1, 2])
    with col_run:
        run_pipeline = st.button("⚡ EJECUTAR PIPELINE COMPLETO",
                                  type="primary", use_container_width=True,
                                  key="run_pipeline")

    # ── LOG en vivo ───────────────────────────────────────────────────────────
    log_placeholder = st.empty()

    def log(msg: str, level: str = "info"):
        ts = datetime.datetime.now().strftime("%H:%M:%S")
        icon = {"info": "ℹ️", "ok": "✅", "warn": "⚠️", "err": "❌", "run": "⏳"}
        entry = f"`{ts}` {icon.get(level,'ℹ️')} {msg}"
        st.session_state.pipeline_log.append(entry)
        log_placeholder.markdown("\n\n".join(st.session_state.pipeline_log[-18:]))

    # ══════════════════════════════════════════════════════════════════════════
    # FUNCIÓN GENERADORA DE PDF
    # ══════════════════════════════════════════════════════════════════════════
    def generate_pdf_report(
        title: str, author: str, institution: str,
        records: list, summary_df: pd.DataFrame,
        alignment=None, conservation_df=None,
        tree=None, dist_matrix=None, dist_labels=None,
        epitope_data: dict = None,
    ) -> bytes:
        """Genera el informe PDF completo con ReportLab."""

        buf = io.BytesIO()
        doc = SimpleDocTemplate(
            buf, pagesize=A4,
            leftMargin=2*cm, rightMargin=2*cm,
            topMargin=2.5*cm, bottomMargin=2*cm,
        )

        # ── Estilos ──────────────────────────────────────────────────────────
        styles = getSampleStyleSheet()

        S_title = ParagraphStyle("vtitle",
            fontSize=22, fontName="Helvetica-Bold",
            textColor=rl_colors.HexColor("#1a6fe8"),
            alignment=TA_CENTER, spaceAfter=6)
        S_subtitle = ParagraphStyle("vsub",
            fontSize=12, fontName="Helvetica",
            textColor=rl_colors.HexColor("#444444"),
            alignment=TA_CENTER, spaceAfter=4)
        S_h1 = ParagraphStyle("vh1",
            fontSize=15, fontName="Helvetica-Bold",
            textColor=rl_colors.HexColor("#1a6fe8"),
            spaceBefore=14, spaceAfter=6,
            borderPad=4)
        S_h2 = ParagraphStyle("vh2",
            fontSize=12, fontName="Helvetica-Bold",
            textColor=rl_colors.HexColor("#333333"),
            spaceBefore=8, spaceAfter=4)
        S_body = ParagraphStyle("vbody",
            fontSize=9, fontName="Helvetica",
            textColor=rl_colors.HexColor("#222222"),
            leading=14, spaceAfter=4)
        S_caption = ParagraphStyle("vcaption",
            fontSize=8, fontName="Helvetica-Oblique",
            textColor=rl_colors.HexColor("#666666"),
            alignment=TA_CENTER, spaceAfter=8)
        S_code = ParagraphStyle("vcode",
            fontSize=7.5, fontName="Courier",
            textColor=rl_colors.HexColor("#333333"),
            backColor=rl_colors.HexColor("#f5f5f5"),
            leading=11, spaceAfter=4, leftIndent=8)

        # ── Estilos de tabla ─────────────────────────────────────────────────
        TABLE_HEADER = rl_colors.HexColor("#1a6fe8")
        TABLE_ODD    = rl_colors.HexColor("#f0f4ff")
        TABLE_EVEN   = rl_colors.white
        TABLE_GRID   = rl_colors.HexColor("#c0ccdd")

        def make_table(data: list[list], col_widths=None, header=True) -> Table:
            t = Table(data, colWidths=col_widths, repeatRows=1 if header else 0)
            n_rows = len(data)
            style_cmds = [
                ("FONTNAME",    (0, 0), (-1, 0 if header else -1), "Helvetica-Bold"),
                ("FONTSIZE",    (0, 0), (-1, -1), 8),
                ("FONTNAME",    (0, 1 if header else 0), (-1, -1), "Helvetica"),
                ("BACKGROUND",  (0, 0), (-1, 0), TABLE_HEADER if header else TABLE_ODD),
                ("TEXTCOLOR",   (0, 0), (-1, 0), rl_colors.white if header else rl_colors.black),
                ("ALIGN",       (0, 0), (-1, -1), "CENTER"),
                ("ALIGN",       (1, 1), (1, -1), "LEFT"),
                ("VALIGN",      (0, 0), (-1, -1), "MIDDLE"),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [TABLE_ODD, TABLE_EVEN]),
                ("GRID",        (0, 0), (-1, -1), 0.4, TABLE_GRID),
                ("TOPPADDING",  (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                ("ROUNDEDCORNERS", [4]),
            ]
            t.setStyle(TableStyle(style_cmds))
            return t

        def fig_to_rl_image(fig_mpl, width_cm: float = 15) -> RLImage | None:
            """Convierte figura matplotlib a imagen ReportLab."""
            try:
                buf_img = io.BytesIO()
                fig_mpl.savefig(buf_img, format="png", dpi=180, bbox_inches="tight",
                                facecolor="white", edgecolor="none")
                buf_img.seek(0)
                img = RLImage(buf_img)
                aspect = img.imageHeight / img.imageWidth
                img.drawWidth  = width_cm * cm
                img.drawHeight = width_cm * cm * aspect
                return img
            except Exception:
                return None

        # ── Construir contenido ───────────────────────────────────────────────
        story = []
        today = datetime.date.today().strftime("%d de %B de %Y")

        # PORTADA
        story.append(Spacer(1, 1.5*cm))
        story.append(Paragraph("🧬 ViroSeq Analyzer", S_title))
        story.append(Paragraph(title, ParagraphStyle("vtitle2",
            fontSize=17, fontName="Helvetica-Bold",
            textColor=rl_colors.HexColor("#333333"),
            alignment=TA_CENTER, spaceAfter=10)))
        story.append(HRFlowable(width="100%", thickness=2,
                                color=rl_colors.HexColor("#1a6fe8"), spaceAfter=10))
        if author:
            story.append(Paragraph(f"<b>Autores:</b> {author}", S_subtitle))
        if institution:
            story.append(Paragraph(f"<b>Institución:</b> {institution}", S_subtitle))
        story.append(Paragraph(f"<b>Fecha:</b> {today}", S_subtitle))
        story.append(Spacer(1, 0.5*cm))

        # Resumen ejecutivo en caja
        n_seqs = len(records)
        mean_len = summary_df["Longitud (bp)"].mean()
        mean_gc  = summary_df["GC (%)"].mean()
        exec_summary = [
            ["Parámetro", "Valor"],
            ["Total de secuencias", str(n_seqs)],
            ["Longitud media (bp)", f"{mean_len:.0f}"],
            ["GC% medio", f"{mean_gc:.1f}%"],
            ["Longitud mínima (bp)", str(summary_df["Longitud (bp)"].min())],
            ["Longitud máxima (bp)", str(summary_df["Longitud (bp)"].max())],
            ["Análisis realizado", today],
            ["Herramienta", "ViroSeq Analyzer v1.0"],
        ]
        story.append(make_table(exec_summary, col_widths=[8*cm, 8*cm]))
        story.append(PageBreak())

        # 1. SECUENCIAS
        story.append(Paragraph("1. Secuencias Analizadas", S_h1))
        story.append(HRFlowable(width="100%", thickness=0.5,
                                color=rl_colors.HexColor("#c0ccdd"), spaceAfter=6))
        story.append(Paragraph(
            f"Se analizaron <b>{n_seqs}</b> secuencias con una longitud media de "
            f"<b>{mean_len:.0f} bp</b> y un contenido GC medio de <b>{mean_gc:.1f}%</b>.",
            S_body))

        # Tabla de secuencias (primeras 30)
        seq_table_data = [["#", "ID", "Longitud (bp)", "GC (%)", "A%", "T%", "G%", "C%"]]
        for i, row in summary_df.head(30).iterrows():
            seq_table_data.append([
                str(i+1),
                str(row["ID"])[:35],
                str(row["Longitud (bp)"]),
                f"{row['GC (%)']:.1f}",
                f"{row['A (%)']:.1f}",
                f"{row['T (%)']:.1f}",
                f"{row['G (%)']:.1f}",
                f"{row['C (%)']:.1f}",
            ])
        if len(summary_df) > 30:
            seq_table_data.append(["...", f"... y {len(summary_df)-30} más", "", "", "", "", "", ""])

        story.append(make_table(seq_table_data,
                                col_widths=[0.8*cm, 5.8*cm, 2.2*cm, 1.6*cm,
                                            1.2*cm, 1.2*cm, 1.2*cm, 1.2*cm]))

        # Gráfico composición nucleotídica
        if MATPLOTLIB_OK and len(records) > 0:
            try:
                fig_comp, ax = plt.subplots(figsize=(9, 3.5))
                ax.set_facecolor("white")
                fig_comp.patch.set_facecolor("white")
                x = range(len(summary_df))
                labels_x = [r[:18] for r in summary_df["ID"]]
                bar_w = 0.2
                colors_comp = ["#3fb950", "#f85149", "#d29922", "#58a6ff"]
                for j, (col, clr) in enumerate(zip(["A (%)", "T (%)", "G (%)", "C (%)"], colors_comp)):
                    ax.bar([xi + j*bar_w for xi in x], summary_df[col], bar_w,
                           label=col[0], color=clr, alpha=0.85)
                ax.set_xticks([xi + 1.5*bar_w for xi in x])
                ax.set_xticklabels(labels_x, rotation=45, ha="right", fontsize=7)
                ax.set_ylabel("Frecuencia (%)", fontsize=8)
                ax.set_title("Composición nucleotídica por secuencia", fontsize=10, fontweight="bold")
                ax.legend(fontsize=8, ncol=4)
                ax.spines[["top","right"]].set_visible(False)
                plt.tight_layout()
                img = fig_to_rl_image(fig_comp, 15)
                if img:
                    story.append(Spacer(1, 0.3*cm))
                    story.append(img)
                    story.append(Paragraph("Figura 1. Composición nucleotídica (A, T, G, C) de cada secuencia analizada.", S_caption))
                plt.close(fig_comp)
            except Exception:
                pass

        story.append(PageBreak())

        # 2. ALINEAMIENTO
        if alignment is not None:
            story.append(Paragraph("2. Alineamiento Múltiple de Secuencias", S_h1))
            story.append(HRFlowable(width="100%", thickness=0.5,
                                    color=rl_colors.HexColor("#c0ccdd"), spaceAfter=6))
            aln_len = alignment.get_alignment_length()
            story.append(Paragraph(
                f"Se realizó el alineamiento múltiple de <b>{len(alignment)}</b> secuencias, "
                f"resultando en una longitud de alineamiento de <b>{aln_len:,} columnas</b>.",
                S_body))

            # Fragmento del alineamiento (texto)
            story.append(Paragraph("Fragmento del alineamiento (primeras 80 columnas):", S_h2))
            aln_fragment = ""
            for rec in alignment:
                aln_fragment += f"{rec.id[:20]:<22}{str(rec.seq)[:80]}\n"
            story.append(Paragraph(aln_fragment.replace("\n", "<br/>"), S_code))
            story.append(PageBreak())

        # 3. CONSERVACIÓN
        if conservation_df is not None:
            story.append(Paragraph("3. Análisis de Conservación y Mutaciones", S_h1))
            story.append(HRFlowable(width="100%", thickness=0.5,
                                    color=rl_colors.HexColor("#c0ccdd"), spaceAfter=6))

            n_cons  = (conservation_df["Conservación (%)"] == 100).sum()
            n_var   = (conservation_df["Conservación (%)"] < 100).sum()
            n_hvar  = (conservation_df["Conservación (%)"] < 80).sum()
            mean_c  = conservation_df["Conservación (%)"].mean()

            story.append(Paragraph(
                f"El análisis de conservación identificó <b>{n_cons}</b> posiciones "
                f"completamente conservadas (100%) y <b>{n_var}</b> posiciones variables, "
                f"de las cuales <b>{n_hvar}</b> presentan alta variabilidad (&lt;80%). "
                f"La conservación media fue de <b>{mean_c:.1f}%</b>.",
                S_body))

            # Gráfico conservación
            if MATPLOTLIB_OK:
                try:
                    fig_cons, axes = plt.subplots(2, 1, figsize=(10, 5), sharex=True)
                    fig_cons.patch.set_facecolor("white")

                    ax1, ax2 = axes
                    ax1.set_facecolor("white")
                    ax2.set_facecolor("white")

                    pos = conservation_df["Posición"]
                    cons = conservation_df["Conservación (%)"]
                    entr = conservation_df["Entropía (bits)"]

                    ax1.fill_between(pos, cons, alpha=0.4, color="#1a6fe8")
                    ax1.plot(pos, cons, color="#1a6fe8", linewidth=0.8)
                    ax1.axhline(100, ls="--", color="#d29922", linewidth=0.8, label="100%")
                    ax1.axhline(80, ls=":", color="#f85149", linewidth=0.8, label="80%")
                    ax1.set_ylabel("Conservación (%)", fontsize=8)
                    ax1.set_title("Conservación por posición", fontsize=9, fontweight="bold")
                    ax1.legend(fontsize=7)
                    ax1.spines[["top","right"]].set_visible(False)

                    ax2.bar(pos, entr, color="#f85149", alpha=0.7, width=1.0)
                    ax2.set_ylabel("Entropía (bits)", fontsize=8)
                    ax2.set_xlabel("Posición (bp)", fontsize=8)
                    ax2.set_title("Entropía de Shannon (variabilidad)", fontsize=9, fontweight="bold")
                    ax2.spines[["top","right"]].set_visible(False)

                    plt.tight_layout()
                    img = fig_to_rl_image(fig_cons, 15)
                    if img:
                        story.append(img)
                        story.append(Paragraph(
                            "Figura 2. Conservación por posición (arriba) y entropía de Shannon (abajo). "
                            "Valores de entropía cercanos a 0 indican alta conservación.",
                            S_caption))
                    plt.close(fig_cons)
                except Exception:
                    pass

            # Top 15 posiciones más variables
            top_var = conservation_df.nsmallest(15, "Conservación (%)")[
                ["Posición", "Conservación (%)", "Entropía (bits)", "NT más frecuente", "Variantes"]]
            story.append(Paragraph("Top 15 posiciones con mayor variabilidad:", S_h2))
            var_data = [list(top_var.columns)] + [[str(v) for v in row] for row in top_var.values]
            story.append(make_table(var_data,
                                    col_widths=[2.5*cm, 3.5*cm, 3.5*cm, 3.5*cm, 2.5*cm]))
            story.append(PageBreak())

        # 4. ÁRBOL FILOGENÉTICO
        if tree is not None:
            story.append(Paragraph("4. Árbol Filogenético", S_h1))
            story.append(HRFlowable(width="100%", thickness=0.5,
                                    color=rl_colors.HexColor("#c0ccdd"), spaceAfter=6))
            story.append(Paragraph(
                "Se construyó el árbol filogenético usando el método Neighbor-Joining "
                "con distancias p calculadas a partir del alineamiento múltiple.", S_body))

            # Árbol con matplotlib/Bio.Phylo
            if MATPLOTLIB_OK:
                try:
                    n_tips = len(tree.get_terminals())
                    fig_h = max(5, n_tips * 0.45)
                    fig_tree, ax_t = plt.subplots(figsize=(10, fig_h))
                    ax_t.set_facecolor("white")
                    fig_tree.patch.set_facecolor("white")
                    Phylo.draw(tree, axes=ax_t, do_show=False)
                    ax_t.set_title("Árbol filogenético (Neighbor-Joining)", fontsize=10, fontweight="bold")
                    for spine in ax_t.spines.values():
                        spine.set_edgecolor("#cccccc")
                    plt.tight_layout()
                    img_tree = fig_to_rl_image(fig_tree, 15)
                    if img_tree:
                        story.append(img_tree)
                        story.append(Paragraph(
                            "Figura 3. Árbol filogenético Neighbor-Joining calculado con distancias p "
                            "del alineamiento múltiple. La longitud de las ramas es proporcional "
                            "a la distancia evolutiva.",
                            S_caption))
                    plt.close(fig_tree)
                except Exception as ex:
                    story.append(Paragraph(f"[Visualización del árbol no disponible: {ex}]", S_body))

            # Formato Newick
            newick_str = tree_to_newick(tree)
            story.append(Paragraph("Representación Newick del árbol:", S_h2))
            newick_wrapped = textwrap.fill(newick_str, 90)
            story.append(Paragraph(newick_wrapped, S_code))

            # Matriz de distancias
            if dist_matrix is not None and dist_labels:
                story.append(Paragraph("Matriz de distancias genéticas (p-distance):", S_h2))
                if MATPLOTLIB_OK:
                    try:
                        fig_dm, ax_dm = plt.subplots(figsize=(8, 6))
                        fig_dm.patch.set_facecolor("white")
                        short_labels = [l[:20] for l in dist_labels]
                        im = ax_dm.imshow(dist_matrix, cmap="YlOrRd", aspect="auto")
                        ax_dm.set_xticks(range(len(short_labels)))
                        ax_dm.set_yticks(range(len(short_labels)))
                        ax_dm.set_xticklabels(short_labels, rotation=45, ha="right", fontsize=7)
                        ax_dm.set_yticklabels(short_labels, fontsize=7)
                        plt.colorbar(im, ax=ax_dm, shrink=0.8).set_label("Distancia p", fontsize=8)
                        ax_dm.set_title("Heatmap de distancias genéticas", fontsize=10, fontweight="bold")
                        # Anotaciones numéricas
                        for i in range(len(dist_labels)):
                            for j in range(len(dist_labels)):
                                ax_dm.text(j, i, f"{dist_matrix[i,j]:.2f}",
                                           ha="center", va="center", fontsize=5.5,
                                           color="white" if dist_matrix[i,j] > 0.3 else "black")
                        plt.tight_layout()
                        img_dm = fig_to_rl_image(fig_dm, 13)
                        if img_dm:
                            story.append(img_dm)
                            story.append(Paragraph(
                                "Figura 4. Mapa de calor de distancias genéticas p entre todas las secuencias. "
                                "Valores más altos (colores más oscuros) indican mayor divergencia.",
                                S_caption))
                        plt.close(fig_dm)
                    except Exception:
                        pass
            story.append(PageBreak())

        # 5. EPÍTOPOS
        if epitope_data:
            story.append(Paragraph("5. Predicción de Epítopos", S_h1))
            story.append(HRFlowable(width="100%", thickness=0.5,
                                    color=rl_colors.HexColor("#c0ccdd"), spaceAfter=6))
            story.append(Paragraph(
                f"Se realizó el análisis de antigenicidad sobre la secuencia "
                f"<b>{epitope_data.get('seq_id','N/A')}</b> "
                f"({epitope_data.get('length', 0)} residuos) usando los índices de "
                f"Kyte-Doolittle (hidropatía, ventana=9) y Parker (antigenicidad, ventana=7).",
                S_body))

            if MATPLOTLIB_OK and "kd_scores" in epitope_data and "pk_scores" in epitope_data:
                try:
                    kd_s = epitope_data["kd_scores"]
                    pk_s = epitope_data["pk_scores"]
                    pos_e = list(range(1, len(kd_s)+1))

                    fig_ep, axes_e = plt.subplots(2, 1, figsize=(10, 5), sharex=True)
                    fig_ep.patch.set_facecolor("white")

                    ax_kd, ax_pk = axes_e
                    ax_kd.set_facecolor("white")
                    ax_pk.set_facecolor("white")

                    ax_kd.fill_between(pos_e, kd_s, alpha=0.35, color="#3b82f6")
                    ax_kd.plot(pos_e, kd_s, color="#3b82f6", linewidth=0.8)
                    ax_kd.axhline(0, ls="-", color="#aaaaaa", linewidth=0.6)
                    ax_kd.axhline(1.6, ls="--", color="#d29922", linewidth=0.8, label="Umbral TM (1.6)")
                    ax_kd.set_ylabel("Hidropatía (KD)", fontsize=8)
                    ax_kd.set_title("Índice de Kyte-Doolittle (hidropatía)", fontsize=9, fontweight="bold")
                    ax_kd.legend(fontsize=7)
                    ax_kd.spines[["top","right"]].set_visible(False)

                    thresh = epitope_data.get("threshold", 0.5)
                    bar_colors = ["#22c55e" if v >= thresh else "#94a3b8" for v in pk_s]
                    ax_pk.bar(pos_e, pk_s, color=bar_colors, alpha=0.8, width=1.0)
                    ax_pk.axhline(thresh, ls="--", color="#f85149", linewidth=0.8,
                                  label=f"Umbral ({thresh})")
                    ax_pk.set_ylabel("Antigenicidad (Parker)", fontsize=8)
                    ax_pk.set_xlabel("Posición (aa)", fontsize=8)
                    ax_pk.set_title("Índice de antigenicidad Parker", fontsize=9, fontweight="bold")
                    ax_pk.legend(fontsize=7)
                    ax_pk.spines[["top","right"]].set_visible(False)

                    plt.tight_layout()
                    img_ep = fig_to_rl_image(fig_ep, 15)
                    if img_ep:
                        story.append(img_ep)
                        story.append(Paragraph(
                            "Figura 5. Análisis de propiedades fisicoquímicas. Arriba: hidropatía Kyte-Doolittle "
                            "(valores positivos = regiones hidrofóbicas; negativo = hidrofílicas). "
                            "Abajo: antigenicidad Parker (verde = regiones por encima del umbral, candidatas a epítopos).",
                            S_caption))
                    plt.close(fig_ep)
                except Exception:
                    pass

            epitopes_list = epitope_data.get("epitopes", [])
            story.append(Paragraph(
                f"<b>{len(epitopes_list)}</b> regiones antigénicas potenciales identificadas "
                f"(umbral Parker ≥ {epitope_data.get('threshold', 0.5)}, longitud mínima 5 aa):",
                S_body))

            if epitopes_list:
                ep_table = [["Inicio", "Fin", "Longitud", "Parker medio", "KD medio", "Secuencia"]]
                for ep in epitopes_list[:20]:
                    ep_table.append([
                        str(ep["Inicio"]), str(ep["Fin"]),
                        str(ep["Longitud"]),
                        f"{ep['Parker medio']:.3f}",
                        f"{ep['KD medio']:.3f}",
                        ep["Secuencia"][:30] + ("..." if len(ep["Secuencia"]) > 30 else ""),
                    ])
                story.append(make_table(ep_table,
                                        col_widths=[1.5*cm, 1.5*cm, 2*cm, 2.5*cm, 2.5*cm, 5*cm]))
            else:
                story.append(Paragraph("No se identificaron regiones antigénicas con los parámetros actuales.", S_body))
            story.append(PageBreak())

        # MÉTODOS Y REFERENCIAS
        story.append(Paragraph("6. Materiales y Métodos", S_h1))
        story.append(HRFlowable(width="100%", thickness=0.5,
                                color=rl_colors.HexColor("#c0ccdd"), spaceAfter=6))
        methods = """
        <b>Carga y procesamiento de secuencias:</b> Las secuencias fueron cargadas en formato FASTA 
        y procesadas utilizando BioPython v1.83+. El contenido GC y la composición nucleotídica 
        fueron calculados para cada secuencia.<br/><br/>
        <b>Alineamiento múltiple:</b> Las secuencias fueron alineadas utilizando alineamiento 
        interno (padding) o herramientas externas (MUSCLE/MAFFT) según disponibilidad.<br/><br/>
        <b>Análisis de conservación:</b> Se calculó la conservación por posición como la frecuencia 
        del nucleótido/aminoácido más común en cada columna del alineamiento. La variabilidad fue 
        cuantificada mediante la entropía de Shannon.<br/><br/>
        <b>Filogenia:</b> El árbol filogenético fue construido usando el método Neighbor-Joining 
        con distancias p calculadas a partir del alineamiento múltiple (Saitou & Nei, 1987).<br/><br/>
        <b>Predicción de epítopos:</b> Se utilizaron los índices de Kyte-Doolittle para hidropatía 
        (Kyte & Doolittle, 1982) y Parker para antigenicidad (Parker et al., 1986) con ventanas 
        deslizantes de 9 y 7 residuos respectivamente.
        """
        story.append(Paragraph(methods, S_body))

        story.append(Paragraph("7. Referencias", S_h1))
        story.append(HRFlowable(width="100%", thickness=0.5,
                                color=rl_colors.HexColor("#c0ccdd"), spaceAfter=6))
        refs = [
            "Cock PJA et al. (2009) Biopython: freely available Python tools for computational molecular biology. Bioinformatics 25(11):1422–1423.",
            "Kyte J & Doolittle RF (1982) A simple method for displaying the hydropathic character of a protein. J Mol Biol 157(1):105–132.",
            "Parker JM et al. (1986) New hydrophilicity scale derived from high-performance liquid chromatography peptide retention data. Biochemistry 25(19):5425–5432.",
            "Saitou N & Nei M (1987) The neighbor-joining method: a new method for reconstructing phylogenetic trees. Mol Biol Evol 4(4):406–425.",
            "Shannon CE (1948) A mathematical theory of communication. Bell System Technical Journal 27(3):379–423.",
        ]
        for i, ref in enumerate(refs, 1):
            story.append(Paragraph(f"{i}. {ref}", S_body))

        # Pie de página final
        story.append(Spacer(1, 1*cm))
        story.append(HRFlowable(width="100%", thickness=1,
                                color=rl_colors.HexColor("#1a6fe8"), spaceAfter=6))
        story.append(Paragraph(
            f"Generado por <b>ViroSeq Analyzer v1.0</b> · {today}",
            ParagraphStyle("vfooter", fontSize=8, fontName="Helvetica-Oblique",
                           textColor=rl_colors.HexColor("#888888"), alignment=TA_CENTER)))

        doc.build(story)
        buf.seek(0)
        return buf.read()

    # ══════════════════════════════════════════════════════════════════════════
    # EJECUTAR PIPELINE
    # ══════════════════════════════════════════════════════════════════════════
    if run_pipeline:
        st.session_state.pipeline_log = []
        st.session_state.pipeline_pdf = None
        start_time = time.time()

        log(f"Pipeline iniciado · {len(records)} secuencias", "run")
        progress_bar = st.progress(0)
        step = 0
        total_steps = sum([run_alignment, run_conservation, run_tree, run_epitopes, True])  # +1 PDF

        # PASO 1 · ALINEAMIENTO
        if run_alignment:
            log("Ejecutando alineamiento múltiple...", "run")
            aln = None
            if pipe_aligner == "MUSCLE":
                aln = align_with_muscle(records)
                if aln is None:
                    log("MUSCLE no disponible. Usando alineamiento interno.", "warn")
            elif pipe_aligner == "MAFFT":
                aln = align_with_mafft(records)
                if aln is None:
                    log("MAFFT no disponible. Usando alineamiento interno.", "warn")

            if aln is None:
                aln = align_sequences_biopython(records)

            if aln:
                st.session_state.alignment = aln
                log(f"Alineamiento OK · {len(aln)} seqs × {aln.get_alignment_length()} cols", "ok")
            else:
                log("Error en alineamiento. Continuando sin él.", "err")
            step += 1
            progress_bar.progress(step / total_steps)

        # PASO 2 · CONSERVACIÓN
        if run_conservation and st.session_state.alignment:
            log("Calculando conservación y entropía...", "run")
            try:
                st.session_state.conservation_df = compute_conservation(st.session_state.alignment)
                n_c = (st.session_state.conservation_df["Conservación (%)"] == 100).sum()
                log(f"Conservación OK · {n_c} posiciones 100% conservadas", "ok")
            except Exception as e:
                log(f"Error en conservación: {e}", "err")
            step += 1
            progress_bar.progress(step / total_steps)

        # PASO 3 · ÁRBOL
        if run_tree and st.session_state.alignment:
            log(f"Construyendo árbol ({pipe_tree_method})...", "run")
            try:
                if "NJ" in pipe_tree_method:
                    st.session_state.tree = build_tree_nj(st.session_state.alignment)
                else:
                    st.session_state.tree = build_tree_upgma(st.session_state.alignment)
                dm_m, dm_l = compute_distance_matrix(st.session_state.alignment)
                st.session_state.dist_matrix = dm_m
                st.session_state.dist_labels = dm_l
                log(f"Árbol OK · {len(st.session_state.tree.get_terminals())} hojas", "ok")
            except Exception as e:
                log(f"Error en árbol: {e}", "err")
            step += 1
            progress_bar.progress(step / total_steps)

        # PASO 4 · EPÍTOPOS
        epitope_data = None
        if run_epitopes:
            log(f"Calculando epítopos para: {pipe_epitope_seq}...", "run")
            try:
                sel_rec = next(r for r in records if r.id == pipe_epitope_seq)
                seq_str_e = str(sel_rec.seq).upper()

                if "Nucleótido" in pipe_epitope_type:
                    try:
                        seq_str_e = str(Seq(seq_str_e[:len(seq_str_e)//3*3]).translate(to_stop=True))
                        log(f"Traducido a proteína: {len(seq_str_e)} aa", "info")
                    except Exception as et:
                        log(f"Error traduciendo: {et}. Usando secuencia original.", "warn")

                valid_aa = set("ACDEFGHIKLMNPQRSTVWY")
                seq_clean_e = "".join(c for c in seq_str_e if c in valid_aa)

                if len(seq_clean_e) >= 10:
                    kd_s = kyte_doolittle(seq_clean_e, 9)
                    pk_s = parker_antigenicity(seq_clean_e, 7)
                    threshold_e = 0.5

                    epitopes_e = []
                    in_ep = False; ep_st = 0
                    for i, score in enumerate(pk_s):
                        if score >= threshold_e and not in_ep:
                            in_ep = True; ep_st = i
                        elif score < threshold_e and in_ep:
                            if i - ep_st >= 5:
                                epitopes_e.append({
                                    "Inicio": ep_st+1, "Fin": i, "Longitud": i-ep_st,
                                    "Secuencia": seq_clean_e[ep_st:i],
                                    "Parker medio": round(np.mean(pk_s[ep_st:i]), 3),
                                    "KD medio": round(np.mean(kd_s[ep_st:i]), 3),
                                })
                            in_ep = False

                    epitope_data = {
                        "seq_id": pipe_epitope_seq,
                        "length": len(seq_clean_e),
                        "kd_scores": kd_s,
                        "pk_scores": pk_s,
                        "threshold": threshold_e,
                        "epitopes": epitopes_e,
                    }
                    log(f"Epítopos OK · {len(epitopes_e)} regiones identificadas", "ok")
                else:
                    log("Secuencia proteica demasiado corta para epítopos.", "warn")
            except Exception as e:
                log(f"Error en epítopos: {e}", "err")
            step += 1
            progress_bar.progress(step / total_steps)

        # PASO 5 · GENERAR PDF
        log("Generando informe PDF...", "run")
        if not REPORTLAB_OK:
            log("ReportLab no instalado. PDF no disponible. Instala: pip install reportlab", "err")
        else:
            try:
                df_sum_pipe = summary_table(records)
                pdf_bytes = generate_pdf_report(
                    title=report_title,
                    author=report_author,
                    institution=report_institution,
                    records=records,
                    summary_df=df_sum_pipe,
                    alignment=st.session_state.alignment if run_alignment else None,
                    conservation_df=st.session_state.conservation_df if run_conservation else None,
                    tree=st.session_state.tree if run_tree else None,
                    dist_matrix=st.session_state.dist_matrix if run_tree else None,
                    dist_labels=st.session_state.dist_labels if run_tree else None,
                    epitope_data=epitope_data,
                )
                st.session_state.pipeline_pdf = pdf_bytes
                log(f"PDF generado · {len(pdf_bytes)/1024:.1f} KB", "ok")
            except Exception as e:
                log(f"Error generando PDF: {e}", "err")

        step += 1
        progress_bar.progress(1.0)
        elapsed = time.time() - start_time
        log(f"Pipeline completado en {elapsed:.1f} segundos 🎉", "ok")

    # ── Resultado final ───────────────────────────────────────────────────────
    if st.session_state.pipeline_log:
        st.markdown("---")
        st.markdown("### 📋 Log del pipeline")
        log_placeholder.markdown("\n\n".join(st.session_state.pipeline_log))

    if st.session_state.pipeline_pdf:
        st.markdown("---")
        st.success("✅ **Informe PDF listo para descargar**")

        col_dl_pdf, col_dl_zip = st.columns(2)
        with col_dl_pdf:
            fname_pdf = (report_title.replace(" ", "_")[:30] + "_ViroSeq.pdf"
                         if "report_title" in dir() else "informe_ViroSeq.pdf")
            st.download_button(
                "📄 Descargar informe PDF completo",
                data=st.session_state.pipeline_pdf,
                file_name=fname_pdf,
                mime="application/pdf",
                use_container_width=True,
                key="dl_pdf_final"
            )

        with col_dl_zip:
            # ZIP con todos los archivos
            import zipfile
            zip_buf = io.BytesIO()
            with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
                # PDF
                zf.writestr("informe_viroseq.pdf", st.session_state.pipeline_pdf)
                # FASTA
                zf.writestr("secuencias.fasta", records_to_fasta_bytes(records).decode())
                # Resumen CSV
                zf.writestr("resumen_secuencias.csv", df_to_csv_bytes(summary_table(records)).decode())
                # Alineamiento
                if st.session_state.alignment:
                    zf.writestr("alineamiento.fasta",
                                alignment_to_bytes(st.session_state.alignment).decode())
                # Conservación
                if st.session_state.conservation_df is not None:
                    zf.writestr("conservacion.csv",
                                df_to_csv_bytes(st.session_state.conservation_df).decode())
                # Árbol
                if st.session_state.tree:
                    zf.writestr("arbol.nwk", tree_to_newick(st.session_state.tree))
                # Distancias
                if st.session_state.dist_matrix is not None:
                    dm_df = pd.DataFrame(
                        st.session_state.dist_matrix,
                        index=st.session_state.dist_labels,
                        columns=st.session_state.dist_labels
                    )
                    zf.writestr("matriz_distancias.csv", dm_df.to_csv())

            zip_buf.seek(0)
            st.download_button(
                "📦 Descargar todo como ZIP",
                data=zip_buf.read(),
                file_name="viroseq_resultados_completos.zip",
                mime="application/zip",
                use_container_width=True,
                key="dl_zip_final"
            )

        # Vista previa de métricas del pipeline
        st.markdown("### 📊 Resumen del análisis realizado")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("🧬 Secuencias", len(records))
        c2.metric("🔗 Alineamiento",
                  f"{st.session_state.alignment.get_alignment_length()} cols"
                  if st.session_state.alignment else "—")
        c3.metric("✅ Posiciones conservadas",
                  f"{(st.session_state.conservation_df['Conservación (%)'] == 100).sum()}"
                  if st.session_state.conservation_df is not None else "—")
        c4.metric("🌳 Hojas en árbol",
                  f"{len(st.session_state.tree.get_terminals())}"
                  if st.session_state.tree else "—")

    elif not run_pipeline:
        st.markdown("---")
        st.info("""
        ### ¿Cómo funciona el pipeline?
        
        1. **Carga** secuencias (pestaña 📂) o búscalas en NCBI (pestaña 🌐)
        2. **Configura** las opciones arriba (herramienta de alineamiento, título del informe, módulos)
        3. Pulsa **⚡ EJECUTAR PIPELINE COMPLETO**
        4. El sistema ejecuta automáticamente: alineamiento → conservación → árbol → epítopos
        5. Descarga el **PDF científico** o el **ZIP completo** con todos los archivos
        
        El PDF incluye: portada, tabla de secuencias, gráficos de composición,
        visualización del alineamiento, gráficos de conservación/entropía, 
        árbol filogenético, heatmap de distancias, análisis de epítopos, 
        materiales y métodos, y referencias bibliográficas.
        """)

    if not REPORTLAB_OK:
        st.warning("""
        ⚠️ **ReportLab no instalado** — el PDF no estará disponible hasta que instales:
        ```
        pip install reportlab
        ```
        El resto del pipeline (alineamiento, árbol, conservación) sí funcionará.
        """)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 9 · IDENTIFICAR VIRUS CON BLAST
# ══════════════════════════════════════════════════════════════════════════════
with tab9:
    st.markdown("## 🔎 Identificar Virus (BLAST)")
    st.markdown("""
    Pega una secuencia desconocida y la app la enviará a **NCBI BLAST** para identificar 
    a qué virus pertenece, con porcentaje de similitud y links directos a GenBank.
    """)

    if not BIOPYTHON_OK:
        st.error("BioPython es necesario. Instala: `pip install biopython`")
        st.stop()

    from Bio import Entrez
    from Bio.Blast import NCBIWWW, NCBIXML

    # ── Configuración ─────────────────────────────────────────────────────────
    with st.expander("⚙️ Configuración (email NCBI)", expanded=False):
        blast_email = st.text_input(
            "📧 Tu email (recomendado por NCBI)",
            placeholder="investigador@universidad.edu",
            key="blast_email"
        )
        if blast_email:
            Entrez.email = blast_email

    st.markdown("---")

    # ── Entrada de secuencia ──────────────────────────────────────────────────
    st.markdown("### 🧬 Secuencia a identificar")

    col_input, col_opts = st.columns([2, 1])

    with col_input:
        input_mode = st.radio(
            "Fuente de la secuencia",
            ["✏️ Pegar secuencia", "📋 Usar secuencia ya cargada"],
            horizontal=True, key="blast_input_mode"
        )

        if "✏️" in input_mode:
            blast_seq_raw = st.text_area(
                "Pega tu secuencia aquí (con o sin cabecera FASTA)",
                height=200,
                placeholder=""">Mi_secuencia_desconocida
ATGGAGAGCCTTGTCCCTGGTTTCAACGAGAAAACACACGTCCAACTCAGT
TTGCCTGTTTTACAGGTTCGCGACGTGCTTCGCGATCGTTTGAGTTTTAGT
GAGATCGTTGAGCGGTTGATGGCTTATTTCTTTTGCGGCAATGAAACGACT""",
                key="blast_seq_input"
            )
        else:
            if not st.session_state.records:
                st.warning("No hay secuencias cargadas. Ve a 📂 Carga de Secuencias primero.")
                blast_seq_raw = ""
            else:
                selected_blast = st.selectbox(
                    "Selecciona la secuencia",
                    [r.id for r in st.session_state.records],
                    key="blast_seq_select"
                )
                sel_rec = next(r for r in st.session_state.records if r.id == selected_blast)
                blast_seq_raw = f">{sel_rec.id}\n{str(sel_rec.seq)}"
                st.code(blast_seq_raw[:200] + ("..." if len(blast_seq_raw) > 200 else ""),
                        language=None)

    with col_opts:
        st.markdown("### ⚙️ Opciones BLAST")
        blast_program = st.selectbox(
            "Programa",
            ["blastn", "blastx", "blastp"],
            help="blastn = nucleótido vs nucleótido\nblastx = nucleótido traducido vs proteína\nblatp = proteína vs proteína"
        )
        blast_db = st.selectbox(
            "Base de datos",
            ["nt", "nr", "refseq_rna", "refseq_protein"],
            help="nt = todas las secuencias nucleotídicas\nrefseq_rna = ARN de referencia"
        )
        blast_hitlist = st.slider("Número de resultados", 5, 50, 15)
        blast_entrez_query = st.text_input(
            "Filtrar organismos (opcional)",
            placeholder='Viruses[Organism]',
            help="Limita la búsqueda a virus únicamente para resultados más relevantes"
        )

        st.markdown("---")
        st.info("""
        ⏱️ **BLAST tarda 30-120 segundos** dependiendo de la longitud de la secuencia 
        y la carga de los servidores de NCBI. 
        
        No cierres la app mientras esperas.
        """)

    # ── Botón BLAST ───────────────────────────────────────────────────────────
    run_blast = st.button("🚀 Enviar a BLAST e identificar virus",
                          type="primary", use_container_width=False, key="run_blast")

    if run_blast:
        # Limpiar y validar secuencia
        seq_clean = ""
        if blast_seq_raw.strip():
            lines = blast_seq_raw.strip().splitlines()
            seq_lines = [l for l in lines if not l.startswith(">")]
            seq_clean = "".join(seq_lines).strip().upper()
            seq_clean = "".join(c for c in seq_clean if c in "ACGTUNRYSWKMBDHV-")

        if len(seq_clean) < 20:
            st.error("⚠️ La secuencia es demasiado corta o inválida. Mínimo 20 nucleótidos.")
        elif len(seq_clean) > 100000:
            st.error("⚠️ La secuencia es demasiado larga (máx. 100,000 bp para BLAST online).")
        else:
            st.info(f"📡 Enviando secuencia de **{len(seq_clean)} bp** a NCBI BLAST... "
                    f"Esto puede tardar hasta 2 minutos.")

            progress = st.progress(0)
            status_text = st.empty()

            try:
                status_text.markdown("⏳ Conectando con NCBI BLAST...")
                progress.progress(10)

                # Construir query
                entrez_filter = blast_entrez_query.strip() if blast_entrez_query.strip() else "Viruses[Organism]"

                # Lanzar BLAST
                result_handle = NCBIWWW.qblast(
                    program=blast_program,
                    database=blast_db,
                    sequence=seq_clean,
                    hitlist_size=blast_hitlist,
                    entrez_query=entrez_filter,
                    format_type="XML",
                )

                progress.progress(70)
                status_text.markdown("✅ Resultados recibidos. Procesando...")

                # Parsear XML
                blast_records = list(NCBIXML.parse(result_handle))
                progress.progress(90)

                if not blast_records or not blast_records[0].alignments:
                    st.warning("""
                    ⚠️ No se encontraron hits significativos.
                    
                    Posibles causas:
                    - La secuencia no es de un virus conocido
                    - El filtro de organismo es demasiado restrictivo (prueba quitar el filtro)
                    - La secuencia tiene demasiadas bases ambiguas (N)
                    """)
                else:
                    blast_record = blast_records[0]
                    hits = blast_record.alignments

                    progress.progress(100)
                    status_text.empty()

                    st.success(f"✅ **{len(hits)} hits** encontrados para tu secuencia.")

                    # ── Tabla de resultados ───────────────────────────────────
                    st.markdown("---")
                    st.markdown(f"### 🦠 Identificación — Top {len(hits)} resultados")

                    rows = []
                    for i, alignment in enumerate(hits):
                        hsp = alignment.hsps[0]  # mejor HSP

                        # Extraer accession e ID limpio
                        title = alignment.title
                        acc = title.split("|")[1] if "|" in title else title.split()[0]
                        desc = " ".join(title.split()[1:])[:80] if " " in title else title[:80]

                        identity_pct = round(hsp.identities / hsp.align_length * 100, 1)
                        coverage_pct = round(hsp.align_length / len(seq_clean) * 100, 1)

                        rows.append({
                            "#": i + 1,
                            "Accession": acc,
                            "Organismo / Descripción": desc,
                            "Identidad (%)": identity_pct,
                            "Cobertura (%)": min(coverage_pct, 100.0),
                            "Score": hsp.score,
                            "E-value": f"{hsp.expect:.2e}",
                            "Gaps": hsp.gaps,
                        })

                    results_df = pd.DataFrame(rows)

                    # Colorear por identidad
                    def color_identity(val):
                        if val >= 95:
                            return "background-color: #dcfce7; color: #166534; font-weight:700"
                        elif val >= 80:
                            return "background-color: #fef9c3; color: #854d0e; font-weight:600"
                        else:
                            return "background-color: #fee2e2; color: #991b1b"

                    st.dataframe(
                        results_df.style.applymap(color_identity, subset=["Identidad (%)"]),
                        use_container_width=True, hide_index=True
                    )

                    # ── Interpretación automática ─────────────────────────────
                    st.markdown("---")
                    st.markdown("### 🧠 Interpretación automática")

                    top = rows[0]
                    top_identity = top["Identidad (%)"]
                    top_desc = top["Organismo / Descripción"]
                    top_acc = top["Accession"]
                    top_evalue = top["E-value"]

                    if top_identity >= 97:
                        verdict_icon = "🟢"
                        verdict = "**Identificación muy fiable**"
                        verdict_detail = f"La secuencia es casi idéntica a `{top_desc}` con un {top_identity}% de identidad."
                    elif top_identity >= 85:
                        verdict_icon = "🟡"
                        verdict = "**Identificación probable**"
                        verdict_detail = f"La secuencia es muy similar a `{top_desc}` ({top_identity}% identidad). Podría ser una variante o cepa diferente."
                    elif top_identity >= 70:
                        verdict_icon = "🟠"
                        verdict = "**Posible relación lejana**"
                        verdict_detail = f"La secuencia tiene similitud moderada ({top_identity}%) con `{top_desc}`. Podría ser un virus relacionado o una región conservada."
                    else:
                        verdict_icon = "🔴"
                        verdict = "**Identificación incierta**"
                        verdict_detail = f"Baja similitud ({top_identity}%) con cualquier secuencia conocida. La secuencia podría ser novedosa, estar muy degradada, o no ser viral."

                    st.markdown(f"""
                    <div style="background:#f0f9ff; border-left:4px solid #2563eb; 
                                border-radius:8px; padding:20px; margin:10px 0">
                        <h3 style="margin:0 0 8px 0; color:#1e293b">{verdict_icon} {verdict}</h3>
                        <p style="color:#374151; margin:0 0 12px 0; font-size:1.05rem">{verdict_detail}</p>
                        <table style="width:100%; border-collapse:collapse">
                            <tr>
                                <td style="padding:4px 12px 4px 0; color:#64748b; font-weight:500">Mejor match:</td>
                                <td style="color:#1e293b; font-weight:600">{top_desc}</td>
                            </tr>
                            <tr>
                                <td style="padding:4px 12px 4px 0; color:#64748b; font-weight:500">Accession:</td>
                                <td><a href="https://www.ncbi.nlm.nih.gov/nucleotide/{top_acc}" 
                                       target="_blank" style="color:#2563eb">{top_acc} ↗</a></td>
                            </tr>
                            <tr>
                                <td style="padding:4px 12px 4px 0; color:#64748b; font-weight:500">Identidad:</td>
                                <td style="color:#1e293b; font-weight:600">{top_identity}%</td>
                            </tr>
                            <tr>
                                <td style="padding:4px 12px 4px 0; color:#64748b; font-weight:500">E-value:</td>
                                <td style="color:#1e293b">{top_evalue}</td>
                            </tr>
                        </table>
                    </div>
                    """, unsafe_allow_html=True)

                    # ── Gráfico de barras de identidad ────────────────────────
                    if PLOTLY_OK and len(rows) > 1:
                        st.markdown("### 📊 Comparativa de identidad con los mejores hits")

                        # Acortar descripciones para el gráfico
                        labels = [r["Organismo / Descripción"][:45] + "..." 
                                  if len(r["Organismo / Descripción"]) > 45 
                                  else r["Organismo / Descripción"] 
                                  for r in rows]
                        identities = [r["Identidad (%)"] for r in rows]
                        bar_colors = [
                            "#16a34a" if v >= 95 else "#ca8a04" if v >= 80 else "#dc2626"
                            for v in identities
                        ]

                        fig_blast = go.Figure(go.Bar(
                            x=identities,
                            y=labels,
                            orientation="h",
                            marker_color=bar_colors,
                            text=[f"{v}%" for v in identities],
                            textposition="outside",
                            textfont=dict(size=10, color="#1e293b"),
                        ))
                        fig_blast.add_vline(x=95, line_dash="dot",
                                            line_color="#16a34a",
                                            annotation_text="95% (muy fiable)",
                                            annotation_font_color="#16a34a")
                        fig_blast.add_vline(x=80, line_dash="dot",
                                            line_color="#ca8a04",
                                            annotation_text="80%",
                                            annotation_font_color="#ca8a04")
                        fig_blast.update_layout(
                            paper_bgcolor="#ffffff", plot_bgcolor="#f8fafc",
                            font=dict(color="#1e293b"),
                            xaxis=dict(title="Identidad (%)", range=[0, 105],
                                       gridcolor="#e2e8f0"),
                            yaxis=dict(autorange="reversed"),
                            height=max(350, len(rows) * 38 + 100),
                            margin=dict(l=20, r=80, t=30, b=40),
                            title=dict(text="Porcentaje de identidad con secuencias conocidas",
                                       font=dict(color="#1d4ed8")),
                        )
                        st.plotly_chart(fig_blast, use_container_width=True)

                    # ── Leyenda de colores ────────────────────────────────────
                    st.markdown("""
                    <div style="display:flex; gap:20px; margin-top:8px; flex-wrap:wrap">
                        <span style="background:#dcfce7;color:#166534;padding:4px 10px;
                                     border-radius:6px;font-weight:600;font-size:0.85rem">
                            🟢 ≥95% Identificación muy fiable
                        </span>
                        <span style="background:#fef9c3;color:#854d0e;padding:4px 10px;
                                     border-radius:6px;font-weight:600;font-size:0.85rem">
                            🟡 80–94% Identificación probable
                        </span>
                        <span style="background:#fee2e2;color:#991b1b;padding:4px 10px;
                                     border-radius:6px;font-weight:600;font-size:0.85rem">
                            🔴 &lt;80% Identificación incierta
                        </span>
                    </div>
                    """, unsafe_allow_html=True)

                    # ── Opciones post-identificación ──────────────────────────
                    st.markdown("---")
                    st.markdown("### ➡️ ¿Qué quieres hacer con esta secuencia?")

                    col_a, col_b, col_c = st.columns(3)
                    with col_a:
                        if st.button("📥 Añadir al análisis principal", key="blast_to_main"):
                            if blast_seq_raw.strip():
                                new_recs = parse_fasta_text(
                                    blast_seq_raw if blast_seq_raw.startswith(">")
                                    else f">Secuencia_BLAST\n{seq_clean}"
                                )
                                if new_recs:
                                    st.session_state.records.extend(new_recs)
                                    st.success("✅ Añadida. Ve a 📂 Carga de Secuencias para verla.")
                    with col_b:
                        st.download_button(
                            "💾 Descargar resultados (CSV)",
                            data=df_to_csv_bytes(results_df),
                            file_name="blast_resultados.csv",
                            mime="text/csv",
                            key="dl_blast_csv"
                        )
                    with col_c:
                        # Link a BLAST online con la misma búsqueda
                        st.markdown(f"""
                        <a href="https://blast.ncbi.nlm.nih.gov/blast/Blast.cgi?PROGRAM={blast_program}&PAGE_TYPE=BlastSearch&DATABASE={blast_db}" 
                           target="_blank">
                            <button style="background:#2563eb;color:white;border:none;
                                           border-radius:8px;padding:8px 16px;
                                           font-weight:600;cursor:pointer;width:100%">
                                🌐 Abrir en NCBI BLAST
                            </button>
                        </a>
                        """, unsafe_allow_html=True)

            except Exception as e:
                progress.empty()
                status_text.empty()
                st.error(f"❌ Error conectando con NCBI BLAST: {e}")
                st.markdown("""
                **Posibles causas:**
                - Sin conexión a internet
                - Servidores de NCBI temporalmente caídos
                - Secuencia con caracteres inválidos
                
                Puedes usar BLAST directamente en: https://blast.ncbi.nlm.nih.gov
                """)

    # ── Guía de interpretación ────────────────────────────────────────────────
    with st.expander("📖 Guía de interpretación de resultados BLAST"):
        st.markdown("""
        ### ¿Cómo leer los resultados?

        | Campo | Significado |
        |-------|-------------|
        | **Identidad (%)** | Porcentaje de bases idénticas entre tu secuencia y el hit |
        | **Cobertura (%)** | Qué porcentaje de tu secuencia está cubierto por el hit |
        | **E-value** | Probabilidad de que el hit sea por azar. Cuanto más pequeño (ej. `1e-50`), más significativo |
        | **Score** | Puntuación de la alineación. Mayor score = mejor alineamiento |

        ### Guía rápida de identidad
        - **≥ 99%** → Misma cepa o muy cercana
        - **95–99%** → Misma especie viral, diferente aislado
        - **85–95%** → Misma especie, diferente linaje/variante
        - **70–85%** → Virus relacionado (mismo género)
        - **< 70%** → Relación lejana o secuencia problemática

        ### ¿Por qué usar `Viruses[Organism]` como filtro?
        Sin filtro, BLAST puede devolver hits de bacterias, humanos u otros organismos 
        que también contienen secuencias similares. El filtro `Viruses[Organism]` 
        restringe los resultados solo a secuencias virales.

        ### Programas BLAST
        - **blastn** → Para secuencias de DNA/RNA viral (lo más habitual)
        - **blastx** → Traduce tu secuencia nucleotídica y busca en proteínas (útil para regiones conservadas)
        - **blastp** → Si tienes directamente una secuencia de aminoácidos
        """)


# ══════════════════════════════════════════════════════════════════════════════
# FOOTER
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("---")
st.markdown("""
<div style="text-align:center; color:#64748b; font-size:0.85rem; padding: 10px 0">
    🧬 <strong>ViroSeq Analyzer</strong> · Herramienta de bioinformática para virología
    · Desarrollada con <a href="https://streamlit.io" style="color:#2563eb">Streamlit</a> 
    + <a href="https://biopython.org" style="color:#2563eb">BioPython</a> 
    + <a href="https://plotly.com" style="color:#2563eb">Plotly</a>
</div>
""", unsafe_allow_html=True)
