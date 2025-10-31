# -*- coding: utf-8 -*-
"""
ANALYSER - Analizador Jurídico con RAG + Gemini Pro  
----------------------------------------------------
- Bases RAG versionadas: Múltiples dominios jurídicos
- LLM: Gemini 2.5 Pro
- Recuperación: Chroma con re-scoring jurídico
- Entrada: PDF (con OCR) o texto/consulta
- Salida: Informe PDF + JSON

Uso CLI:
  python analyser.py --pdf ./ejemplo.pdf --out reporte.pdf
  python analyser.py --query "consentimiento informado en cirugía" --out reporte.pdf
"""

from __future__ import annotations
import os, re, json, logging
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Optional
from datetime import datetime
from pathlib import Path

# Configuración centralizada
from config import (
    BASES_RAG, DEFAULT_LLM, DEFAULT_TEMPERATURE,
    KEYWORDS_JURIDICO_SANIT, AnalysisConfig, DEFAULT_ANALYSIS_CONFIG,
    get_analysis_prompt
)

# Sistema de versionado y validación
from embedding_validator import VALIDATOR
from version_manager import REGISTRY

# LLM (Gemini)
from langchain_google_genai import ChatGoogleGenerativeAI

# Vectorstore / Embeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

# PDF Report
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.units import cm

# Extracción PDF (de ingesta)
try:
    from ingesta import extract_text_from_pdf, DEFAULT_INGESTA_CONFIG
except ImportError:
    extract_text_from_pdf = None
    DEFAULT_INGESTA_CONFIG = None

# ======================
# Logger
# ======================

LOG = logging.getLogger("analyser")
if not LOG.handlers:
    LOG.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    LOG.addHandler(handler)

# ======================
# Prompts (jurídico-médicos, multilingües)
# ======================

SYSTEM_INSTRUCCIONES = (
    "Eres un analista jurídico-médico multilingüe (ES/EN/FR/DE/IT). "
    "Evalúas responsabilidad profesional médica, lex artis, consentimiento informado, "
    "causalidad, daño, nexo adecuado y estándares OMS/OPS. Estructura tus respuestas."
)

PROMPT_ANALISIS = get_analysis_prompt()

# ======================
# LLM Wrapper
# ======================

def make_llm(model: str = DEFAULT_LLM, temperature: float = DEFAULT_TEMPERATURE) -> ChatGoogleGenerativeAI:
    """Inicializa el modelo Gemini con validación de clave"""
    api_key = os.environ.get("GOOGLE_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("⚠️ Falta GOOGLE_API_KEY en el entorno para usar Gemini.")
    return ChatGoogleGenerativeAI(model=model, temperature=temperature)

# ======================
# Carga de Vectorstores con Validación
# ======================

def load_all_retrievers(bases: Dict[str, str]) -> Dict[str, Chroma]:
    """
    Carga todas las bases vectoriales con validación de compatibilidad.

    Args:
        bases: Diccionario de bases (nombre -> carpeta)

    Returns:
        Diccionario de vectorstores cargados
    """
    return VALIDATOR.load_all_vectorstores(bases)

# ======================
# Búsqueda mejorada (jurídico-sanitaria)
# ======================

def score_dinamico(
    query: str,
    doc: Document,
    score_sim: float,
    config: AnalysisConfig = DEFAULT_ANALYSIS_CONFIG
) -> float:
    """
    Re-scoring dinámico con heurísticas jurídico-sanitarias.

    Args:
        query: Consulta original
        doc: Documento candidato
        score_sim: Score de similitud base
        config: Configuración de análisis

    Returns:
        Score ajustado
    """
    base = float(score_sim)
    text = doc.page_content.lower()

    # 1. BONUS por keywords jurídico-sanitarias
    kw_hits = sum(1 for k in KEYWORDS_JURIDICO_SANIT if k in text)
    base += min(config.keyword_bonus_max, kw_hits * 0.02)

    # 2. BONUS por tipo estructural coincidente
    if config.enable_rescoring:
        tipo = doc.metadata.get("tipo_estructura", "parrafo")
        q = query.lower()
        if "considerando" in q and tipo == "considerando":
            base += config.structure_bonus
        if ("art." in q or "artículo" in q or "articulo" in q) and tipo == "articulo":
            base += config.structure_bonus
        if "resuelve" in q and tipo == "resolutivo":
            base += config.structure_bonus

    # 3. BONUS por longitud óptima (70-900 palabras)
    n = len(doc.page_content.split())
    if 70 <= n <= 900:
        base += 0.05

    # 4. BONUS por calidad del fragmento
    base += min(config.quality_bonus_max, float(doc.metadata.get("score_calidad", 0.5)) * 0.1)

    return max(0.0, min(1.5, base))


def enhanced_similarity_search(
    vs: Chroma,
    query: str,
    k: int = 8,
    fetch_k: int = 32,
    config: AnalysisConfig = DEFAULT_ANALYSIS_CONFIG
) -> List[Tuple[Document, float]]:
    """
    Búsqueda con re-ranking jurídico-sanitario.

    Args:
        vs: Vectorstore Chroma
        query: Consulta
        k: Documentos finales a devolver
        fetch_k: Candidatos iniciales a buscar
        config: Configuración de análisis

    Returns:
        Lista de (Document, score) ordenada por relevancia
    """
    try:
        base_docs = vs.similarity_search_with_score(query, k=fetch_k)
    except Exception:
        # Fallback si no soporta score
        docs = vs.similarity_search(query, k=fetch_k)
        base_docs = [(d, 0.5) for d in docs]

    # Re-scoring
    rescored = [(doc, score_dinamico(query, doc, score, config)) for (doc, score) in base_docs]
    rescored.sort(key=lambda x: x[1], reverse=True)

    return rescored[:k]

# ======================
# Construcción de Contexto RAG
# ======================

def build_context(
    query_or_text: str,
    retrievers: Dict[str, Chroma],
    config: AnalysisConfig = DEFAULT_ANALYSIS_CONFIG
) -> str:
    """
    Recupera fragmentos de las bases vectoriales con metadatos enriquecidos.

    Args:
        query_or_text: Consulta o texto a analizar
        retrievers: Diccionario de vectorstores
        config: Configuración de análisis

    Returns:
        Contexto formateado con referencias
    """
    bloques = []

    for etiqueta, vs in retrievers.items():
        try:
            results = enhanced_similarity_search(
                vs,
                query_or_text,
                k=config.k_por_base,
                fetch_k=config.fetch_k,
                config=config
            )

            for doc, score in results:
                meta = doc.metadata or {}

                # Extraer metadata
                fuente = meta.get("archivo_origen", "?")
                autor = meta.get("autor", "Autor desconocido")
                titulo = meta.get("titulo", fuente)
                pagina = meta.get("pagina", meta.get("page", "s/p"))
                jurisdiccion = meta.get("jurisdiccion", "Desconocida")
                idioma = meta.get("idioma", "ES")
                tipo = meta.get("tipo", "texto")
                anio = meta.get("anio", "s/f")
                url = meta.get("url", "")
                origen = meta.get("origen", etiqueta)

                # Formato de referencia
                ref = f"{autor} – *{titulo}* ({anio}), pág. {pagina}"
                if url:
                    ref += f" [{url}]"

                encabezado = (
                    f"📚 FUENTE: {ref} | Jurisdicción: {jurisdiccion} | "
                    f"Idioma: {idioma} | Tipo: {tipo} | Base: {origen} | Score: {score:.2f}"
                )

                contenido = doc.page_content.strip().replace("\n", " ")
                resumen = contenido[:1200]
                bloques.append(f"{encabezado}\n{resumen}\n---")

        except Exception as e:
            LOG.warning(f"Recuperación falló en {etiqueta}: {e}")

    # Limitar contexto total
    return "\n\n".join(bloques[:15])

# ======================
# Invocación Gemini con Extracción Estructurada
# ======================

def invoke_llm_json(llm: ChatGoogleGenerativeAI, system: str, user_prompt: str) -> Dict:
    """
    Ejecuta Gemini Pro y extrae respuesta estructurada.

    Args:
        llm: Instancia de ChatGoogleGenerativeAI
        system: Instrucciones del sistema
        user_prompt: Prompt del usuario

    Returns:
        Diccionario con análisis estructurado
    """
    try:
        full_prompt = f"SYSTEM:\n{system}\n\nUSER:\n{user_prompt}"
        resp = llm.invoke(full_prompt)
        txt = resp.content if hasattr(resp, "content") else str(resp)

        # Detectar y extraer bloque JSON
        m = re.search(r'\{.*\}', txt, flags=re.S)
        parsed = {}
        if m:
            try:
                parsed = json.loads(m.group(0))
            except json.JSONDecodeError:
                parsed = {}

        # Extraer razonamiento narrativo
        razonamiento = txt.split("{")[0].strip()
        parsed["texto_completo"] = razonamiento or txt.strip()

        # Validación y formato mínimo
        parsed.setdefault("tesis", "(sin tesis detectada)")
        parsed.setdefault("conceptos_clave", [])
        parsed.setdefault("debilidades", [])
        parsed.setdefault("preguntas", [])
        parsed.setdefault("probabilidad_exito", "media")
        parsed.setdefault("fuentes_relevantes", [])

        # Normalizar fuentes_relevantes
        if isinstance(parsed["fuentes_relevantes"], list):
            fuentes_limpias = []
            for f in parsed["fuentes_relevantes"]:
                if isinstance(f, dict):
                    fuentes_limpias.append({
                        "autor": f.get("autor", ""),
                        "titulo": f.get("titulo", ""),
                        "anio": f.get("anio", ""),
                        "pagina": f.get("pagina", ""),
                        "url": f.get("url", ""),
                        "tipo": f.get("tipo", "")
                    })
                elif isinstance(f, str):
                    fuentes_limpias.append({
                        "autor": f, "titulo": "", "anio": "",
                        "pagina": "", "url": "", "tipo": ""
                    })
            parsed["fuentes_relevantes"] = fuentes_limpias

        return parsed

    except Exception as e:
        LOG.error(f"Error en LLM: {e}")
        return {
            "tesis": f"[LLM_ERROR] {e}",
            "conceptos_clave": [],
            "debilidades": [],
            "preguntas": [],
            "probabilidad_exito": "media",
            "fuentes_relevantes": [],
            "texto_completo": ""
        }

# ======================
# Análisis Principal
# ======================

def analyse_text_medico(
    texto: str,
    retrievers: Dict[str, Chroma],
    llm: ChatGoogleGenerativeAI,
    config: AnalysisConfig = DEFAULT_ANALYSIS_CONFIG
) -> Dict:
    """
    Analiza texto con RAG jurídico-sanitario.

    Args:
        texto: Texto o consulta a analizar
        retrievers: Diccionario de vectorstores
        llm: Modelo Gemini
        config: Configuración de análisis

    Returns:
        Diccionario con análisis estructurado
    """
    contexto = build_context(texto, retrievers, config)
    texto_truncado = texto[:config.max_context_length]
    user_prompt = PROMPT_ANALISIS.format(contexto=contexto, texto=texto_truncado)
    return invoke_llm_json(llm, SYSTEM_INSTRUCCIONES, user_prompt)


def analyse_pdf_medico(
    pdf_path: Path,
    retrievers: Dict[str, Chroma],
    llm: ChatGoogleGenerativeAI,
    config: AnalysisConfig = DEFAULT_ANALYSIS_CONFIG
) -> Dict:
    """
    Analiza PDF con extracción de texto + RAG.

    Args:
        pdf_path: Ruta al PDF
        retrievers: Diccionario de vectorstores
        llm: Modelo Gemini
        config: Configuración de análisis

    Returns:
        Diccionario con análisis estructurado
    """
    if extract_text_from_pdf is None:
        raise RuntimeError("No se encontró función extract_text_from_pdf (importa ingesta).")

    text = extract_text_from_pdf(pdf_path, DEFAULT_INGESTA_CONFIG)

    if not text.strip():
        return {
            "tesis": "[ERROR] No se pudo extraer texto del PDF",
            "conceptos_clave": [],
            "debilidades": [],
            "preguntas": [],
            "probabilidad_exito": "media",
            "fuentes_relevantes": [],
            "texto_completo": ""
        }

    return analyse_text_medico(text, retrievers, llm, config)

# ======================
# Análisis Avanzado Iterativo (Nivel 2+)
# ======================

def analyse_deep_layer(
    result_json: dict,
    llm: ChatGoogleGenerativeAI,
    pregunta: str,
    nivel: int = 2
) -> dict:
    """
    Análisis jurídico profundo iterativo.

    Args:
        result_json: Resultado del análisis previo
        llm: Modelo Gemini
        pregunta: Nueva consulta o profundización
        nivel: Nivel de análisis (2, 3, ...)

    Returns:
        Diccionario con análisis avanzado
    """
    context = json.dumps(result_json, ensure_ascii=False, indent=2)

    prompt = f"""
    [Análisis previo - Nivel {nivel-1}]
    {context}

    [Nueva consulta - Nivel {nivel}]
    {pregunta}

    [Instrucciones]
    1) Usa el análisis previo como base factual. No repitas información.
    2) Profundiza el razonamiento jurídico: doctrina, jurisprudencia, interpretación normativa.
    3) Evalúa contradicciones, debilidades o mejoras posibles.
    4) Si procede, propone estrategias o líneas argumentales.
    5) Devuelve un JSON estructurado con:
       - "analisis_avanzado": texto razonado y extenso
       - "nuevos_fundamentos": lista de fundamentos adicionales
       - "nuevos_riesgos": lista de riesgos adicionales
       - "nivel": número de nivel actual ({nivel})
    """

    return invoke_llm_json(
        llm,
        "Eres un jurista especializado en responsabilidad médica y análisis iterativo de jurisprudencia y doctrina.",
        prompt
    )

# ======================
# Auditoría Vectorial
# ======================

@dataclass
class VectorAudit:
    """Resultado de auditoría de base vectorial"""
    base: str
    frags: int
    avg_words: float
    diversity: float
    coverage_types: float
    rating: float


def audit_vectorstore(vs: Chroma, nombre: str, sample: int = 400) -> VectorAudit:
    """
    Audita calidad de una base vectorial.

    Args:
        vs: Vectorstore Chroma
        nombre: Nombre de la base
        sample: Cantidad de documentos a muestrear

    Returns:
        VectorAudit con métricas
    """
    try:
        docs = vs.similarity_search("", k=sample)
    except Exception:
        docs = []

    if not docs:
        return VectorAudit(nombre, 0, 0.0, 0.0, 0.0, 0.0)

    # Métricas
    sizes = [len(d.page_content.split()) for d in docs]
    avgw = sum(sizes) / len(sizes)

    # Diversidad léxica
    vocab = set()
    tot = 0
    for d in docs:
        ws = d.page_content.lower().split()
        tot += len(ws)
        vocab.update(ws)
    diversity = min(1.0, (len(vocab) / max(tot, 1)) * 3)

    # Cobertura de tipos
    tipos = [d.metadata.get("tipo_estructura", "parrafo") for d in docs]
    coverage = min(1.0, len(set(tipos)) / 5.0)

    # Rating final (0-5)
    rating = round((min(1.0, avgw / 400.0) + diversity + coverage) / 3 * 5, 2)

    return VectorAudit(nombre, len(docs), avgw, diversity, coverage, rating)

# ======================
# Generación de PDF
# ======================

def draw_wrapped_text(
    c: canvas.Canvas,
    text: str,
    x: float,
    y: float,
    max_width: float,
    leading: float = 14
) -> float:
    """Dibuja texto con word-wrap"""
    from reportlab.pdfbase.pdfmetrics import stringWidth
    words = text.split()
    line = ""
    for w in words:
        test = f"{line} {w}".strip()
        if stringWidth(test, "Helvetica", 10) <= max_width:
            line = test
        else:
            c.drawString(x, y, line)
            y -= leading
            line = w
    if line:
        c.drawString(x, y, line)
        y -= leading
    return y


def render_pdf_report(
    output_path: Path,
    titulo: str,
    entrada: str,
    result: Dict,
    auditorias: List[VectorAudit]
) -> None:
    """
    Genera reporte PDF del análisis.

    Args:
        output_path: Ruta de salida del PDF
        titulo: Título del reporte
        entrada: Descripción de la entrada
        result: Resultado del análisis
        auditorias: Lista de auditorías de bases
    """
    c = canvas.Canvas(str(output_path), pagesize=A4)
    W, H = A4
    margin = 2 * cm
    x = margin
    y = H - margin

    # Encabezado
    c.setFont("Helvetica-Bold", 14)
    c.drawString(x, y, titulo)
    y -= 16
    c.setFont("Helvetica", 9)
    c.drawString(x, y, f"Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    y -= 12
    c.drawString(x, y, "Analizador: Gemini 2.5 Pro + RAG sanitario multilingüe")
    y -= 18

    # Entrada
    c.setFont("Helvetica-Bold", 12)
    c.drawString(x, y, "Entrada (resumen):")
    y -= 14
    c.setFont("Helvetica", 10)
    y = draw_wrapped_text(c, entrada[:1000], x, y, W - 2 * margin)

    # Tesis
    c.setFont("Helvetica-Bold", 12)
    c.drawString(x, y, "Tesis (síntesis):")
    y -= 14
    c.setFont("Helvetica", 10)
    y = draw_wrapped_text(c, result.get("tesis", ""), x, y, W - 2 * margin)

    # Conceptos clave
    conceptos = result.get("conceptos_clave", [])
    if conceptos:
        c.setFont("Helvetica-Bold", 12)
        if y < margin + 60:
            c.showPage()
            y = H - margin
        c.drawString(x, y, "Conceptos clave:")
        y -= 14
        c.setFont("Helvetica", 10)
        for concepto in conceptos:
            if y < margin + 60:
                c.showPage()
                y = H - margin
            y = draw_wrapped_text(c, f"- {concepto}", x, y, W - 2 * margin)

    # Debilidades
    debilidades = result.get("debilidades", [])
    if debilidades:
        c.setFont("Helvetica-Bold", 12)
        if y < margin + 60:
            c.showPage()
            y = H - margin
        c.drawString(x, y, "Debilidades:")
        y -= 14
        c.setFont("Helvetica", 10)
        for deb in debilidades:
            if y < margin + 60:
                c.showPage()
                y = H - margin
            y = draw_wrapped_text(c, f"- {deb}", x, y, W - 2 * margin)

    # Preguntas
    preguntas = result.get("preguntas", [])
    if preguntas:
        c.setFont("Helvetica-Bold", 12)
        if y < margin + 60:
            c.showPage()
            y = H - margin
        c.drawString(x, y, "Preguntas derivadas:")
        y -= 14
        c.setFont("Helvetica", 10)
        for preg in preguntas:
            if y < margin + 60:
                c.showPage()
                y = H - margin
            y = draw_wrapped_text(c, f"- {preg}", x, y, W - 2 * margin)

    # Probabilidad de éxito
    c.setFont("Helvetica-Bold", 12)
    if y < margin + 40:
        c.showPage()
        y = H - margin
    c.drawString(x, y, "Probabilidad de éxito:")
    y -= 14
    c.setFont("Helvetica", 10)
    c.drawString(x, y, result.get("probabilidad_exito", "media").upper())
    y -= 18

    # Auditoría vectorial
    if auditorias:
        c.setFont("Helvetica-Bold", 12)
        if y < margin + 80:
            c.showPage()
            y = H - margin
        c.drawString(x, y, "Auditoría de bases vectoriales:")
        y -= 14
        c.setFont("Helvetica", 10)
        for a in auditorias:
            if y < margin + 70:
                c.showPage()
                y = H - margin
            c.drawString(
                x, y,
                f"• {a.base}: frags={a.frags}, avg_words={a.avg_words:.1f}, "
                f"diversidad={a.diversity:.2f}, cobertura={a.coverage_types:.2f}, rating={a.rating:.2f}"
            )
            y -= 12

    # Texto completo del modelo
    if "texto_completo" in result and result["texto_completo"].strip():
        if y < margin + 100:
            c.showPage()
            y = H - margin
        c.setFont("Helvetica-Bold", 12)
        c.drawString(x, y, "Informe completo del modelo:")
        y -= 14
        c.setFont("Helvetica", 10)
        y = draw_wrapped_text(c, result["texto_completo"][:4000], x, y, W - 2 * margin)

    c.showPage()
    c.save()
    LOG.info(f"✅ PDF generado: {output_path}")

# ======================
# CLI
# ======================

def main():
    import argparse
    ap = argparse.ArgumentParser(description="Analizador sanitario (Gemini Pro + RAG) con reporte PDF")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--pdf", help="Ruta a PDF a analizar")
    g.add_argument("--text", help="Ruta a archivo de texto")
    g.add_argument("--query", help="Consulta breve a evaluar")
    ap.add_argument("--out", required=True, help="Ruta de salida del PDF")
    ap.add_argument("--llm", default=DEFAULT_LLM, help=f"Modelo Gemini (por defecto {DEFAULT_LLM})")
    args = ap.parse_args()

    # Cargar LLM
    llm = make_llm(model=args.llm)

    # Cargar retrievers con validación
    retrievers = load_all_retrievers(BASES_RAG)
    if not retrievers:
        LOG.warning("No se encontraron bases sanitarias. Verifica 'chroma_db_legal/*'.")

    # Determinar entrada y ejecutar análisis
    entrada = ""
    result = {}

    if args.pdf:
        entrada = f"[PDF] {args.pdf}"
        result = analyse_pdf_medico(Path(args.pdf), retrievers, llm)
    elif args.text:
        texto = Path(args.text).read_text(encoding="utf-8", errors="ignore")
        entrada = f"[TEXT] {args.text}"
        result = analyse_text_medico(texto, retrievers, llm)
    elif args.query:
        entrada = f"[QUERY] {args.query}"
        result = analyse_text_medico(args.query, retrievers, llm)

    # Auditorías
    auditorias: List[VectorAudit] = []
    for nombre, vs in retrievers.items():
        auditorias.append(audit_vectorstore(vs, nombre))

    # Generar PDF
    out = Path(args.out)
    render_pdf_report(out, "Informe Sanitario Juridificado (Gemini + RAG)", entrada, result, auditorias)

    # Output JSON
    paquete = {
        "entrada": entrada,
        "resultado": result,
        "auditoria": [asdict(a) for a in auditorias],
        "timestamp": datetime.now().isoformat()
    }
    print(json.dumps(paquete, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
