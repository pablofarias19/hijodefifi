# -*- coding: utf-8 -*-
"""
APP STREAMLIT CORREGIDA - Sistema RAG Jurídico
==============================================
Versión corregida con RAG inteligente integrado y exportación PDF
"""

import streamlit as st
import os
import chromadb
from sentence_transformers import SentenceTransformer
from pathlib import Path
import time
from datetime import datetime
import json
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from io import BytesIO

# Importar motor inteligente
try:
    from motor_rag_inteligente import RagMotorInteligente, listar_prompts_disponibles
    MOTOR_INTELIGENTE_DISPONIBLE = True
except ImportError:
    MOTOR_INTELIGENTE_DISPONIBLE = False

# Configuración
DB_PATH = Path("chroma_db_legal")
EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"

def configurar_pagina():
    """Configuración de la página"""
    st.set_page_config(
        page_title="Sistema RAG Jurídico - App Completa",
        page_icon="⚖️",
        layout="wide"
    )

def crear_pdf_reporte(consulta, base_seleccionada, resultado, modo="inteligente", prompt_usado=None):
    """Crea un reporte PDF del análisis"""
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, rightMargin=72, leftMargin=72, topMargin=72, bottomMargin=18)
    
    # Estilos
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=16,
        spaceAfter=30,
        textColor=colors.darkblue
    )
    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=12,
        spaceAfter=12,
        textColor=colors.darkgreen
    )
    
    # Contenido del documento
    story = []
    
    # Título
    story.append(Paragraph("⚖️ SISTEMA RAG JURÍDICO - REPORTE DE ANÁLISIS", title_style))
    story.append(Spacer(1, 12))
    
    # Información general
    info_data = [
        ['Consulta:', consulta],
        ['Base de Datos:', base_seleccionada],
        ['Modo:', 'RAG Inteligente' if modo == 'inteligente' else 'Búsqueda Tradicional'],
        ['Prompt Utilizado:', prompt_usado if prompt_usado else 'Sistema predeterminado'],
        ['Fecha:', datetime.now().strftime('%d/%m/%Y %H:%M:%S')]
    ]
    
    info_table = Table(info_data, colWidths=[2*inch, 4*inch])
    info_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (0, -1), colors.lightgrey),
        ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    
    story.append(info_table)
    story.append(Spacer(1, 20))
    
    if modo == 'inteligente' and resultado:
        # Respuesta inteligente
        story.append(Paragraph("🎯 ANÁLISIS INTELIGENTE", heading_style))
        
        respuesta_text = resultado.get('respuesta', 'Sin respuesta disponible')
        # Limpiar texto para PDF
        respuesta_text = respuesta_text.replace('**', '').replace('*', '').replace('#', '')
        
        story.append(Paragraph(respuesta_text, styles['Normal']))
        story.append(Spacer(1, 20))
        
        # Fuentes consultadas con referencias bibliográficas
        if 'fragmentos' in resultado and resultado['fragmentos']:
            story.append(Paragraph("📚 REFERENCIAS BIBLIOGRÁFICAS", heading_style))
            
            for i, fragmento in enumerate(resultado['fragmentos'], 1):
                metadata = fragmento.get('metadata', {})
                
                # Generar referencia bibliográfica completa
                try:
                    from extractor_metadatos import generar_cita_bibliografica
                    referencia = generar_cita_bibliografica(metadata)
                except ImportError:
                    # Fallback manual
                    autor = metadata.get('autor', 'Autor desconocido')
                    titulo = metadata.get('titulo', metadata.get('filename', 'Sin título'))
                    pagina = metadata.get('pagina', 'sin página')
                    anio = metadata.get('anio', '')
                    
                    referencia = f"{autor} – {titulo}"
                    if anio:
                        referencia += f" ({anio})"
                    if pagina != 'sin página' and pagina != 's/p':
                        referencia += f", pág. {pagina}"
                
                story.append(Paragraph(f"<b>[{i}]</b> {referencia}", styles['Normal']))
                
                # Información adicional si está disponible
                if metadata.get('editorial', '') != 'Editorial desconocida':
                    story.append(Paragraph(f"Editorial: {metadata.get('editorial', '')}", styles['Normal']))
                if metadata.get('isbn', '') != 'Sin ISBN':
                    story.append(Paragraph(f"ISBN: {metadata.get('isbn', '')}", styles['Normal']))
                
                story.append(Paragraph(f"Relevancia: {fragmento.get('relevancia', 0):.3f}", styles['Normal']))
                story.append(Spacer(1, 12))
    
    else:
        # Resultados tradicionales
        story.append(Paragraph("🔍 RESULTADOS DE BÚSQUEDA", heading_style))
        
        if isinstance(resultado, list):
            for i, fragmento in enumerate(resultado, 1):
                story.append(Paragraph(f"<b>Resultado {i}</b>", styles['Normal']))
                story.append(Paragraph(f"<b>Fuente:</b> {fragmento.get('fuente', 'Sin fuente')}", styles['Normal']))
                story.append(Paragraph(f"<b>Relevancia:</b> {fragmento.get('relevancia', 0):.3f}", styles['Normal']))
                
                contenido = fragmento.get('contenido', '')[:400] + '...' if len(fragmento.get('contenido', '')) > 400 else fragmento.get('contenido', '')
                story.append(Paragraph(f"<b>Contenido:</b> {contenido}", styles['Normal']))
                story.append(Spacer(1, 15))
    
    # Pie de página
    story.append(Spacer(1, 30))
    story.append(Paragraph("---", styles['Normal']))
    story.append(Paragraph("Generado por Sistema RAG Jurídico | pensarjuridico.com", styles['Normal']))
    
    # Construir PDF
    doc.build(story)
    buffer.seek(0)
    return buffer

def generar_nombre_archivo(consulta, extension="pdf"):
    """Genera nombre de archivo basado en la consulta"""
    # Tomar primeras palabras de la consulta
    palabras = consulta.split()[:3]
    nombre_base = "_".join(palabra.replace("¿", "").replace("?", "").replace(",", "") for palabra in palabras)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    return f"analisis_{nombre_base}_{timestamp}.{extension}"
def verificar_api_key():
    """Verifica si hay API key de Gemini"""
    return bool(os.getenv('GOOGLE_API_KEY'))

def verificar_motor_inteligente():
    """Verifica si el motor inteligente está disponible"""
    return MOTOR_INTELIGENTE_DISPONIBLE and verificar_api_key()

@st.cache_resource
def cargar_motor_inteligente():
    """Carga el motor RAG inteligente"""
    try:
        if verificar_motor_inteligente():
            return RagMotorInteligente()
        return None
    except Exception as e:
        st.error(f"Error cargando motor inteligente: {e}")
        return None

def obtener_bases_disponibles():
    """Obtiene bases disponibles"""
    bases = []
    if DB_PATH.exists():
        for base_dir in DB_PATH.iterdir():
            if base_dir.is_dir() and (base_dir / "chroma.sqlite3").exists():
                bases.append(base_dir.name)
    return sorted(bases)

def conectar_base(nombre_base):
    """Conecta a una base específica"""
    base_path = DB_PATH / nombre_base
    if not base_path.exists():
        return None, 0
    
    try:
        client = chromadb.PersistentClient(path=str(base_path))
        collections = client.list_collections()
        if not collections:
            return None, 0
        
        collection = collections[0]
        return collection, collection.count()
    except Exception:
        return None, 0

@st.cache_resource
def cargar_modelo():
    """Carga el modelo de embeddings"""
    try:
        return SentenceTransformer(EMBEDDING_MODEL)
    except Exception as e:
        st.error(f"Error cargando modelo: {e}")
        return None

def realizar_consulta_inteligente(consulta, nombre_base, prompt_especifico=None):
    """Realiza consulta usando el motor inteligente con prompt opcional"""
    try:
        motor = cargar_motor_inteligente()
        if not motor:
            return None
        
        # Realizar consulta inteligente con prompt específico si se proporciona
        if prompt_especifico:
            resultado = motor.consultar_con_prompt_especifico(consulta, nombre_base, prompt_especifico)
        else:
            resultado = motor.consulta_completa(consulta, nombre_base)
        return resultado
        
    except Exception as e:
        st.error(f"Error en consulta inteligente: {e}")
        return None

def gestionar_prompts():
    """Interfaz para gestionar prompts desde la aplicación"""
    with st.expander("🎯 **Gestión de Prompts** - Personaliza tu análisis"):
        
        # Pestañas para organizar funcionalidad
        tab1, tab2, tab3 = st.tabs(["📋 Seleccionar Prompt", "✍️ Crear Prompt", "📁 Ver Prompts"])
        
        with tab1:
            st.markdown("### 🎯 Seleccionar Prompt para la Consulta")
            
            # Listar prompts disponibles
            try:
                prompts_disponibles = listar_prompts_disponibles()
                
                if prompts_disponibles:
                    opciones_prompt = ["🔄 Usar predeterminado"] + [f"📝 {p['nombre']}" for p in prompts_disponibles]
                    
                    prompt_seleccionado = st.selectbox(
                        "Elige el tipo de análisis:",
                        opciones_prompt,
                        help="Selecciona 'predeterminado' para usar el sistema automático, o elige un prompt específico"
                    )
                    
                    # Mostrar información del prompt seleccionado
                    if prompt_seleccionado != "🔄 Usar predeterminado":
                        nombre_prompt = prompt_seleccionado.replace("📝 ", "")
                        prompt_info = next((p for p in prompts_disponibles if p['nombre'] == nombre_prompt), None)
                        
                        if prompt_info:
                            st.info(f"**Archivo:** {prompt_info['archivo']}")
                            
                            # Vista previa del prompt
                            if st.checkbox("👁️ Ver contenido del prompt"):
                                try:
                                    with open(prompt_info['ruta'], 'r', encoding='utf-8') as f:
                                        contenido = f.read()
                                    st.text_area("Contenido del prompt:", contenido, height=200, disabled=True)
                                except Exception as e:
                                    st.error(f"Error leyendo prompt: {e}")
                    
                    # Guardar selección en session_state
                    if prompt_seleccionado == "🔄 Usar predeterminado":
                        st.session_state.prompt_seleccionado = None
                    else:
                        st.session_state.prompt_seleccionado = prompt_seleccionado.replace("📝 ", "")
                else:
                    st.warning("No se encontraron prompts disponibles")
                    
            except Exception as e:
                st.error(f"Error listando prompts: {e}")
        
        with tab2:
            st.markdown("### ✍️ Crear Nuevo Prompt")
            
            col1, col2 = st.columns([1, 1])
            
            with col1:
                nombre_nuevo_prompt = st.text_input(
                    "Nombre del prompt:",
                    placeholder="ej: analisis_comercial, prompt_rapido_civil",
                    help="El archivo se guardará como {nombre}.md"
                )
            
            with col2:
                tipo_prompt = st.selectbox(
                    "Tipo de prompt:",
                    ["Personalizado", "Basado en template"],
                    help="Elige si crear desde cero o usar un template"
                )
            
            if tipo_prompt == "Basado en template":
                template_base = st.selectbox(
                    "Template base:",
                    ["analisis_profundo", "analisis_rapido", "analisis_prueba"],
                    help="Selecciona un template para personalizar"
                )
            
            contenido_nuevo_prompt = st.text_area(
                "Contenido del prompt:",
                placeholder="""# PROMPT DE ANÁLISIS PERSONALIZADO

Eres un asistente jurídico especializado en...

## INSTRUCCIONES PRINCIPALES
- Realiza análisis de...
- Enfócate en...

### ESTRUCTURA DE RESPUESTA
{
  "campo1": "descripción",
  "campo2": "descripción"
}

**CONTEXTO:**
{contexto}

**CONSULTA:**
{texto}""",
                height=300,
                help="Usa {contexto} y {texto} como marcadores de posición"
            )
            
            if st.button("💾 Guardar Prompt", type="primary"):
                if nombre_nuevo_prompt and contenido_nuevo_prompt:
                    try:
                        # Asegurar que el nombre termine en .md
                        if not nombre_nuevo_prompt.endswith('.md'):
                            nombre_archivo = f"{nombre_nuevo_prompt}.md"
                        else:
                            nombre_archivo = nombre_nuevo_prompt
                        
                        ruta_prompt = Path("prompts") / nombre_archivo
                        
                        with open(ruta_prompt, 'w', encoding='utf-8') as f:
                            f.write(contenido_nuevo_prompt)
                        
                        st.success(f"✅ Prompt guardado: {ruta_prompt}")
                        st.rerun()
                        
                    except Exception as e:
                        st.error(f"❌ Error guardando prompt: {e}")
                else:
                    st.warning("⚠️ Por favor, completa nombre y contenido")
        
        with tab3:
            st.markdown("### 📁 Prompts Disponibles")
            
            try:
                prompts_disponibles = listar_prompts_disponibles()
                
                if prompts_disponibles:
                    for prompt in prompts_disponibles:
                        with st.container():
                            col1, col2, col3 = st.columns([2, 1, 1])
                            
                            with col1:
                                st.write(f"**📝 {prompt['nombre']}**")
                                st.caption(f"Archivo: {prompt['archivo']}")
                            
                            with col2:
                                if st.button(f"👁️ Ver", key=f"ver_{prompt['nombre']}"):
                                    try:
                                        with open(prompt['ruta'], 'r', encoding='utf-8') as f:
                                            contenido = f.read()
                                        st.text_area(f"Contenido de {prompt['nombre']}:", contenido, height=300, disabled=True)
                                    except Exception as e:
                                        st.error(f"Error: {e}")
                            
                            with col3:
                                if st.button(f"🗑️ Eliminar", key=f"del_{prompt['nombre']}", type="secondary"):
                                    if st.session_state.get(f"confirmar_del_{prompt['nombre']}", False):
                                        try:
                                            os.remove(prompt['ruta'])
                                            st.success(f"✅ Prompt {prompt['nombre']} eliminado")
                                            st.rerun()
                                        except Exception as e:
                                            st.error(f"❌ Error eliminando: {e}")
                                    else:
                                        st.session_state[f"confirmar_del_{prompt['nombre']}"] = True
                                        st.warning("⚠️ Haz clic de nuevo para confirmar")
                            
                            st.divider()
                else:
                    st.info("No hay prompts disponibles")
                    
            except Exception as e:
                st.error(f"Error listando prompts: {e}")

def realizar_consulta_inteligente_original(consulta, nombre_base):
    """Función original mantenida para compatibilidad"""
    return realizar_consulta_inteligente(consulta, nombre_base, None)
def realizar_busqueda(consulta, nombre_base, num_resultados=5):
    """Realiza búsqueda en la base"""
    try:
        # Cargar modelo
        modelo = cargar_modelo()
        if not modelo:
            return []
        
        # Conectar a base
        collection, total_docs = conectar_base(nombre_base)
        if not collection:
            st.error(f"No se pudo conectar a la base {nombre_base}")
            return []
        
        # Generar embedding
        embedding = modelo.encode([consulta])
        
        # Buscar
        resultados = collection.query(
            query_embeddings=embedding.tolist(),
            n_results=num_resultados
        )
        
        # Procesar resultados
        fragmentos = []
        if resultados['documents'] and resultados['documents'][0]:
            documentos = resultados['documents'][0]
            metadatos = resultados['metadatas'][0] if resultados['metadatas'] else [{}] * len(documentos)
            distancias = resultados['distances'][0] if resultados['distances'] else [0] * len(documentos)
            
            for i, (doc, meta, dist) in enumerate(zip(documentos, metadatos, distancias)):
                fragmentos.append({
                    'contenido': doc,
                    'fuente': meta.get('source', 'Fuente desconocida'),
                    'relevancia': 1 - dist,
                    'chunk_id': meta.get('chunk_id', i),
                    'metadata': meta
                })
        
        return fragmentos
        
    except Exception as e:
        st.error(f"Error en búsqueda: {e}")
        return []

def sidebar():
    """Sidebar con configuración"""
    with st.sidebar:
        st.header("🔧 Configuración")
        
        # Estado de la API y motor inteligente
        api_ok = verificar_api_key()
        motor_ok = verificar_motor_inteligente()
        
        if motor_ok:
            st.success("✅ RAG Inteligente disponible")
        elif api_ok:
            st.warning("⚠️ API OK, motor no disponible")
        else:
            st.warning("⚠️ Google API no configurada")
        
        # Bases disponibles
        bases = obtener_bases_disponibles()
        if not bases:
            st.error("❌ No hay bases disponibles")
            return None, None, None
        
        st.success(f"✅ {len(bases)} bases disponibles")
        
        # Selector de base
        base_seleccionada = st.selectbox(
            "📚 Selecciona base:",
            bases,
            help="Base de datos a consultar"
        )
        
        # Información de la base
        collection, total_docs = conectar_base(base_seleccionada)
        if collection:
            st.metric("📄 Documentos", total_docs)
            st.success(f"✅ Conectado")
        else:
            st.error("❌ Error de conexión")
        
        # Configuración avanzada
        st.subheader("⚙️ Parámetros")
        num_resultados = st.slider("Resultados:", 1, 10, 5)
        
        # Modo de búsqueda
        st.subheader("🧠 Modo de búsqueda")
        if motor_ok:
            # RAG Inteligente por defecto, opción para desactivar
            usar_inteligente = not st.checkbox(
                "📄 Usar búsqueda tradicional", 
                value=False,
                help="Cambiar a fragmentos básicos (sin IA)"
            )
            if usar_inteligente:
                st.success("🧠 Modo: RAG Inteligente con IA")
            else:
                st.info("📄 Modo: Búsqueda tradicional")
        else:
            usar_inteligente = False
            st.warning("⚠️ Configura Google API para usar RAG Inteligente")
        
        return base_seleccionada, num_resultados, usar_inteligente

def main():
    configurar_pagina()
    
    # Título
    st.title("⚖️ Sistema RAG Jurídico - App Completa")
    st.markdown("**Análisis inteligente de documentos legales con IA integrada**")
    
    # Mostrar modo actual
    motor_disponible = verificar_motor_inteligente()
    if motor_disponible:
        st.success("🧠 **Modo por defecto:** RAG Inteligente con Gemini AI activado")
    else:
        st.warning("⚠️ **Modo actual:** Búsqueda tradicional (configura Google API para IA)")
    
    st.markdown("---")
    
    # Sidebar
    config = sidebar()
    if not config[0]:  # No hay bases
        st.stop()
    
    base_seleccionada, num_resultados, usar_inteligente = config
    
    # Área principal
    st.subheader("🔍 Realizar Consulta")
    
    # Gestión de prompts (solo si el motor inteligente está disponible)
    if motor_disponible:
        gestionar_prompts()
    
    # Campo de consulta
    consulta = st.text_area(
        "❓ Escribe tu consulta:",
        placeholder="Ejemplo: ¿Qué establece la teoría general del proceso sobre el debido proceso?",
        height=100
    )
    
    # Botones
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        if usar_inteligente:
            buscar = st.button("🧠 Consultar con IA", type="primary")
        else:
            buscar = st.button("🔍 Buscar Fragmentos", type="primary")
    
    with col2:
        if st.button("🔄 Limpiar"):
            st.rerun()
    
    with col3:
        if st.button("ℹ️ Ayuda"):
            with st.expander("💡 Información de uso"):
                st.markdown("""
                **🧠 RAG Inteligente (por defecto):**
                - Respuestas procesadas con Gemini AI
                - Análisis contextual profesional
                - Citas específicas de fuentes
                
                **📄 Búsqueda tradicional:**
                - Fragmentos de texto directo
                - Sin procesamiento de IA
                - Activar en configuración
                """)
    
    # Realizar búsqueda
    if buscar:
        if not consulta.strip():
            st.warning("⚠️ Ingresa una consulta")
        else:
            if usar_inteligente:
                # Usar RAG Inteligente con prompt seleccionado
                prompt_especifico = st.session_state.get('prompt_seleccionado', None)
                
                with st.spinner("🧠 Procesando consulta inteligente..."):
                    resultado = realizar_consulta_inteligente(consulta, base_seleccionada, prompt_especifico)
                    
                    if not resultado:
                        st.error("❌ Error en consulta inteligente")
                    else:
                        # Mostrar respuesta inteligente
                        st.success("✅ Respuesta generada con IA")
                        
                        # Mostrar información del prompt usado
                        if prompt_especifico:
                            st.info(f"🎯 **Prompt utilizado:** {prompt_especifico}")
                        else:
                            st.info("🔄 **Prompt utilizado:** Sistema predeterminado")
                        
                        # Métricas
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("📚 Base", base_seleccionada)
                        with col2:
                            st.metric("🧠 Modo", "Inteligente")
                        with col3:
                            st.metric("📄 Fuentes", len(resultado.get('fuentes_consultadas', [])))
                        
                        st.markdown("---")
                        
                        # Respuesta principal
                        st.subheader("🎯 Respuesta")
                        respuesta_container = st.container()
                        with respuesta_container:
                            st.markdown(resultado.get('respuesta', 'Sin respuesta'))
                        
                        # Referencias bibliográficas
                        if 'fragmentos' in resultado and resultado['fragmentos']:
                            st.markdown("---")
                            st.subheader("📚 Referencias Bibliográficas")
                            
                            # Mostrar referencias en formato académico
                            referencias_container = st.container()
                            with referencias_container:
                                for i, fragmento in enumerate(resultado['fragmentos'], 1):
                                    metadata = fragmento.get('metadata', {})
                                    autor = metadata.get('autor', 'Autor desconocido')
                                    titulo = metadata.get('titulo', metadata.get('filename', 'Sin título'))
                                    volumen = metadata.get('edicion', 'Sin edición')
                                    editorial = metadata.get('editorial', 'Editorial desconocida')
                                    anio = metadata.get('anio', 'Sin fecha')
                                    chunk_id = fragmento.get('chunk_id', metadata.get('chunk_id', 'N/A'))
                                    pagina = metadata.get('pagina', 's/p')
                                    relevancia = fragmento.get('relevancia', 0)
                                    isbn = metadata.get('isbn', 'Sin ISBN')

                                    referencia = f"{autor} – {titulo}"
                                    if anio and anio != 'Sin fecha':
                                        referencia += f" ({anio})"
                                    if volumen and volumen != 'Sin edición':
                                        referencia += f", {volumen}"
                                    if pagina and pagina not in ['sin página', 's/p']:
                                        referencia += f", pág. {pagina}"

                                    # Mostrar la referencia y metadatos de forma clara y reutilizable
                                    with st.expander(f"📖 [{i}] {referencia}"):
                                        col1_ref, col2_ref = st.columns([3, 1])
                                        with col1_ref:
                                            st.write(f"**👤 Autor:** {autor}")
                                            st.write(f"**� Título:** {titulo}")
                                            st.write(f"**📚 Volumen/Edición:** {volumen}")
                                            st.write(f"**🏢 Editorial:** {editorial}")
                                            st.write(f"**📅 Año:** {anio}")
                                            st.write(f"**🔢 ISBN:** {isbn}")
                                            if pagina and pagina not in ['sin página', 's/p']:
                                                st.write(f"**📄 Página:** {pagina}")
                                            st.markdown("**📝 Fragmento consultado:**")
                                            st.text_area(
                                                "Contenido:",
                                                fragmento.get('contenido', ''),
                                                height=120,
                                                key=f"fragmento_{i}",
                                                disabled=True
                                            )
                                        with col2_ref:
                                            st.metric("🎯 Relevancia", f"{relevancia:.3f}")
                                            st.metric("🔢 Fragmento", chunk_id)
                            
                            # Botón para copiar todas las referencias
                            referencias_texto = ""
                            for i, fragmento in enumerate(resultado['fragmentos'], 1):
                                metadata = fragmento.get('metadata', {})
                                try:
                                    from extractor_metadatos import generar_cita_bibliografica
                                    referencia = generar_cita_bibliografica(metadata)
                                except ImportError:
                                    autor = metadata.get('autor', 'Autor desconocido')
                                    titulo = metadata.get('titulo', metadata.get('filename', 'Sin título'))
                                    pagina = metadata.get('pagina', 'sin página')
                                    anio = metadata.get('anio', '')
                                    
                                    referencia = f"{autor} – {titulo}"
                                    if anio:
                                        referencia += f" ({anio})"
                                    if pagina != 'sin página' and pagina != 's/p':
                                        referencia += f", pág. {pagina}"
                                
                                referencias_texto += f"[{i}] {referencia}\n"
                            
                            st.text_area(
                                "📋 Referencias en formato cita:",
                                referencias_texto,
                                height=100,
                                help="Copia estas referencias para uso académico"
                            )
                        
                    # Opciones adicionales
                    st.markdown("---")
                    st.subheader("💾 Exportar Análisis")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Exportar a PDF con información del prompt
                        prompt_especifico = st.session_state.get('prompt_seleccionado', None)
                        pdf_buffer = crear_pdf_reporte(consulta, base_seleccionada, resultado, "inteligente", prompt_especifico)
                        nombre_pdf = generar_nombre_archivo(consulta, "pdf")
                        
                        st.download_button(
                            label="📄 Descargar PDF",
                            data=pdf_buffer.getvalue(),
                            file_name=nombre_pdf,
                            mime="application/pdf",
                            type="primary"
                        )
                    
                    with col2:
                        # Exportar resultado inteligente JSON
                        resultado_export = {
                            'consulta': consulta,
                            'base': base_seleccionada,
                            'modo': 'inteligente',
                            'timestamp': datetime.now().isoformat(),
                            'resultado': resultado
                        }
                        
                        nombre_json = generar_nombre_archivo(consulta, "json")
                        
                        st.download_button(
                            label="📥 Descargar JSON",
                            data=json.dumps(resultado_export, indent=2, ensure_ascii=False),
                            file_name=nombre_json,
                            mime="application/json"
                        )
                        
                        with col2:
                            if st.button("🔄 Nueva consulta"):
                                st.rerun()
            
            else:
                # Usar búsqueda tradicional
                with st.spinner("🔍 Buscando..."):
                    fragmentos = realizar_busqueda(consulta, base_seleccionada, num_resultados)
                    
                    if not fragmentos:
                        st.warning("❌ No se encontraron resultados")
                    else:
                        # Mostrar resultados tradicionales
                        st.success(f"✅ {len(fragmentos)} resultados encontrados")
                        
                        # Métricas
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("📚 Base", base_seleccionada)
                        with col2:
                            st.metric("📄 Fragmentos", len(fragmentos))
                        with col3:
                            mejor_relevancia = max(f['relevancia'] for f in fragmentos)
                            st.metric("🎯 Mejor relevancia", f"{mejor_relevancia:.3f}")
                        
                        st.markdown("---")
                        
                        # Resultados detallados
                        for i, fragmento in enumerate(fragmentos, 1):
                            metadata = fragmento.get('metadata', {})
                            with st.expander(f"📖 Resultado {i} - Relevancia: {fragmento['relevancia']:.3f}"):
                                col1, col2 = st.columns([3, 1])
                                with col1:
                                    st.markdown(f"**👤 Autor:** {metadata.get('autor', 'Autor desconocido')}")
                                    st.markdown(f"**📖 Título:** {metadata.get('titulo', metadata.get('filename', 'Sin título'))}")
                                    st.markdown(f"**� Volumen/Edición:** {metadata.get('edicion', 'Sin edición')}")
                                    st.markdown(f"**🏢 Editorial:** {metadata.get('editorial', 'Editorial desconocida')}")
                                    st.markdown(f"**📅 Año:** {metadata.get('anio', 'Sin fecha')}")
                                    st.markdown(f"**🔢 Chunk ID:** {fragmento.get('chunk_id', metadata.get('chunk_id', 'N/A'))}")
                                    st.markdown(f"**📄 Página:** {metadata.get('pagina', 's/p')}")
                                    st.markdown("**📝 Contenido:**")
                                    st.text_area(
                                        "Contenido:",
                                        fragmento['contenido'],
                                        height=150,
                                        key=f"content_{i}",
                                        disabled=True
                                    )
                                with col2:
                                    st.metric("🎯 Relevancia", f"{fragmento['relevancia']:.3f}")
                                    st.metric("🔢 Fragmento", fragmento.get('chunk_id', metadata.get('chunk_id', 'N/A')))
                        
                        # Opciones adicionales
                        st.markdown("---")
                        st.subheader("💾 Exportar Resultados")
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            # Exportar a PDF
                            pdf_buffer = crear_pdf_reporte(consulta, base_seleccionada, fragmentos, "tradicional")
                            nombre_pdf = generar_nombre_archivo(consulta, "pdf")
                            
                            st.download_button(
                                label="📄 Descargar PDF",
                                data=pdf_buffer.getvalue(),
                                file_name=nombre_pdf,
                                mime="application/pdf",
                                type="primary"
                            )
                        
                        with col2:
                            # Exportar resultados JSON
                            resultados_json = {
                                'consulta': consulta,
                                'base': base_seleccionada,
                                'modo': 'tradicional',
                                'timestamp': datetime.now().isoformat(),
                                'resultados': fragmentos
                            }
                            
                            nombre_json = generar_nombre_archivo(consulta, "json")
                            
                            st.download_button(
                                label="📥 Descargar JSON",
                                data=json.dumps(resultados_json, indent=2, ensure_ascii=False),
                                file_name=nombre_json,
                                mime="application/json"
                            )
                        
                        with col2:
                            if st.button("🧠 Análisis Inteligente"):
                                st.info("💡 Activa el modo RAG Inteligente en configuración (desactivar búsqueda tradicional)")

if __name__ == "__main__":
    main()