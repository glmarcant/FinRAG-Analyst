# SEC RAG System

Un sistema RAG (Retrieval-Augmented Generation) para analizar filings 10-K de la SEC usando la API de Gemini. Permite realizar consultas en lenguaje natural sobre información financiera de empresas públicas estadounidenses, combinando búsqueda semántica vectorial con generación de respuestas mediante IA.

## Setup

1. **Crear entorno virtual**:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Linux/Mac
   # o
   .venv\Scripts\activate     # Windows
   ```

2. **Instalar dependencias**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Configurar API key**:
   - Conseguir API key gratuita en [aistudio.google.com](https://aistudio.google.com)
   - Copiar `.env.example` a `.env`
   - Agregar tu API key en el archivo `.env`

4. **Configurar User-Agent para SEC**:
   - Editar `sec_rag/config.py`
   - Reemplazar el placeholder con tu nombre y email real (requerido por SEC)

## Pipeline

El sistema funciona en 4 etapas principales:

1. **ingest** → Descarga filings 10-K desde SEC EDGAR
2. **financials** → Extrae datos financieros estructurados
3. **index** → Crea índice vectorial para búsqueda semántica
4. **cli** → Interfaz de consultas en lenguaje natural