# 📖 Sistema de Lectura Inmersiva con Audio Generativo y Gaze Tracking
Este proyecto implementa un sistema de lectura inmersiva que sincroniza la narración escrita con sonidos generados dinámicamente, en función del contenido del texto y la mirada del lector. Utiliza generación de audio a través de AudioLDM, construcción contextual de prompts mediante Gemini (Google) y seguimiento ocular en tiempo real para mejorar la experiencia inmersiva del lector.

## 🚀 Características Principales
- 🎧 Audio Generativo Inmersivo con AudioLDM, usando prompts acústicamente detallados.
- 🤖 Generación Inteligente de Prompts mediante Gemini y LangChain, con reglas estrictas de calidad y relevancia.
- 👁 Gaze Tracking para sincronizar reproducción sonora según la línea de lectura activa.
- 🔗 Integración modular y fácilmente escalable, con procesamiento por párrafos y líneas.

## 🧠 Arquitectura del Sistema
- Entrada: Texto narrativo dividido por párrafos y líneas.
- Análisis: Cada línea es evaluada por Gemini, que genera (si es relevante) un prompt de audio inmersivo siguiendo un set estricto de criterios.
- Generación: Los prompts se envían a AudioLDM para producir los sonidos.
- Sincronización: Un sistema de eye tracking detecta la posición de lectura del usuario y reproduce el audio correspondiente en tiempo real.

## 🧩 Componentes Técnicos
📝 Prompt Engineering con LangChain y Gemini

Se ha diseñado cuidadosamente un prompt base especializado que guía a Gemini para generar solo sonidos estrictamente justificados según el contenido narrativo.

Características del diseño del prompt:
- Una línea del texto solo genera un sonido si añade valor emocional, atmosférico o narrativo.
- Cada prompt incluye:
    - Objeto o sujeto sonoro (e.g., "wooden door").
    - Acción acústica (e.g., "creaking open").
    - Ambiente y superficie (e.g., "in a quiet room").
    - Cualidades del sonido (e.g., "metallic", "muffled").
    - Estado emocional/contextual (e.g., "eerie", "tense").
- Duración realista para cada sonido: entre 1 y 6 segundos.
- Formato de salida: lista JSON con línea afectada, prompt generado y duración estimada.

Ejemplo de Output:
```
[
  {
    "prompt": "Slow footsteps echoing in a narrow, dimly lit hallway",
    "linea": 1,
    "length": 2.0
  }
]
```

## 🧪 Pipeline Principal

El archivo main.py ejecuta el flujo completo:
- Carga el texto narrativo desde archivo.
- Divide por párrafos y analiza cada uno con audio_chain.
- Genera audios con AudioLDM desde los prompts.
- Inicia el sistema de gaze tracking y reproduce los sonidos sincronizados.

## Ejecutar con el comando
python main.py 2>/dev/null