# Strata-V100

**Un fork de [Strata](https://github.com/Niko1221/Strata) mantenido por la comunidad y centrado en la V100.** Este fork mantiene en funcionamiento el soporte de NVIDIA Volta (`sm_70`) y desarrolla y mide cambios de rendimiento en hardware Tesla V100. También integra periódicamente mejoras de upstream para funciones y correcciones que van más allá de la V100.

Strata es un runtime local de código abierto para [Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next). Combina la GPU, la memoria del sistema y el almacenamiento para ejecutar este modelo grande en un ordenador personal. Este fork conserva el resto del proyecto Strata. El hardware NVIDIA y AMD compatible, la interfaz web, la API compatible con OpenAI, las herramientas y la entrada de imágenes siguen siendo importantes. Las funciones de upstream también siguen siendo importantes.

> **Un agradecimiento enorme a Niko ([@Niko1221](https://github.com/Niko1221)) y a los colaboradores originales de Strata.** Este proyecto se apoya en su trabajo. El runtime del modelo, la arquitectura, las funciones de Strata y el trabajo que hizo posible ejecutar este modelo en local vienen del proyecto upstream. El fork para V100 no existiría sin esa base. Gracias por hacerlo posible y por seguir desarrollando Strata.

[Strata upstream](https://github.com/Niko1221/Strata) · [Issues](https://github.com/jmnargi/Strata-V100/issues) · [Pull requests](https://github.com/jmnargi/Strata-V100/pulls) · [Discussions](https://github.com/jmnargi/Strata-V100/discussions)

**Idiomas:** [English](README.md) · [Deutsch](README.de.md) · [Español](README.es.md) · [Français](README.fr.md) · [日本語](README.ja.md) · [Português (Brasil)](README.pt-BR.md) · [简体中文](README.zh-CN.md)

## Contenido

- [Modelos y cuantizaciones que se ejecutan aquí](#modelos-y-cuantizaciones-que-se-ejecutan-aquí)
- [En qué se centra este fork](#en-qué-se-centra-este-fork)
- [Documentación](#documentación)
- [Notas de versión](#notas-de-versión)
- [Benchmark V100](#benchmark-v100)
- [Trabajo de rendimiento en V100](#trabajo-de-rendimiento-en-v100)
- [Otro hardware](#otro-hardware)
- [Instalación y arranque](#instalación-y-arranque)
- [Uso de Strata](#uso-de-strata)
- [Si algo va mal](#si-algo-va-mal)
- [Contribuir](#contribuir)
- [Historia del proyecto](#historia-del-proyecto)
- [Créditos y licencia](#créditos-y-licencia)

## Modelos y cuantizaciones que se ejecutan aquí

Strata ejecuta **Qwen3.8-Flash-Next** a partir de siete familias de versiones cuantizadas, además de un **Qwen3.6-35B-A3B** más pequeño. El motor y el setup de este fork admiten todas
las filas de abajo; la última columna indica qué se ha ejecutado realmente de principio a fin en
el par de V100 de este equipo. Tamaños, calidad y velocidades por tamaño: [docs/MODELS.md](docs/MODELS.md).

| Modelo / cuantización | Expertos enrutados | De dónde vienen los pesos | Descarga | Ejecución en este par de V100 |
| --- | --- | --- | ---: | --- |
| **NVFP4 — `local-inference-lab` (QAD)** | NVFP4, grupo 16; atención y expertos compartidos MXFP8; tabla n-gram NVFP4 | [`local-inference-lab/Qwen3.8-Flash-Next-NVFP4`](https://huggingface.co/local-inference-lab/Qwen3.8-Flash-Next-NVFP4), convertido en tu PC ([NVFP4.md](docs/NVFP4.md#the-qad-distillation-local-inference-labqwen38-flash-next-nvfp4)) | 106 GB | sí — 62.3 tokens/s en decode |
| **NVFP4 — NVIDIA, entrenado para FP4 (`FP4-Local`, QAD)** | NVFP4, grupo 16 (escalas en el formato CSF de NVIDIA) | [`nvidia/Qwen3.8-Flash-Next-FP4-Local`](https://huggingface.co/nvidia/Qwen3.8-Flash-Next-FP4-Local), convertido en tu PC ([NVFP4.md](docs/NVFP4.md#nvidias-fp4-local-checkpoint)) | 102 GB (+110 GB durante la preparación) | compatible; no ejecutado aquí |
| **NVFP4 — la versión de NVIDIA** | NVFP4, grupo 16 | [`Maximilian228/Qwen3.8-Flash-Next-NVIDIA-NVFP4-Strata`](https://huggingface.co/Maximilian228/Qwen3.8-Flash-Next-NVIDIA-NVFP4-Strata) (pack ya preparado) | 128 GB | compatible; no ejecutado aquí |
| **NVFP4 — la versión de Strata** | NVFP4, grupo 16 | [`Maximilian228/Qwen3.8-Flash-Next-NVFP4-GPTQ-Strata`](https://huggingface.co/Maximilian228/Qwen3.8-Flash-Next-NVFP4-GPTQ-Strata) (pack ya preparado, GPTQ) | 128 GB | compatible; no ejecutado aquí |
| **Unsloth Dynamic v3.1 (1-4 bits)** | UD-IQ1_M, UD-Q2_0, UD-IQ2_XS, UD-Q2_K_XL, UD-IQ3_XXS, UD-Q3_K_XL, UD-IQ4_XS-v31 | [`unsloth/Qwen3.8-Flash-Next-GGUF`](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF), revisión del 2026-11-04 | 68-94 GB | sí — UD-IQ4_XS-v31, 81 tokens/s |
| **Unsloth Dynamic (revisión anterior)** | UD-IQ4_XS, UD-Q4_K_XL, UD-Q5_K_XL, UD-Q6_K_XL, Q8_0 | [`unsloth/Qwen3.8-Flash-Next-GGUF`](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF) | 94-188 GB | compatible; no ejecutado aquí |
| **i-quants y Q2_0 de ISTA-DASLab (2-3.5 bits)** | Q2_0, IQ2_XS, IQ3_XXS, IQ3_S | [`ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF`](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF) | 66-84 GB | sí — IQ3_S, el perfil diario anterior |
| **Gyro (agentionai), trellis** | Gyro-S TQ1_0, Gyro-M TQ2_0 (trellis por pares) | [`agentionai/Qwen3.8-Flash-Next-Gyro-GGUF`](https://huggingface.co/agentionai/Qwen3.8-Flash-Next-Gyro-GGUF) ([GYRO.md](docs/GYRO.md)) | 58.5 / 92 GB | sí — Gyro-M, unos 70 tokens/s |
| **Coder (ISTA-DASLab), la mitad de los expertos** | IQ1_M, IQ2_S-IQ4_XS como IQ3_S | [`ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-Coder-GGUF`](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-Coder-GGUF) | 58 GB | compatible; no ejecutado aquí |
| **OrcaRouter Uncensored** | IQ3_XXS | El Flash-Next Uncensored de `OrcaRouter` ([ORCA.md](docs/ORCA.md); un paso de empaquetado manual, no una opción del instalador) | 84 GB | compatible; no ejecutado aquí |
| **Qwen3.6-35B-A3B** (un modelo más pequeño) | UD-Q4_K_XL, UD-Q6_K_XL y Q8_0 de Unsloth, con la capa de borrador MTP | [`unsloth/Qwen3.6-35B-A3B-MTP-GGUF`](https://huggingface.co/unsloth/Qwen3.6-35B-A3B-MTP-GGUF) ([MODELS.md](docs/MODELS.md#qwen36-35b-a3b)) | 23-38 GB | compatible; no ejecutado aquí |

Las filas NVFP4 necesitan tarjetas NVIDIA (las RTX 50 multiplican a anchura FP4; las tarjetas más antiguas, incluida
la V100, a precisión de 8 bits), y las dos filas QAD se preparan en tu PC a partir de los safetensors de los autores:
aquí no se redistribuye ningún peso cuantizado suyo. Las filas Gyro y trellis necesitan el motor 0.1.45.1 o posterior;
las filas Unsloth con downs Q5_0, 0.1.45.7.

## En qué se centra este fork

- **Soporte de Tesla V100 / Volta (`sm_70`)**: conservar las compilaciones normales para V100 y el soporte del runtime. Las compilaciones para V100 usan un toolkit CUDA 12.x; CUDA 13 no compila código `sm_70`.
- **Máquina de prueba:** dos tarjetas Tesla V100-PCIE-16GB; la GPU 0 usa PCIe Gen3 x4 y la GPU 1 usa Gen3 x16. Las mediciones de pipeline de abajo usan las dos tarjetas. El benchmark histórico de una sola tarjeta usa solo la tarjeta x16. Un sistema con dos enlaces Gen3 x16 completos no se ha medido aquí.
- **Integración de upstream**: esto no es un reemplazo separado de Strata. Los cambios de upstream se fusionan periódicamente, con la compatibilidad de V100 comprobada y corregida donde hace falta. Las funciones nuevas de upstream pueden tener requisitos de hardware por encima de `sm_70`; no aceleran la V100 de forma automática.
- **La comunidad de Strata en general**: las mejoras, los informes de errores, las pruebas, la documentación y las contribuciones para otras plataformas compatibles son bienvenidos.

El soporte de V100 no implica que todas las funciones de upstream o todas las demás configuraciones de GPU se hayan probado en V100. Consulta el código, las notas de versión y los informes de medición correspondientes antes de confiar en una función específica de hardware.

## Documentación

El [índice de documentación](docs/README.md) enumera todas las páginas y su propósito. Estas páginas son la
referencia:

- [Flags del motor](docs/ENGINE_FLAGS.md): cada opción de línea de comandos, su valor y su valor por defecto.
- [Variables de entorno](docs/ENVIRONMENT.md): cada variable que leen el motor, el servidor y el
  terminal.
- [Servidor](docs/SERVER.md): opciones del servidor, las claves de configuración de ejecución y todos los endpoints de la API.
- [Terminal](docs/TERMINAL.md): cada comando y subcomando de `strata`.
- [Ajuste](docs/TUNING.md): los ajustes por objetivo, las optimizaciones y los resultados medidos.
- [Cómo funciona](docs/HOW_IT_WORKS.md) y el [glosario](docs/GLOSSARY.md): las partes del
  motor y los términos que usa esta documentación.
- [Detalles](docs/DETAILS.md): el registro de mediciones, los formatos de petición y el historial de ajustes.

Las páginas siguen el [estilo de documentación](docs/STYLE.md), que es ASD-STE100 (inglés técnico
simplificado).

## Notas de versión

### Motor 0.1.45.8 — 16 de noviembre de 2026

La [versión v0.1.45.8](https://github.com/jmnargi/Strata-V100/releases/tag/v0.1.45.8)
da al motor V100 actual una versión propia. Incluye la corrección del desplazamiento
de filas en el prefill FP16 de expertos, trabajo en decode NVFP4, ahorro opcional de
memoria HC_Q8 entre tarjetas, mejoras opcionales de puntuación de prompts y KV
transmitido, y progreso del prompt por slot. Este cambio de versión no modifica el
código de inferencia ni los ajustes por defecto. Las mediciones y los límites están
en [NVIDIA_V100.md](docs/NVIDIA_V100.md).

### Conjuntos Unsloth Dynamic v3.1 (1-4 bits) — 8 de noviembre de 2026

Resuelve [#95](https://github.com/jmnargi/Strata-V100/issues/95): el menú `--family unsloth` del setup antes se detenía en
UD-IQ4_XS (unos 4 bits). Ahora ofrece también los archivos Dynamic v3.1 de Unsloth, de ~1 a ~4 bits, en la revisión
que Unsloth publicó el 2026-11-04 (commit `d9620119`):

| Tamaño | Descarga | Expertos enrutados | RAM con todos los expertos en RAM | Notas |
| --- | ---: | ---: | ---: | --- |
| UD-IQ1_M | 68.0 GB | 32.4 GiB | ~48 GB | el más pequeño; el más débil en trabajo cuidadoso o largo |
| UD-Q2_0 | 70.5 GB | 34.8 GiB | ~51 GB | expertos Q2_0, como el pack de 2 bits más rápido |
| UD-IQ2_XS | 75.8 GB | 39.4 GiB | ~55 GB | 1 capa con downs Q5_0 |
| UD-Q2_K_XL | 79.6 GB | 42.8 GiB | ~59 GB | 1 capa con downs Q5_0 |
| UD-IQ3_XXS | 83.6 GB | 46.8 GiB | ~63 GB | sin Q5_0 |
| UD-Q3_K_XL | 89.5 GB | 52.3 GiB | ~68 GB | 20 capas con downs Q5_0 |
| UD-IQ4_XS-v31 | 93.7 GB | 56.2 GiB | ~72 GB | 33 capas con downs Q5_0; nombre propio en el catálogo, porque sus archivos del Hub se llaman UD-IQ4_XS |

Los cinco archivos que ya se ofrecían (UD-IQ4_XS, UD-Q4_K_XL, UD-Q5_K_XL, UD-Q6_K_XL, Q8_0) siguen fijados al
commit anterior, y el segundo diseño de UD-Q4_K_XL (cuatro fragmentos, con la tabla n-gram al final) en el commit
nuevo se acepta desde archivos que ya están en disco. UD-IQ1_S sigue rechazado: mezcla dos formatos de experto en una capa.

El trabajo de motor que esto necesitó: el producto escalar de expertos Q5_0. Su desplazamiento de -16 por peso se aplicaba a la suma de las
activaciones originales (la forma CUDA de llama.cpp), así que una capa cuya proyección down es Q5_0 salía 2-3% menos precisa por experto
que cualquier otro formato down. Ahora multiplica la suma de activaciones cuantizada, como las rutas Q5_1 y Q4_0 y ggml-cpu
(CUDA y SYCL). Medido en este equipo con la paridad sintética de expertos (`native_expert_parity --synthetic`, gate-up IQ2_XXS /
IQ2_S / IQ2_XS / IQ3_XXS / IQ4_XS con downs Q5_0, error L1 relativo contra una referencia float): **2.40e-2 ..
3.18e-2 antes, 1.25e-2 .. 1.30e-2 después**, lo mismo que los downs IQ4_NL y Q2_0 (1.29e-2 .. 1.31e-2) con las mismas semillas.
Los lanzamientos de expertos agrupados siguen siendo idénticos bit a bit a los lanzamientos previos a la ventana (`native_grouped_parity`: Q5_0 añadido a cada
formato gate/up, y las formas Flash-Next que usan estos archivos, 0 fallos). Versión del motor: 0.1.45.7; el setup rechaza un
motor anterior para estos siete conjuntos antes de descargar nada.

Los cinco archivos fijados, sus bytes y cualquier otro formato no cambian: solo los pesos de *experto* Q5_0 leen este kernel
(la ruta densa Q5_0 es `native_mmvq`, sin tocar), y su error de experto completo contra la referencia float pasa de
2.4-3.2e-2 a 1.25-1.30e-2.

### Portes de rendimiento seleccionados de upstream v0.2.11 — 5 de noviembre de 2026

Esta actualización porta de forma selectiva optimizaciones de upstream; no es una
integración completa de v0.2.11. Añade kernels de expertos trellis más rápidos, búferes temporales de prompt
más pequeños, menos asignaciones del lector de disco PLE y una corrección de zero-spin
de CPU/CUDA en Linux. La versión del motor sigue siendo 0.1.45.6.

**Medido en la máquina de dos V100 de este fork:** dos tarjetas V100-PCIE-16GB a
175 W (GPU0 Gen3 x4, GPU1 x16), un Ryzen 5 3600 y Gyro-M abliterated Huihui
TQ2_0. El perfil usó split 16, dos ventanas de pipeline, MTP 4, KV int8
y contexto de 524,288 tokens con YaRN 2.

| Tokens de prompt | Tiempo de prompt de control | Group scratch + WMMA desactivado |
| ---: | ---: | ---: |
| 2,048 | 2,360.8 ms | 2,214.3 ms |
| 8,192 | 6,932.7 ms | 6,541.0 ms |
| 32,768 | 18,008.6 ms | 17,636.7 ms |

Son medias de ejecuciones en frío e intercaladas A-B-A-B, con cuatro peticiones por tamaño
y por rama. Ambas ramas ya incluían los portes trellis, HC y PLE. Esto no es
una ganancia medida en todos los tamaños de contexto ni una garantía de velocidad de la petición completa.

La combinación opcional medida es `STRATA_TQ_SUB8=1 STRATA_EXPERT_PAIR=0
STRATA_PF_HC_REGION=1 STRATA_PF_GROUP_SCRATCH=1 STRATA_PF_WMMA=0`.
Group scratch puede cambiar el redondeo de coma flotante. El cambio de asignación PLE
no necesita ningún flag. La corrección de zero-spin se aplica con `STRATA_POOL_SPIN_US=0`; no
demuestra un ahorro con el ajuste diario de 100.

Consulta el [informe de portes para V100](docs/NVIDIA_V100.md#selected-upstream-0211-ports-gyro-m-2026-11-05)
para ver mediciones individuales, protecciones de opciones, comprobaciones numéricas, límites de compilación
y candidatos probados que no se conservaron. No cambia ningún valor por defecto del servicio.

### Sincronización seleccionada de upstream — 19 de octubre de 2026

Esta rama importa cambios compartidos del motor CUDA y del servidor desde el commit
`3d1fb480` de upstream. Conserva los kernels Volta del fork, el crecimiento K/V del pipeline
y los cambios de caché de conversaciones. Excluye paquetes de escritorio no relacionados y
actualizaciones separadas de los backends Metal, AMD e Intel. Consulta el
[alcance de la sincronización](improvements/RELEASE-2026-10-19-V100.md).
Usa CUDA 12.8 con `sm_70`. Una opción importada no demuestra una ganancia de rendimiento
en V100.

### Integración de upstream v0.1.39

Esta actualización integra [Strata v0.1.39](https://github.com/Niko1221/Strata/releases/tag/v0.1.39), incluidas las versiones intermedias. Conserva los expertos de núcleos tensoriales para V100 del fork, los cambios de atención, el split asimétrico de GPU y la caché de conversaciones en disco.

- **Peticiones en paralelo:** define `"parallel": 2` en la configuración del modelo, o usa `./setup.sh --parallel 2`. El setup escribe este valor por defecto para una configuración Volta nueva y conserva un ajuste existente. El motor puede hacer decode de conversaciones separadas a la vez. Las peticiones adicionales esperan a un hueco libre. Los huecos usan memoria de GPU adicional y pueden reducir la velocidad de una sola petición. Si dos sesiones de hueco no caben, el motor lo indica y atiende una petición a la vez, sin cambiar la configuración. Consulta [batch slots](docs/BATCHING.md) para ver ajustes de exactitud, costes de memoria y límites.
- **APIs e interfaz:** la actualización añade la API Responses de OpenAI, ajustes del modelo, monitorización de la caché de conversaciones y métricas por hueco. Consulta los [detalles de la API](docs/DETAILS.md#using-it).
- **Rendimiento y memoria:** incluye reducciones de lanzamientos en la pasada de verificación y anillos de prompt dimensionados en bytes. También añade páginas grandes de Linux, correcciones del presupuesto de RAM y opciones multi-GPU. Un `--layer-split` explícito ahora carga automáticamente solo los pesos densos de las capas propias de cada etapa (el flag `--trim-stage-weights` de upstream y `STRATA_STAGE_TRIM` ya no existen). La VRAM liberada va a la caché de expertos. Una optimización para GPU más nuevas no implica una mejora de velocidad en V100. Las rutas opcionales que cambian el redondeo se deben medir antes de usarlas.
- **Otras plataformas:** las rutas de Intel Arc, CPU más antiguas y AMD adicionales de upstream siguen disponibles. Esta máquina no aporta validación de hardware para esas plataformas.

La V100 necesita CUDA 12.x y una compilación que incluya `sm_70`. No instales el motor CUDA 13 en una V100. Se recomienda configurar el paralelismo de forma explícita en instalaciones existentes; una actualización no sobrescribe la configuración del modelo.

**Medido en las dos tarjetas V100 (4 de octubre de 2026, compilación final).** Una ejecución controlada de antes y después
de la compilación integrada contra el runtime instalado previamente, de una petición a la vez
(sin huecos de lote), en frío, con Qwen3.8-Flash-Next Q2_0 y KV int8 en el contexto de 524,288 tokens,
el split de capas 20/28 y MTP (`--spec 8`) sin cambios; dos peticiones nuevas por tamaño, medianas:

| Prompt objetivo | Antes (tok/s) | Integrado (tok/s) | Cambio |
| ---: | ---: | ---: | ---: |
| 2K | 701.41 | 718.5 | +2.44 % |
| 8K | 1,402.82 | 1,458.0 | +3.93 % |
| 32K | 2,262.4 | 2,283.7 | +0.94 % |

Cada valor es la mediana de dos peticiones nuevas (`reused=0`). Las ganancias son pequeñas; dos repeticiones
por tamaño no son una afirmación estadística. Los seis primeros tokens emparejados coincidieron con los de la línea base.
No se afirma que las respuestas completas sean idénticas: el nivel adaptativo de expertos mantiene una mezcla distinta
de expertos en la GPU y en la CPU para la compilación integrada, y las dos rutas redondean
de forma distinta. No se afirma ningún cambio de decode ni de calidad. Estos son los valores medidos de la compilación
final (SHA-256 del motor `e9ad337a…`), recogidos en el
[informe completo](benchmarks/v100-q2_0-upstream-v0139-2026-10-04.md).

Los huecos en paralelo con el contexto de 524,288 tokens dependen del split y de la memoria que queda en la
última etapa; el caso de reserva medido (se pidieron 2, se atendió una petición a la vez, configuración
intacta) se describe en [docs/BATCHING.md](docs/BATCHING.md).

### Rendimiento del pipeline con dos V100 — 15 de octubre de 2026

El perfil de servicio medido activa `--pipeline-windows 2` con el orden de GPU
`0,1` y split por capas `16`. Usa dos tarjetas V100 de 16 GiB a 175 W,
un Ryzen 5 3600, el modelo nativo IQ3_S de formato mixto, KV int8 y MTP
con `--spec 8`. La capacidad de contexto configurada sigue siendo 524,288 tokens.
El prompt medido más grande es de 32,768 tokens.

**Seguridad en respuestas largas:** usa una compilación con la
[corrección del crecimiento KV del pipeline](benchmarks/v100-pipeline-kv-growth-fix-2026-10-16.md)
cuando combines este perfil con `STRATA_KV_GROW=1`. El bucle de pipeline anterior
podía acceder a celdas KV sin mapear durante una respuesta larga. La corrección
agota el trabajo especulativo antes de ampliar el mapeo y después reanuda
el pipeline. Las mediciones de velocidad con salidas cortas de abajo no ejercitaron
ese límite.

#### Prefill de prompt nuevo hasta 32K

Cada valor es la mediana de tres peticiones con tokens exactos, sin reutilización de prompt
y con un límite de salida de 16 tokens. El tiempo de prefill excluye la generación de la respuesta.
El rendimiento es el número de tokens de prompt dividido por el tiempo de prefill mediano.

| Tokens de prompt | Tiempo de prefill anterior | Tiempo de prefill con pipeline | Tokens/s anteriores | Tokens/s con pipeline | Cambio de rendimiento |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 2,048 | 2.089 s | 2.090 s | 980.6 | 979.8 | −0.09% |
| 8,192 | 5.487 s | 5.494 s | 1,493.0 | 1,491.2 | −0.13% |
| 16,384 | 8.828 s | 8.857 s | 1,855.9 | 1,849.8 | −0.33% |
| 32,768 | 15.668 s | 15.677 s | 2,091.4 | 2,090.2 | −0.06% |

El prefill de un prompt nuevo es prácticamente igual. Estas diferencias pequeñas
no demuestran una regresión significativa.

#### Decode de conversación

La comparación HTTP separada usa un prompt inicial de 8,828 tokens y seis
turnos de continuación. El contexto llega a aproximadamente 11.9K tokens. Cada valor
es la mediana de cuatro peticiones, con el orden de las ramas invertido en rondas alternas
y un límite de salida de 256 tokens. Los tokens de prompt nuevos son los tokens procesados en
ese turno, no el tamaño total del contexto.

| Turno | Tokens de prompt nuevos | Tokens/s de decode anteriores | Tokens/s de decode con pipeline | Cambio de decode |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 121 | 70.20 | 79.45 | +13.18% |
| 2 | 321 | 70.60 | 82.05 | +16.22% |
| 3 | 1,086 | 76.60 | 79.00 | +3.13% |
| 4 | 128 | 73.00 | 79.90 | +9.45% |
| 5 | 313 | 74.35 | 86.55 | +16.41% |
| 6 | 1,089 | 74.15 | 80.75 | +8.90% |

La mediana de las seis ratios de rendimiento de decode es **+11.31%**.
Esto no es una ganancia de decode medida en todos los tamaños de contexto hasta 32K.
Las longitudes generadas y el trabajo MTP difieren con el reparto automático de CPU y
el enrutado residente. La latencia de la petición completa es mixta; una mediana HTTP
de turno corto empeora. No interpretes la ganancia de decode como una aceleración garantizada
de la petición completa ni como una comparación con salidas emparejadas.

Un control separado con ajustes fijos coincide en las tres secuencias completas de salida de 16 tokens
con tamaños de prompt de 2K, 8K y 32K. No demuestra una salida idéntica
en todas las peticiones de producción.

Consulta el [informe de evaluación](benchmarks/v100-upstream-pr-evaluation-2026-10-15.md)
para ver tiempos exactos, procedencia de la compilación, pruebas rechazadas y comandos
de reproducción. Consulta la [guía del perfil de servicio](docs/NVIDIA_V100.md#dual-pcie-pipeline-service-profile)
antes de activar la opción en otra instalación.

### Caché KV TurboQuant — 25 de octubre de 2026

TurboQuant es un formato de caché KV opcional. Rota cada cabeza de 256 valores con una transformada de Hadamard y luego
cuantiza los valores rotados contra un libro de códigos Lloyd-Max. Cada cabeza conserva una norma FP16. Las anchuras son
de 3, 4, 6 y 8 bits por valor. El ajuste es solo de error cuadrático mínimo; este corte no tiene residuo QJL.

Usa `--kv tq6` (o `tq3`, `tq4`, `tq8`). La caché es solo residente: permanece en VRAM, no hay copia
en host y `--kv-resident` se rechaza. La capa de borrador MTP conserva su caché INT8. El setup no ofrece estos
valores. Nada es un valor por defecto, y el servicio diario sigue en int8.

Con la capacidad configurada de 524,288 tokens y las 12 capas de atención del modelo, la caché KV es:

| `--kv` | Bytes por celda, una capa | KV a capacidad completa, 12 capas | Cambio respecto a int8 |
| --- | ---: | ---: | ---: |
| int8 | 1,056 | 6.1875 GiB | — |
| tq8 | 1,032 | 6.046875 GiB | −0.14 GiB |
| tq6 | 776 | 4.546875 GiB | −1.64 GiB |
| tq4 | 520 | 3.046875 GiB | −3.14 GiB |
| tq3 | 392 | 2.296875 GiB | −3.89 GiB |

El ahorro principal de tq6 es 1.64 GiB a capacidad completa. Estas son capacidades calculadas, no reducciones
medidas de `nvidia-smi`: la caché de expertos llena la VRAM liberada, así que la memoria total de GPU se mantiene cerca de 15,600 MiB.

El lector de prompts de V100 es un segundo opt-in: `STRATA_PROMPT_ATTN_TQ=1` en sm_70. Decodifica teselas comprimidas
para la ruta de prompt; sin definir o con `0` se mantiene el fallback de atención de decode. No cambia los bytes de KV.

Medido en dos tarjetas Tesla V100 de 16 GB a 175 W con IQ3_S, `--spec 4`, `--layer-split 16`,
`--pipeline-windows 2`, un límite de salida en frío de 2,048 tokens y una ejecución por tamaño y rama. La rama tq6 tenía
el lector de prompts activado. Con 32,768 tokens de prompt, tq6 hizo prefill a 1,687 tokens/s frente a 2,117 de int8,
y decode a 120.2 tokens/s frente a 125.5. Con el lector activado, el tiempo total de la petición de 51,746 tokens de tq6
bajó de 40.3–40.4 s a 31.7–31.9 s, todavía más lento que los 25.8 s de int8. Estas mediciones no muestran una
ganancia de velocidad. Una prueba de aguja encontró 15 de 15 celdas en ambas ramas hasta 122,289 tokens de prompt; eso no es un
resultado de calidad general.

Consulta el [informe completo](benchmarks/v100-turboquant-2026-10-25.md) y
los [detalles](docs/DETAILS.md#tesla-v100-fork-benchmark).

### Trellis por pares Gyro-L en dos V100 — 31 de octubre de 2026

Esta rama lee `Qwen3.8-Flash-Next-Gyro-L-TQ3_0.gguf` de agentionai (108,323,559,808 bytes, revisión
`05892758a67efff0895d23c6df10a6fdb6be8cf4` de Hugging Face). Sus expertos enrutados se almacenan como los tipos trellis por pares
TQP5, TQP6 y TQP7 (tipos ggml 147 a 149) en una base rotada por Hadamard. El paso de empaquetado
(`tools/iq_pack.py --compat-bf16`) y los kernels se describen en [docs/GYRO.md](docs/GYRO.md). Se admite CPU y CUDA;
Metal y SYCL no tienen kernels trellis por pares. Esto necesita el motor 0.1.45.6 o posterior.

Primero hubo que reparar dos defectos del motor. Los kernels de expertos ya eran exactos bit a bit contra el
ggml de los autores:

- **La ventana de decode perdía la rotación.** En un pack plegado, la ventana rotaba la entrada de los expertos y después
  sobrescribía esa copia rotada con la no rotada antes de que los expertos la leyeran, así que las proyecciones gate/up
  se ejecutaban en la base equivocada en cada token decodificado. La salida de un solo disparo parecía plausible y luego entraba en bucle;
  el servidor estaba mal desde el primer token. La ventana ahora conserva una marca de "imagen escrita" por capa.
- **El ayudante de prompt del split por capas** (opcional) rotaba solo su salida SwiGLU y escribía las filas de un grupo en un
  búfer con el diseño de las teselas WMMA de la etapa. Ahora rota la entrada que recibe y posee su H.

Medido en el par de este fork: dos Tesla V100-PCIE-16GB (GPU 0 en PCIe Gen3 x4, GPU 1 en x16) a 175 W, un
Ryzen 5 3600, split por capas 14, `--spec 4 --spec-min-p 0.70`, KV int8 con crecimiento KV, `--pipeline-windows 2`,
contexto de 262,144 tokens, el entorno propio del servicio y repeticiones intercaladas:

| Tokens de prompt | Prefill (tokens/s) |
| ---: | ---: |
| 2,328 | 632 |
| 8,199 | 1,036-1,084 |
| 16,371 | 1,344-1,411 |
| 32,679 | 1,597-1,642 |

Las respuestas hacen decode de 68 a 75 tokens/s con contexto de 2K a 32K (medianas de respuestas repetidas de 520 tokens); el punto
de split no lo mueve más allá del ruido entre ejecuciones. Una lectura está limitada por los expertos que no están en VRAM: la
tarjeta x16 transmite 15.01 GiB por PCIe para una lectura de 2.3K tokens (perfil con split por capas 16) y las dos etapas se ejecutan
una tras otra, así que la tarjeta x4 queda inactiva unos 700 ms de cada lectura.

El pack sirve a través de la API compatible con OpenAI: "Paris" para la pregunta de la capital, un turno de continuación que
reutiliza 86 tokens de prompt, una petición de código, una llamada a la herramienta `get_weather` y una respuesta correcta a partir del resultado de la herramienta,
y una pregunta sobre un pasaje de 442 tokens respondida a partir del pasaje.

**Split por capas 14 en lugar de 16.** Catorce capas en la tarjeta x4 y 34 en la tarjeta x16 leen un 7.9% más rápido con
2.3K tokens (632 frente a 586 tokens/s) y un 6% más rápido con 8.2K. Rinden igual con 16.3K y son un 1.5% más lentas con 32.7K.
15 y 13 quedan en medio, y 20 es un 23% más lento con 2.3K y un 13% con 8.2K. Una lectura es en lo que emplea el tiempo
un turno agéntico, así que este perfil usa 14.

**Lo que se midió y no compensó en este par:** `--spec 6` y `--spec 8`, `--spec-min-p 0.50` y
`0.85`, un `STRATA_PREFILL_CPU_SHARE` fijo, `STRATA_PREFILL_CPU_SHARE_MAX` por encima de 6,144, `--prefill-pipe`,
`--max-context 131072`, `STRATA_HC_Q8=1` y un `--vram-reserve-mib` mayor. El propio ayudante de prompt es
más lento aquí (484 frente a 583 tokens/s con 2.3K): solo puede tomar los expertos de la primera etapa, y en este par
esa etapa es la tarjeta x4, que transmite 1.77 GiB de los 16.8 GiB que mueve una lectura. `--mtp` y el valor por defecto
`STRATA_ROUTE_TAIL_SKIP=7` merecen conservarse: el decode cae a 35.5 y 48 tokens/s sin ellos.

## Benchmark V100

La tabla de abajo registra una ejecución con una sola tarjeta en una Tesla V100-PCIE-16GB con PCIe Gen3 x16. La máquina de prueba también tiene una segunda V100 en PCIe Gen3 x4; no se usó en esta ejecución. Los resultados de una configuración de dos tarjetas o de una máquina con dos enlaces de ancho completo pueden diferir y podrían ser mejores. Es una medición reproducible, no una promesa de velocidad en otros sistemas.

| Prompt objetivo | Tokens de prompt medianos | Prefill (tokens/s) | Decode (tokens/s) | Temperatura máxima de GPU |
| ---: | ---: | ---: | ---: | ---: |
| 1K | 1,036 | 398.6 | 52.0 | 56 °C |
| 4K | 4,111 | 1,214.0 | 50.8 | 61 °C |
| 8K | 8,185 | 1,442.4 | 48.1 | 63 °C |
| 16K | 16,367 | 1,490.7 | 49.8 | 67 °C |
| 32K | 32,765 | 1,495.7 | 47.3 | 72 °C |
| 64K | 65,533 | 1,484.3 | 43.0 | 81 °C |
| 128K | 131,063 | 1,033.0 † | 40.9 | 84 °C |
| 256K | 256,080 | 617.8 † | 36.8 | 84 °C |

**Configuración de prueba:** Qwen3.8-Flash-Next Q2_0; solo una V100 (GPU1, PCIe Gen3 x16); Ryzen 5 3600; 48 GB DDR4-3200; CUDA 12.8. Contexto de 262,144 tokens; caché KV int8; prefill automático. MTP con `--spec 8` y suelo de borrador 0.70; variante de expertos emparejados; reserva de visión de 700 MiB. Cada valor es la mediana de tres peticiones nuevas sin caché (`reused=0`) con exactamente 256 tokens de salida. Los tiempos vienen del endpoint `/metrics` del motor.

Antes de cada petición, la tarjeta de refrigeración pasiva estuvo inactiva al menos 120 segundos. También alcanzó 55 °C o menos sin ralentización térmica por software durante 15 segundos. Las temperaturas iniciales fueron de 51–55 °C. Todas las peticiones están incluidas. Los prompts de 128K y 256K alcanzaron 84 °C y sufrieron ralentización térmica por software durante el procesamiento del prompt, a pesar del arranque en frío. Las ejecuciones largas pueden limitar la frecuencia. † indica ralentización durante el prompt.

Medido el 2026-10-02. Motor: versión del fork 0.1.31, commit de compilación `78417ea` (SHA-256 `512b1f25d60479a2ddb66fcf1ddca5407963e45378a3c8a463413ad159edae17`). El benchmark se registró antes de la integración posterior de upstream v0.1.36 y de los PR de atención para V100. No es una medición de la compilación actual de `main`. La tabla histórica se conserva con su procedencia original. No la interpretes como una comparación controlada con tablas anteriores ni con compilaciones actuales.

Datos en bruto: [`summary.json`](bench/results/2026-10-02-v100-single-cooled/summary.json), [`protocol.json`](bench/results/2026-10-02-v100-single-cooled/protocol.json), [`matrix.json`](bench/results/2026-10-02-v100-single-cooled/matrix.json), [`completed-requests.json`](bench/results/2026-10-02-v100-single-cooled/completed-requests.json) y [`telemetría de GPU`](bench/results/2026-10-02-v100-single-cooled/gpu.csv). Consulta el [método completo y el historial](docs/DETAILS.md#tesla-v100-fork-benchmark), las [instrucciones del benchmark](benchmarks/README.md) y los [informes de rendimiento de V100](benchmarks/).

Los pull requests recientes de V100 incluyen cambios medidos a nivel de kernel e informan de forma explícita cuando no se demuestran ganancias de extremo a extremo. Por ejemplo, el porte de atención de prompt del PR #600 de upstream informó un 27.9% menos de tiempo de kernel int8 en su forma medida. En las ejecuciones informadas, los cambios de prefill del modelo variaron de +1.3% a +3.0%. El porte no demostró una ganancia fiable de decode. Lee el [informe completo](benchmarks/v100-q2_0-pr600-2026-10-03.md) antes de comparar resultados.

## Trabajo de rendimiento en V100

El trabajo en V100 abarca la ruta de prompt, la ruta de decode, el uso de memoria, las lecturas de almacenamiento y la ejecución multi-GPU. Los cambios de abajo están en el historial fusionado del fork; cada enlace incluye su propio método de prueba, detalles de hardware y limitaciones. Las mejoras de tiempo de kernel no producen necesariamente el mismo porcentaje de ganancia en el rendimiento del modelo completo.

### Procesamiento de prompts

- **GEMM de núcleos tensoriales de BF16 a FP16 en Volta:** Volta no tiene soporte nativo de GEMM BF16. El fork convierte los pesos BF16 exactamente a FP16 y usa los núcleos tensoriales de la V100 para los GEMM de prompt en lugar del fallback escalar. Las imágenes de activación de prompt FP16 asociadas usan la representación de bits correcta.
- **Lecturas más rápidas de la tabla PLE:** cambia las lecturas PLE de POSIX a operaciones O_DIRECT de `io_uring` en cola, con un fallback síncrono. Una recogida PLE nueva de 8K medida bajó de unos 6.9 segundos a 7 milisegundos.
- **Prefill MMQ Q2_0:** activa la ruta cuantizada matriz-matriz Q2_0 de llama.cpp existente para el tipo de experto interno de Strata. Un prompt medido de 14,486 tokens mejoró un 6.5% en rendimiento de prefill (de 926.1 a 986.2–986.9 tokens/s).
- **Puntuar solo los bloques QSA vivos:** evita puntuar bloques más allá del rango vivo del prompt actual. En un prompt medido de 15,423 tokens, el tiempo de selección QSA bajó un 91.1%. El rendimiento de prefill de extremo a extremo subió de 961.0 a 1,111.3–1,112.9 tokens/s en las pruebas informadas.
- **Atención de prompt con núcleos tensoriales para Volta:** usa `mma.m8n8k4` en la V100 en lugar del fallback FP32. Un PR anterior midió aproximadamente 1.6–1.9× de velocidad en el kernel de atención y más rendimiento de prompt en sus pruebas. El porte posterior del PR #600 de upstream redujo un 27.9% el tiempo del kernel de atención int8 en una forma medida. Los deltas de prefill de extremo a extremo informados fueron menores (+1.3% a +3.0% en sus pruebas de 4K–32K).

### Decode y ejecución multi-GPU

- **Ventana de verificación por lotes:** ejecuta varias posiciones de decode especulativo juntas y reduce el coste de lanzamiento por token. Las pruebas históricas emparejadas informaron de una generación de salida más rápida, pero los resultados varían según el prompt y las condiciones del runtime.
- **Puntuación de bloques QSA por teselas:** reutiliza los datos de clave seleccionados entre teselas de consulta en lugar de leerlos repetidamente. El kernel medido de 256 consultas / 65K bloques bajó de 3.5 ms a 1.3 ms; su aritmética por puntuación siguió siendo idéntica bit a bit.
- **GR de Volta fusionado y anexado KV por lotes:** combina el trabajo entre posiciones de verificación y anexa datos KV en menos lanzamientos. En la configuración de dos V100 probada, el decode aumentó un 5.6–17.3% en prompts de 4K–32K, mientras que el prefill quedó prácticamente igual.
- **Ajuste asimétrico de dos V100:** mejora las filas de host, el despacho del plan de dispositivos y la propiedad de la memoria con dos tarjetas. Una configuración instalada anterior mostró ganancias de prefill del 13.3–18.7% en 4K–32K. Los resultados de decode fueron mixtos, y ambas GPU se ralentizaron térmicamente durante esa ejecución. Estos valores no aíslan cada cambio de código.
- **Reducir el coste de pesos y despacho:** carga solo las capas que posee cada tarjeta cuando usas un split explícito. Evita la cuantización de activaciones en CPU innecesaria y las barreras de despacho vacías, y opcionalmente ejecuta kernels de expertos de filas emparejadas. La asignación densa nativa medida de GPU0 bajó de 1,376.20 MiB a 541.27 MiB. El modo de filas emparejadas es opcional; los resultados informados del modelo completo no demostraron una mejora de decode.

### Kernels de atención y tráfico de memoria

- **Atención de decode (porte del PR #540 de upstream):** reduce los intercambios de puntuaciones y el tráfico de memoria compartida y conserva el orden de acumulación. El tiempo de kernel medido bajó de 52.90 a 31.13 microsegundos en GPU0 para M=1. Para M=8 bajó de 180.42 a 111.85 microsegundos. Los cambios informados de decode del modelo completo fueron pequeños; el prefill quedó efectivamente igual.
- **Atención de prompt (porte del PR #600 de upstream):** usa la implementación de núcleos tensoriales de Volta con cuatro cálculos MMA independientes. El PR informa de comprobaciones de corrección y mediciones a nivel de kernel y de modelo. La reducción de kernel del 27.9% no se debe leer como un aumento del 27.9% en la velocidad del modelo completo.
- **Recogida KV y recurrencia GDN (portes del PR #627 de upstream):** usa cargas KV int8 alineadas más anchas con un fallback seguro al ancho original. También usa varios acumuladores para la recurrencia de prompt de V100. Los tiempos de kernel medidos mejoraron alrededor de un 10% para una recogida KV de 32K y alrededor de un 7% para una recurrencia de 4,096 tokens. El prefill emparejado del modelo difirió menos de un 0.2%; no se demostró una ganancia fiable de decode de extremo a extremo.

### Soporte de compilación y evidencia

El fork también ha reparado la configuración y el enlazado de CUDA para `sm_70`. Ha mantenido la admisión de dispositivos V100 a través de las fusiones de upstream, y ha adaptado los cambios de upstream cuando sus requisitos de arquitectura excluyen Volta. Estas correcciones de compatibilidad mantienen compilables las rutas optimizadas; no son por sí mismas afirmaciones de rendimiento.

Para ver la evidencia completa y el alcance de la implementación, consulta los [#1](https://github.com/jmnargi/Strata-V100/pull/1), [#2](https://github.com/jmnargi/Strata-V100/pull/2), [#5](https://github.com/jmnargi/Strata-V100/pull/5), [#6](https://github.com/jmnargi/Strata-V100/pull/6), [#11](https://github.com/jmnargi/Strata-V100/pull/11), [#12](https://github.com/jmnargi/Strata-V100/pull/12), [#15](https://github.com/jmnargi/Strata-V100/pull/15), [#16](https://github.com/jmnargi/Strata-V100/pull/16) y [#17](https://github.com/jmnargi/Strata-V100/pull/17). Los resultados usan fechas, compilaciones, prompts, ubicaciones de GPU y protocolos distintos. Son mediciones históricas, no una única comparación controlada de antes y después ni una garantía de ganancias en todos los sistemas V100.

## Otro hardware

- **Tarjetas gráficas más antiguas** (Tesla P40 / V100, GTX 10, Radeon VII / MI50, RX 6700 XT, RX 5500 XT): [GPU más antiguas](docs/OLDER_GPUS.md).
- **Intel Arc**, compilado desde el código fuente en Linux: [Intel Arc](docs/INTEL_ARC.md).
- **AMD Ryzen AI Max (Strix Halo)**, compilado desde el código fuente en Linux: [Strix Halo](docs/STRIX_HALO.md).
- **Procesadores más antiguos sin AVX2**: funcionan, pero despacio. [CPU más antiguas](docs/INSTALL.md#older-cpus-experimental).

## Instalación y arranque

Este fork sigue la experiencia de instalación de upstream. Para ver los requisitos de plataforma completos y actuales, consulta la [guía de instalación](docs/INSTALL.md). Consulta también la [guía de modelos](docs/MODELS.md) y la [guía de setup de upstream](docs/AI_SETUP.md) para ver las opciones de modelo y las opciones disponibles.

- **Windows:** descarga o clona este repositorio y ejecuta `START-HERE.bat`.
- **Linux:** clona este repositorio y ejecuta `./setup.sh`.
- Sigue las indicaciones para seleccionar un modelo y un tamaño de contexto. El setup descarga los datos del modelo y arranca el servicio local.
- Abre la dirección que imprime el setup (normalmente `http://127.0.0.1:8080`).

Para una Tesla V100, usa el motor CUDA 12 con soporte de `sm_70`. El setup selecciona el motor para GPU más antiguas en Volta; en Linux lo compila localmente. El archivo CUDA 12 de Windows de upstream incluye código Volta, pero no se probó en esta máquina Linux. Consulta la [guía para GPU más antiguas](docs/OLDER_GPUS.md) para ver los requisitos de toolkit y controlador. Para Docker, sigue las [instrucciones de Docker](docs/INSTALL.md) y compila para la arquitectura de GPU objetivo.

## Uso de Strata

- **Aplicación web:** usa la URL local que imprime el setup para el chat y el monitor en vivo.
- **API compatible con OpenAI:** define la URL base de tu aplicación como `http://127.0.0.1:8080/v1`.
- **API compatible con Anthropic:** usa `http://127.0.0.1:8080/v1/messages`.
- **API Responses de OpenAI:** usa `http://127.0.0.1:8080/v1/responses`. Este endpoint sin estado admite llamadas a herramientas al estilo Codex; no almacena `previous_response_id`.
- **Imágenes, MCP, multi-GPU y configuración:** consulta los [detalles](docs/DETAILS.md), el [servidor MCP](docs/MCP_SERVER.md) y la [guía multi-GPU](docs/MULTI_GPU.md).

El servidor normalmente escucha en localhost. Si lo expones a otras máquinas, configura una clave de API y usa una red de confianza. No publiques secretos ni configuración privada en issues, resultados de benchmark ni pull requests.

### Caché de conversaciones en disco (opcional)

El estado de las conversaciones tiene tres niveles. L1 es la sesión activa y sus checkpoints de prompt residentes. L2 es la caché opcional en RAM del host (`--conversation-cache-mib`). L3 es un almacén opcional en disco que conserva las conversaciones guardadas y los grupos de checkpoints entre reinicios. Las capas KV transmitidas usan sus pools de host autoritativos cuando el motor captura el estado.

Para usar el almacén en disco, añade `--conversation-cache-disk strata-conversations --conversation-cache-disk-gib 25` a los argumentos del motor. La ruta y un presupuesto positivo en GiB son obligatorios juntos. El almacén está desactivado por defecto, necesita `--serve`, funciona con `--conversation-cache-mib 0` y admite `--layer-split` en varias GPU. Opcional: `--conversation-cache-disk-slots N` limita el número de registros, y `--conversation-cache-disk-min-free-mib N` mantiene espacio libre en el sistema de archivos.

El motor conserva un checkpoint activo antes de que la retención lo elimine o de que una petición sobrescriba su rama. Cada checkpoint guardado incluye el estado GDN, el historial PLE, el estado del indexador QSA y el respaldo KV necesario. Los datos KV de tokens por sí solos no bastan para reanudar este modelo.

Los checkpoints relacionados comparten una imagen KV por grupo de archivo. El motor mueve y amplía esa imagen en lugar de copiar el prefijo completo para cada checkpoint. Copia solo las páginas KV nuevas o modificadas cuando el respaldo sigue siendo válido. Un grupo deja de crecer antes de alcanzar el límite de metadatos del disco. Un grupo posterior tiene su propio respaldo. El motor no combina un checkpoint guardado con bytes KV de otro cálculo del prefijo.

Los grupos de archivo usan los mismos límites de bytes y de huecos que las conversaciones guardadas en L2. Las víctimas por capacidad de L2 pasan a L3 cuando el nivel de disco está activado. Una admisión correcta en L2 no escribe también la imagen en disco. Si una imagen no cabe en L2, el motor intenta guardarla en L3. Los límites de RAM física y de espacio libre en disco siguen aplicándose. El ajuste de tokens mínimos también se aplica a los archivos de checkpoints.

Conservar un checkpoint es una optimización, no un requisito. Si el suelo de RAM del host o el presupuesto de disco rechazan la imagen, el motor descarta el checkpoint, registra el descarte y sigue sirviendo. No detiene el servidor. La sesión sigue siendo correcta; solo se pierde la reutilización de ese checkpoint.

La búsqueda selecciona el prefijo compatible más largo entre la sesión activa, L2, L3 y los huecos de lote inactivos. Los tokens, la identidad de imagen y el modo de vector de control deben coincidir. Para una edición de un mensaje anterior, la reutilización termina en un checkpoint guardado antes del primer token cambiado. El motor debe procesar el sufijo de nuevo; no puede reanudar en un token arbitrario que coincida sin el estado recurrente en esa posición. Aumentar `--prompt-cache` no sustituye la conservación de checkpoints.

Un archivo de L2 sigue disponible después de una restauración. Una imagen válida de L3 se promociona a L2 cuando el presupuesto de bytes y el suelo de RAM física lo permiten. Solo un subconjunto acotado de sus checkpoints pasa a ser residente en la sesión activa. Restaurar una rama no elimina otros checkpoints del archivo guardado.

El almacén en disco admite cadenas de conversación de solo anexado. Una cadena en curso puede anexar nuevos IDs de token, checkpoints, estado en ejecución y páginas KV. Un desalojo de L2 escribe una imagen semilla completa; no necesita una captura de GPU. Un rebobinado inicia una rama separada en lugar de reemplazar la cadena de origen. El formato de archivo no cambia para los archivos de checkpoints.

Los paneles de disco tienen sumas de comprobación y campos de versión. Una semilla usa un archivo temporal y un renombrado. Un anexado incompleto se recupera hasta el último panel completo. Los registros no válidos o incompatibles se eliminan. El almacén en disco desaloja las cadenas menos usadas recientemente para respetar su presupuesto de bytes, el límite de registros y el suelo de espacio libre. Por tanto, ni L2 ni L3 ofrecen historial ilimitado. Si ambos niveles han desalojado el checkpoint necesario, la petición debe procesar de nuevo el prefijo que falta.

El motor captura una vez una raíz de prompt de sistema compartida en L2 cuando la retención de raíz está activada. L3 recibe la raíz en un desalojo de L2 o en una admisión fallida en L2. Con L2 desactivado, un L3 activado puede recibir la raíz directamente. `--conversation-cache-keep-root` y `--conversation-cache-disk-keep-root` controlan la retención de raíz en cada nivel. Sus variantes `--no-` la desactivan. Las raíces tienen menor prioridad de desalojo que las imágenes normales; los anclajes explícitos de checkpoints de petición siguen protegidos.

El nivel de disco no añade ejecución concurrente. Consulta [batch slots](docs/BATCHING.md) para ver los costes de memoria de peticiones concurrentes. La captura debe terminar antes de que una petición sobrescriba los bytes necesarios de la sesión. La restauración es síncrona. Las escrituras por desalojo de L2 usan instantáneas del host y pueden añadir latencia de E/S de disco. Dimensiona ambos presupuestos de caché para las conversaciones y el historial de checkpoints que necesites. Consulta la [guía de V100](docs/NVIDIA_V100.md) para ver la verificación de retención de checkpoints y el [informe del benchmark de disco](benchmarks/v100-l3-conversation-cache-2026-10-03.md) para ver mediciones anteriores.

#### Resultado medido en V100

En la configuración de dos V100 con split por capas de este fork, una conversación de 33,725 tokens se reanudó desde una caché NVMe de 25 GiB. La lectura tardó 1,614.6 ms y la restauración 315.8 ms. La fase completa de prompt reanudado, incluidos 22 tokens nuevos, tardó 3,015.1 ms. Una lectura en frío del mismo prefijo de 33,725 tokens tardó 20,332.7 ms. Esto es una **aceleración de 6.74x** y una **reducción del 85.2% en la latencia del prompt** para la petición reanudada. El registro de 943.6 MiB restauró un estado del modelo principal exacto byte a byte y una salida idéntica en ambas GPU. La recuperación tras reinicio pasó la paridad exacta byte a byte, y el fallback ante registros corruptos produjo una salida coincidente. Consulta el [informe del benchmark](benchmarks/v100-l3-conversation-cache-2026-10-03.md).

Medido en el equipo de dos V100 con split por capas de este fork (Qwen3.8-Flash-Next Q2_0, KV int8): dos
conversaciones alternando por una caché de trabajo sirvieron correctamente en 8 peticiones con
`corruptions=0 evictions=0`; un archivo de cadena por conversación creció de 238 MiB -> 594 MiB a lo largo de
tres guardados de +12 tokens nuevos cada uno (un almacén de instantánea completa habría escrito un archivo nuevo de ~2.4 GiB
por guardado con 100k+ tokens). El primer guardado de un chat corto nuevo siembra la cadena
(~118 MiB: una página K/V por capa más el estado en ejecución fijo del modelo); después, cada guardado
es el delta (+12 tokens ~ 118 MiB) y la ruta de lectura y restauración no cambia (~0.4 s para leer,
~0.04-0.08 s para restaurar).

### Visión diferida (opcional)

El codificador de imágenes (`strata-vision`) ocupa aproximadamente 1.74 GiB de VRAM (1.43 GiB en la primera GPU, 0.31 GiB en la segunda) mientras se ejecuta. Con la visión diferida no se ejecuta entre peticiones de imagen. Esa VRAM se queda con las cachés de expertos, así que el trabajo solo de texto conserva el número completo de expertos residentes.

Para activarla, añade a la configuración del modelo:

- `"vram_elastic": true` (el flag del motor `--vram-elastic`): las cachés de expertos están segmentadas, así que el motor puede devolver VRAM y volver a tomarla entre peticiones.
- `"vision": {"lazy": true, "idle_s": 900}`: el codificador arranca con la primera petición de imagen y se detiene tras `idle_s` segundos sin ninguna (900 = 15 minutos). El codificador necesita el modo de una petición a la vez del motor. Por eso no lo combines con `"parallel"` (o da al codificador su VRAM de otra forma, por ejemplo `POST /v1/vram`).

Cómo se comporta:

- Una petición de texto nunca arranca el codificador. Una petición de imagen primero indica al motor que devuelva la VRAM de las GPU. Hay una reserva por GPU, `[1700, 600]` con split por capas y `[1800]` con una GPU, y se puede sobrescribir con `"vram_mib"`. Después arranca el codificador y codifica la imagen mientras el motor está inactivo.
- Tras `idle_s` segundos sin petición de imagen, el codificador se detiene y las cachés de expertos vuelven a crecer hasta su tamaño completo. Las imágenes codificadas siguen en caché en disco, así que enviar la misma imagen otra vez no la vuelve a codificar.
- Medido en el split por capas de dos V100 de este fork. Con el codificador inactivo, las cachés mantienen **14,005** expertos residentes (12,688 con el codificador residente). Una petición de imagen en frío se respondió en **5.9 s**. Ambas cachés volvieron a 14,005 después de descargar el codificador. Consulta los [detalles](docs/DETAILS.md).

## Si algo va mal

Los problemas habituales y sus soluciones están en [TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md). Esa página cubre
la aplicación de escritorio, el comando `strata` y el instalador. La tabla completa de mensajes de error está en
[DETAILS.md](docs/DETAILS.md#troubleshooting).

Si el servidor arranca pero una petición falla, ejecuta `strata doctor`. Lee los componentes instalados e
imprime lo que falta o está desactualizado; consulta [TERMINAL.md](docs/TERMINAL.md#doctor).

Si una respuesta es lenta en lugar de estar rota, lee [TUNING.md](docs/TUNING.md). Da el ajuste que
importa para cada objetivo y el coste medido de cada uno.

## Contribuir

Las contribuciones son bienvenidas. No necesitas una V100 para ayudar. La documentación, las pruebas, el setup, el comportamiento del servidor, la compatibilidad de API y las mejoras para otros dispositivos compatibles son útiles. El código y los resultados de rendimiento específicos de V100 se benefician de la validación en hardware Volta real.

1. Consulta los [issues abiertos](https://github.com/jmnargi/Strata-V100/issues) y los [pull requests](https://github.com/jmnargi/Strata-V100/pulls) para evitar trabajo duplicado.
2. Para funciones amplias de upstream, comprueba si el cambio corresponde al [repositorio upstream de Niko](https://github.com/Niko1221/Strata). Este fork integra cambios de upstream periódicamente; las correcciones y mediciones centradas en V100 se pueden proponer aquí.
3. Haz un cambio centrado, describe el hardware y los pasos exactos usados para probarlo, e incluye las pruebas relevantes.
4. Para afirmaciones de rendimiento, informa de la línea base y el candidato, la configuración completa del runtime, la carga de trabajo, el número de repeticiones y los límites. Incluye datos en bruto o un comando reproducible cuando sea posible. Separa el tiempo de kernel del rendimiento del modelo completo y no afirmes una ganancia que las mediciones no muestran.
5. Abre un pull request contra la rama `main` de este repositorio. Explica si el cambio es específico de V100, una integración de upstream o una mejora general de Strata. Mantén las credenciales, la configuración personal y los archivos locales no relacionados fuera del cambio.

Consulta [`benchmarks/README.md`](benchmarks/README.md) y [`docs/DETAILS.md`](docs/DETAILS.md) para ver el enfoque de medición actual. Las contribuciones siguen sujetas a la licencia del proyecto y a las licencias de los componentes y archivos de modelo incluidos.

## Historia del proyecto

Este fork desarrolla el soporte de V100 y el trabajo específico de V100 y mantiene el desarrollo upstream de Strata. Los pull requests recientes de integración y rendimiento ilustran ese proceso:

- [#14 — Fusionar upstream v0.1.36 conservando el soporte de V100](https://github.com/jmnargi/Strata-V100/pull/14)
- [#15 — Portar los cambios medidos de recogida KV y recurrencia GDN para V100 del PR #627 de upstream](https://github.com/jmnargi/Strata-V100/pull/15)
- [#16 — Reducir los intercambios de atención de decode y el tráfico de memoria compartida en V100](https://github.com/jmnargi/Strata-V100/pull/16)
- [#17 — Portar el kernel de atención de prompt para Volta del PR #600 de upstream](https://github.com/jmnargi/Strata-V100/pull/17)
- [#88 — Expertos trellis por pares de Gyro-L (TQP5/6/7), corrección de ventana de decode plegada y de ayudante de prompt, y perfil de V100 en `layer_split 14`](https://github.com/jmnargi/Strata-V100/pull/88)

Estos informes describen cambios probados concretos; no son una afirmación general de rendimiento más rápido en todos los modelos o GPU. Para más detalle, explora el [historial completo de pull requests](https://github.com/jmnargi/Strata-V100/pulls?q=is%3Apr+is%3Amerged) y el [historial de commits](https://github.com/jmnargi/Strata-V100/commits/main).

## Créditos y licencia

Este repositorio es un fork de [Strata, de Niko1221](https://github.com/Niko1221/Strata). **Estamos profundamente agradecidos a Niko y a todos los colaboradores de upstream.** Su trabajo original es la razón por la que este fork, su soporte de V100 y este runtime local de modelos son posibles. Queremos dar crédito y seguir el trabajo de upstream a medida que integramos sus cambios periódicamente.

Strata incorpora [llama.cpp / ggml](https://github.com/ggml-org/llama.cpp) y otros componentes de código abierto. El proyecto está bajo la [licencia MIT](LICENSE); los componentes y los archivos de modelo pueden tener términos distintos. Consulta las [notas de créditos y licencia](docs/HOW_IT_WORKS.md#credits).
