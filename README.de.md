# Strata-V100

**Ein von der Community gepflegter Fork von [Strata](https://github.com/Niko1221/Strata) mit Schwerpunkt auf der V100.** Dieser Fork hält die Unterstützung für NVIDIA Volta (`sm_70`) funktionsfähig, entwickelt und misst Leistungsänderungen auf Tesla-V100-Hardware und führt regelmäßig Upstream-Verbesserungen für Funktionen und Fehlerbehebungen außerhalb der V100 zusammen.

Strata ist eine lokale Open-Source-Laufzeitumgebung für [Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next). Sie kombiniert GPU, Systemspeicher und Datenträger, um dieses große Modell auf einem Privatcomputer auszuführen. Dieser Fork behält das breitere Strata-Projekt bei: andere unterstützte NVIDIA- und AMD-Hardware, die Weboberfläche, die OpenAI-kompatible API, Werkzeuge, Bildeingabe und Upstream-Funktionen bleiben ebenfalls wichtig.

> **Großer Dank an Niko ([@Niko1221](https://github.com/Niko1221)) und die ursprünglichen Strata-Mitwirkenden.** Dieses Projekt baut auf ihrer Arbeit auf: die Modell-Laufzeit, die Architektur und die Funktionen von Strata und die Arbeit, die den lokalen Betrieb dieses Modells möglich gemacht hat, stammen alle aus dem Upstream-Projekt. Ohne diese Grundlage würde der V100-Fork nicht existieren. Danke, dass ihr das möglich gemacht habt, und danke für die weitere Entwicklung von Strata.

[Upstream Strata](https://github.com/Niko1221/Strata) · [Issues](https://github.com/jmnargi/Strata-V100/issues) · [Pull-Requests](https://github.com/jmnargi/Strata-V100/pulls) · [Diskussionen](https://github.com/jmnargi/Strata-V100/discussions)

**Sprachen:** [English](README.md) · [Deutsch](README.de.md) · [Español](README.es.md) · [Français](README.fr.md) · [日本語](README.ja.md) · [Português (Brasil)](README.pt-BR.md) · [简体中文](README.zh-CN.md)

## Inhalt

- [Modelle und Quantisierungen, die hier laufen](#modelle-und-quantisierungen-die-hier-laufen)
- [Worauf dieser Fork den Schwerpunkt legt](#worauf-dieser-fork-den-schwerpunkt-legt)
- [Dokumentation](#dokumentation)
- [Versionshinweise](#versionshinweise)
- [V100-Benchmark](#v100-benchmark)
- [Leistungsarbeit für die V100](#leistungsarbeit-für-die-v100)
- [Weitere Hardware](#weitere-hardware)
- [Installation und Start](#installation-und-start)
- [Strata verwenden](#strata-verwenden)
- [Wenn etwas schiefgeht](#wenn-etwas-schiefgeht)
- [Mitwirken](#mitwirken)
- [Projektgeschichte](#projektgeschichte)
- [Danksagung und Lizenz](#danksagung-und-lizenz)

## Modelle und Quantisierungen, die hier laufen

Strata kann **Qwen3.8-Flash-Next** aus sieben Familien quantisierter Veröffentlichungen ausführen, dazu ein kleineres **Qwen3.6-35B-A3B**. Jede
Zeile unten wird von der Engine und vom Setup dieses Forks unterstützt; die letzte Spalte nennt, was auf dem V100-Paar hier tatsächlich
durchgängig ausgeführt wurde. Größen, Qualität und Geschwindigkeiten je Größe: [docs/MODELS.md](docs/MODELS.md).

| Modell / Quantisierung | Geroutete Experten | Herkunft der Gewichte | Download | Lauf auf diesem V100-Paar |
| --- | --- | --- | ---: | --- |
| **NVFP4 — `local-inference-lab` (QAD)** | NVFP4, Gruppe 16; MXFP8 für Attention und geteilte Experten; NVFP4-n-Gramm-Tabelle | [`local-inference-lab/Qwen3.8-Flash-Next-NVFP4`](https://huggingface.co/local-inference-lab/Qwen3.8-Flash-Next-NVFP4), auf deinem PC konvertiert ([NVFP4.md](docs/NVFP4.md#the-qad-distillation-local-inference-labqwen38-flash-next-nvfp4)) | 106 GB | ja — 62.3 Tokens/s beim Decode |
| **NVFP4 — NVIDIA, für FP4 trainiert (`FP4-Local`, QAD)** | NVFP4, Gruppe 16 (Skalen in NVIDIAs CSF-Form) | [`nvidia/Qwen3.8-Flash-Next-FP4-Local`](https://huggingface.co/nvidia/Qwen3.8-Flash-Next-FP4-Local), auf deinem PC konvertiert ([NVFP4.md](docs/NVFP4.md#nvidias-fp4-local-checkpoint)) | 102 GB (+110 GB beim Vorbereiten) | unterstützt; hier nicht ausgeführt |
| **NVFP4 — NVIDIAs Version** | NVFP4, Gruppe 16 | [`Maximilian228/Qwen3.8-Flash-Next-NVIDIA-NVFP4-Strata`](https://huggingface.co/Maximilian228/Qwen3.8-Flash-Next-NVIDIA-NVFP4-Strata) (fertiges Pack) | 128 GB | unterstützt; hier nicht ausgeführt |
| **NVFP4 — Stratas Version** | NVFP4, Gruppe 16 | [`Maximilian228/Qwen3.8-Flash-Next-NVFP4-GPTQ-Strata`](https://huggingface.co/Maximilian228/Qwen3.8-Flash-Next-NVFP4-GPTQ-Strata) (fertiges Pack, GPTQ) | 128 GB | unterstützt; hier nicht ausgeführt |
| **Unsloth Dynamic v3.1 (1-4 Bit)** | UD-IQ1_M, UD-Q2_0, UD-IQ2_XS, UD-Q2_K_XL, UD-IQ3_XXS, UD-Q3_K_XL, UD-IQ4_XS-v31 | [`unsloth/Qwen3.8-Flash-Next-GGUF`](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF), Revision vom 2026-11-04 | 68-94 GB | ja — UD-IQ4_XS-v31, 81 Tokens/s |
| **Unsloth Dynamic (frühere Revision)** | UD-IQ4_XS, UD-Q4_K_XL, UD-Q5_K_XL, UD-Q6_K_XL, Q8_0 | [`unsloth/Qwen3.8-Flash-Next-GGUF`](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF) | 94-188 GB | unterstützt; hier nicht ausgeführt |
| **ISTA-DASLab i-Quants und Q2_0 (2-3.5 Bit)** | Q2_0, IQ2_XS, IQ3_XXS, IQ3_S | [`ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF`](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF) | 66-84 GB | ja — IQ3_S, das frühere Tagesprofil |
| **Gyro (agentionai), Trellis** | Gyro-S TQ1_0, Gyro-M TQ2_0 (Pair-Trellis) | [`agentionai/Qwen3.8-Flash-Next-Gyro-GGUF`](https://huggingface.co/agentionai/Qwen3.8-Flash-Next-Gyro-GGUF) ([GYRO.md](docs/GYRO.md)) | 58.5 / 92 GB | ja — Gyro-M, etwa 70 Tokens/s |
| **Coder (ISTA-DASLab), die Hälfte der Experten** | IQ1_M, IQ2_S-IQ4_XS wie IQ3_S | [`ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-Coder-GGUF`](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-Coder-GGUF) | 58 GB | unterstützt; hier nicht ausgeführt |
| **OrcaRouter Uncensored** | IQ3_XXS | `OrcaRouter`s Flash-Next Uncensored ([ORCA.md](docs/ORCA.md); ein manueller Packschritt, keine Option des Installers) | 84 GB | unterstützt; hier nicht ausgeführt |
| **Qwen3.6-35B-A3B** (ein kleineres Modell) | UD-Q4_K_XL, UD-Q6_K_XL und Q8_0 von Unsloth, mit der MTP-Draft-Schicht | [`unsloth/Qwen3.6-35B-A3B-MTP-GGUF`](https://huggingface.co/unsloth/Qwen3.6-35B-A3B-MTP-GGUF) ([MODELS.md](docs/MODELS.md#qwen36-35b-a3b)) | 23-38 GB | unterstützt; hier nicht ausgeführt |

Die NVFP4-Zeilen benötigen NVIDIA-Karten. RTX-50-Karten rechnen mit FP4-Breite, ältere Karten einschließlich der V100 mit 8-Bit-Präzision.
Die beiden QAD-Zeilen werden auf deinem PC aus den safetensors der Autoren erstellt; keine quantisierten Gewichte der Autoren werden hier
weitergegeben. Die Gyro- und Trellis-Zeilen benötigen Engine 0.1.45.1 oder neuer, die Unsloth-Zeilen mit Q5_0-downs 0.1.45.7.

## Worauf dieser Fork den Schwerpunkt legt

- **Tesla V100 / Volta-Unterstützung (`sm_70`)**: normale V100-Builds und Laufzeitunterstützung erhalten. V100-Builds verwenden ein CUDA-12.x-Toolkit; CUDA 13 kompiliert keinen `sm_70`-Code.
- **Testmaschine:** zwei Tesla V100-PCIE-16GB; GPU 0 nutzt PCIe Gen3 x4, GPU 1 nutzt Gen3 x16. Die Pipeline-Messungen unten verwenden beide Karten. Der historische Benchmark mit einer Karte verwendet nur die x16-Karte. Ein System mit zwei vollbreiten Gen3-x16-Verbindungen wurde hier nicht gemessen.
- **Upstream-Integration:** Dies ist kein separater Ersatz für Strata. Upstream-Änderungen werden regelmäßig zusammengeführt; die V100-Kompatibilität wird geprüft und bei Bedarf korrigiert. Neue Upstream-Funktionen können Hardware-Anforderungen über `sm_70` haben; sie beschleunigen die V100 nicht automatisch.
- **Die breitere Strata-Community:** Verbesserungen, Fehlerberichte, Tests, Dokumentation und Beiträge für andere unterstützte Plattformen sind willkommen.

Die Unterstützung für die V100 bedeutet nicht, dass jede Upstream-Funktion oder jede andere GPU-Konfiguration auf der V100 getestet wurde. Prüfe den relevanten Code, die Versionshinweise und die Messberichte, bevor du dich auf eine hardwarespezifische Funktion verlässt.

## Dokumentation

Der [Dokumentationsindex](docs/README.md) listet jede Seite und ihren Zweck. Diese Seiten sind die
Referenz:

- [Engine-Flags](docs/ENGINE_FLAGS.md): jede Befehlszeilenoption, ihr Wert und ihr Standard.
- [Umgebungsvariablen](docs/ENVIRONMENT.md): jede Variable, die die Engine, der Server und das
  Terminal lesen.
- [Server](docs/SERVER.md): Serveroptionen, die Schlüssel der Laufzeitkonfiguration und alle API-Endpunkte.
- [Terminal](docs/TERMINAL.md): jeder `strata`-Befehl und jedes Unterkommando.
- [Tuning](docs/TUNING.md): die Einstellungen nach Ziel, die Optimierungen und die gemessenen Ergebnisse.
- [So funktioniert es](docs/HOW_IT_WORKS.md) und das [Glossar](docs/GLOSSARY.md): die Teile der
  Engine und die Begriffe, die diese Dokumentation verwendet.
- [Details](docs/DETAILS.md): die Messaufzeichnung, die Anfrageformate und die Historie der Einstellungen.

Die Seiten folgen dem [Dokumentationsstil](docs/STYLE.md), also ASD-STE100 (Simplified Technical
English).

## Versionshinweise

### Engine 0.1.45.8 — 16. November 2026

[Release v0.1.45.8](https://github.com/jmnargi/Strata-V100/releases/tag/v0.1.45.8)
gibt der aktuellen V100-Engine eine eigene Version. Enthalten sind die Korrektur des
Zeilenversatzes beim FP16-Experten-Prefill, NVFP4-Decode-Arbeiten, optionale
HC_Q8-Speichereinsparungen auf mehreren Karten, optionale Verbesserungen der
Prompt-Bewertung und des gestreamten KV-Caches sowie Prompt-Fortschritt pro Slot.
Diese Versionsanhebung ändert weder Inferenzcode noch Standardwerte.
Messungen und Grenzen stehen in [NVIDIA_V100.md](docs/NVIDIA_V100.md).

### Unsloth Dynamic v3.1-Sets (1-4 Bit) — 8. November 2026

Löst [#95](https://github.com/jmnargi/Strata-V100/issues/95): Das Menü `--family unsloth` im Setup endete bisher bei
UD-IQ4_XS (etwa 4 Bit). Jetzt bietet es auch die Dateien von Unsloths Dynamic v3.1 an, von ~1 bis ~4 Bit, mit der Revision,
die Unsloth am 2026-11-04 veröffentlicht hat (Commit `d9620119`):

| Größe | Download | Geroutete Experten | RAM mit allen Experten im RAM | Hinweise |
| --- | ---: | ---: | ---: | --- |
| UD-IQ1_M | 68.0 GB | 32.4 GiB | ~48 GB | die kleinste; am schwächsten bei sorgfältiger oder langer Arbeit |
| UD-Q2_0 | 70.5 GB | 34.8 GiB | ~51 GB | Q2_0-Experten, wie beim schnellsten 2-Bit-Pack |
| UD-IQ2_XS | 75.8 GB | 39.4 GiB | ~55 GB | 1 Schicht mit Q5_0-downs |
| UD-Q2_K_XL | 79.6 GB | 42.8 GiB | ~59 GB | 1 Schicht mit Q5_0-downs |
| UD-IQ3_XXS | 83.6 GB | 46.8 GiB | ~63 GB | kein Q5_0 |
| UD-Q3_K_XL | 89.5 GB | 52.3 GiB | ~68 GB | 20 Schichten mit Q5_0-downs |
| UD-IQ4_XS-v31 | 93.7 GB | 56.2 GiB | ~72 GB | 33 Schichten mit Q5_0-downs; ein eigener Katalogname, weil die Hub-Dateien UD-IQ4_XS heißen |

Die fünf bereits angebotenen Dateien (UD-IQ4_XS, UD-Q4_K_XL, UD-Q5_K_XL, UD-Q6_K_XL, Q8_0) bleiben an den
vorherigen Commit gebunden. Das zweite Layout von UD-Q4_K_XL (vier Shards, n-Gramm-Tabelle zuletzt) wird am neuen Commit aus
Dateien akzeptiert, die bereits auf dem Datenträger liegen. UD-IQ1_S bleibt abgelehnt: Es mischt zwei Expertenformate in einer Schicht.

Dafür war Engine-Arbeit nötig: das Skalarprodukt der Q5_0-Experten. Sein Offset von -16 pro Gewicht wurde auf die Summe der ursprünglichen
Aktivierungen angewendet (die CUDA-Form von llama.cpp). Deshalb war eine Schicht mit Q5_0 als Down-Projektion pro Experte 2-3% ungenauer
als jedes andere Down-Format. Jetzt multipliziert der Kernel die quantisierte Aktivierungssumme, wie die Q5_1- und Q4_0-Pfade und ggml-cpu
(CUDA und SYCL). Auf dieser Maschine gemessen mit der synthetischen Experten-Parität (`native_expert_parity --synthetic`, gate-up mit IQ2_XXS /
IQ2_S / IQ2_XS / IQ3_XXS / IQ4_XS und Q5_0-downs, relativer L1-Fehler gegen eine Float-Referenz): **2.40e-2 ..
3.18e-2 vorher, 1.25e-2 .. 1.30e-2 nachher**, gleich wie bei IQ4_NL- und Q2_0-downs (1.29e-2 .. 1.31e-2) auf denselben Seeds.
Die Starts der gruppierten Experten bleiben bitgleich zu den Starts vor dem Fenster (`native_grouped_parity`: Q5_0 zu jedem
gate/up-Format hinzugefügt, und die Flash-Next-Formen, die diese Dateien verwenden, 0 Fehler). Engine-Version: 0.1.45.7; das Setup lehnt
für diese sieben Sets eine ältere Engine ab, bevor es etwas herunterlädt.

Die fünf gebundenen Dateien, ihre Bytes und jedes andere Format sind unverändert: Nur die Gewichte der Q5_0-*Experten* lesen diesen Kernel
(der dichte Q5_0-Pfad ist `native_mmvq` und unverändert). Ihr Fehler über den ganzen Experten gegen die Float-Referenz sinkt von
2.4-3.2e-2 auf 1.25-1.30e-2.

### Ausgewählte Performance-Ports aus Upstream v0.2.11 — 5. November 2026

Dieses Update portiert ausgewählte Upstream-Optimierungen; es ist kein vollständiges
Zusammenführen von v0.2.11. Es fügt schnellere Trellis-Experten-Kernel, kleinere Prompt-Scratch-Puffer,
weniger Zuweisungen im PLE-Datenträgerleser und eine Linux-Korrektur für CPU/CUDA ohne Spin
hinzu. Die Engine-Version bleibt 0.1.45.6.

**Auf der Dual-V100-Maschine dieses Forks gemessen:** zwei V100-PCIE-16GB bei
175 W (GPU0 Gen3 x4, GPU1 x16), Ryzen 5 3600 und Gyro-M abliterated Huihui
TQ2_0. Das Profil verwendete Split 16, zwei Pipeline-Fenster, MTP 4, int8-KV
und 524,288-Token-Kontext mit YaRN 2.

| Prompt-Tokens | Prompt-Zeit der Kontrolle | Gruppen-Scratch + WMMA aus |
| ---: | ---: | ---: |
| 2,048 | 2,360.8 ms | 2,214.3 ms |
| 8,192 | 6,932.7 ms | 6,541.0 ms |
| 32,768 | 18,008.6 ms | 17,636.7 ms |

Das sind Mittelwerte aus kalten, verschachtelten A-B-A-B-Läufen, vier Anfragen je Größe
und Variante. Beide Varianten enthielten bereits die Trellis-, HC- und PLE-Ports. Das ist
kein gemessener Gewinn bei jeder Kontextgröße und keine Garantie für die Geschwindigkeit ganzer Anfragen.

Die gemessene optionale Kombination ist `STRATA_TQ_SUB8=1 STRATA_EXPERT_PAIR=0
STRATA_PF_HC_REGION=1 STRATA_PF_GROUP_SCRATCH=1 STRATA_PF_WMMA=0`.
Gruppen-Scratch kann die Gleitkomma-Rundung ändern. Die Änderung an der PLE-Zuweisung
braucht kein Flag. Die Korrektur ohne Spin gilt bei `STRATA_POOL_SPIN_US=0`; sie
belegt keine Einsparung bei der täglichen Einstellung von 100.

Siehe den [V100-Port-Bericht](docs/NVIDIA_V100.md#selected-upstream-0211-ports-gyro-m-2026-11-05)
für einzelne Messungen, Prüfungen der Optionen, numerische Prüfungen, Build-Grenzen
und getestete Kandidaten, die nicht übernommen wurden. Keine Dienst-Standardwerte werden geändert.

### Ausgewählter Upstream-Sync — 19. Oktober 2026

Dieser Branch importiert gemeinsame Änderungen an der CUDA-Engine und am Server aus dem Upstream-Commit
`3d1fb480`. Er behält die Volta-Kernel des Forks, das Pipeline-K/V-Wachstum
und die Änderungen am Gesprächs-Cache. Er schließt unabhängige Desktop-Pakete und
getrennte Aktualisierungen der Backends Metal, AMD und Intel aus. Siehe den
[Sync-Umfang](improvements/RELEASE-2026-10-19-V100.md).
Verwende CUDA 12.8 mit `sm_70`. Eine importierte Option belegt keinen
Leistungsgewinn auf der V100.

### Integration von Upstream v0.1.39

Dieses Update integriert [Strata v0.1.39](https://github.com/Niko1221/Strata/releases/tag/v0.1.39), einschließlich der dazwischenliegenden Releases. Es behält die V100-Tensor-Core-Experten des Forks, die Attention-Änderungen, den asymmetrischen GPU-Split und den Gesprächs-Cache auf dem Datenträger.

- **Parallele Anfragen:** Setze `"parallel": 2` in der Modellkonfiguration oder verwende `./setup.sh --parallel 2`. Das Setup schreibt diesen Standardwert für eine neue Volta-Konfiguration und bewahrt eine vorhandene Einstellung. Die Engine kann getrennte Unterhaltungen gemeinsam decodieren. Zusätzliche Anfragen warten auf einen freien Slot. Slots belegen zusätzlichen GPU-Speicher und können die Geschwindigkeit einzelner Anfragen senken. Wenn zwei Slot-Sitzungen nicht passen, sagt die Engine das und bedient jeweils eine Anfrage; die Konfiguration bleibt unverändert. Siehe [Batch-Slots](docs/BATCHING.md) für Genauigkeitseinstellungen, Speicherkosten und Grenzen.
- **APIs und Oberfläche:** Das Update fügt die OpenAI Responses API, Modelleinstellungen, die Überwachung des Gesprächs-Caches und Metriken je Slot hinzu. Siehe [API-Details](docs/DETAILS.md#using-it).
- **Leistung und Speicher:** Das Update enthält weniger Starts im Verify-Durchlauf, Prompt-Ringe in Bytegröße, Linux-Huge-Pages, Korrekturen am RAM-Budget und Optionen für mehrere GPUs. Ein expliziter `--layer-split` lädt jetzt automatisch nur die dichten Gewichte der eigenen Schichten jeder Stufe (das Upstream-Flag `--trim-stage-weights` und `STRATA_STAGE_TRIM` sind entfallen). Der frei werdende VRAM geht an den Experten-Cache. Eine Optimierung für neuere GPUs bedeutet keinen Geschwindigkeitsgewinn auf der V100. Optionale Pfade, die die Rundung ändern, müssen vor der Verwendung gemessen werden.
- **Andere Plattformen:** Die Upstream-Pfade für Intel Arc, ältere CPUs und weitere AMD-Pfade bleiben verfügbar. Diese Maschine bietet keine Hardware-Validierung für diese Plattformen.

Die V100 benötigt CUDA 12.x und einen Build, der `sm_70` enthält. Installiere die CUDA-13-Engine nicht auf einer V100. Für vorhandene Installationen wird eine explizite parallele Konfiguration empfohlen; ein Update überschreibt die Modellkonfiguration nicht.

**Auf den beiden V100-Karten gemessen (4. Oktober 2026, endgültiger Build).** Ein kontrollierter
Vorher-Nachher-Lauf des integrierten Builds gegen die zuvor installierte Laufzeit, eine Anfrage nach der anderen
(keine Batch-Slots), kalt, Qwen3.8-Flash-Next Q2_0 mit int8-KV beim 524,288-Token-Kontext,
der 20/28-Layer-Split und MTP (`--spec 8`) unverändert; zwei frische Anfragen je Größe, Mediane:

| Ziel-Prompt | Vorher (Tok/s) | Integriert (Tok/s) | Änderung |
| ---: | ---: | ---: | ---: |
| 2K | 701.41 | 718.5 | +2.44 % |
| 8K | 1,402.82 | 1,458.0 | +3.93 % |
| 32K | 2,262.4 | 2,283.7 | +0.94 % |

Jeder Wert ist der Median aus zwei frischen Anfragen (`reused=0`). Die Gewinne sind klein; zwei Wiederholungen
je Größe sind keine statistische Aussage. Alle sechs gepaarten ersten Tokens stimmten mit denen der Basislinie überein.
Für die vollständigen Antworten wird keine Identität behauptet: Die adaptive Expertenstufe hält beim integrierten Build eine andere
Mischung von Experten auf der GPU und der CPU, und die beiden Pfade runden
unterschiedlich. Es wird keine Änderung bei Decode oder Qualität behauptet. Das sind die gemessenen
Werte des endgültigen Builds (Engine-SHA-256 `e9ad337a…`), berichtet im
[vollständigen Bericht](benchmarks/v100-q2_0-upstream-v0139-2026-10-04.md).

Parallele Slots beim 524,288-Token-Kontext hängen vom Split und vom Speicher ab, der auf der
letzten Stufe übrig bleibt. Der gemessene Rückfall (2 angefordert, jeweils eine Anfrage bedient, Konfiguration
unverändert) ist in [docs/BATCHING.md](docs/BATCHING.md) beschrieben.

### Pipeline-Performance mit zwei V100 — 15. Oktober 2026

Das gemessene Dienstprofil aktiviert `--pipeline-windows 2` mit der GPU-Reihenfolge
`0,1` und Layer-Split `16`. Es verwendet zwei 16-GiB-V100-Karten bei 175 W,
einen Ryzen 5 3600, das native IQ3_S-Modell mit gemischten Formaten, int8-KV und MTP
mit `--spec 8`. Die konfigurierte Kontextkapazität bleibt 524,288 Tokens.
Der größte gemessene Prompt hat 32,768 Tokens.

**Sicherheit bei langen Antworten:** Verwende einen Build mit der
[Pipeline-KV-Wachstumskorrektur](benchmarks/v100-pipeline-kv-growth-fix-2026-10-16.md),
wenn du dieses Profil mit `STRATA_KV_GROW=1` kombinierst. Die frühere Pipeline-Schleife
konnte während einer langen Antwort auf nicht abgebildete KV-Zellen zugreifen. Die Korrektur
entleert spekulative Arbeit, bevor sie die Abbildung vergrößert, und setzt dann die Pipeline
fort. Die Geschwindigkeitsmessungen unten für kurze Ausgaben haben
diese Grenze nicht geprüft.

#### Prefill neuer Prompts bis 32K

Jeder Wert ist der Median aus drei Anfragen mit exakten Tokens ohne Prompt-Wiederverwendung
und einer Obergrenze von 16 Ausgabe-Tokens. Die Prefill-Zeit schließt die Antworterzeugung aus.
Der Durchsatz ist die Zahl der Prompt-Tokens geteilt durch die mediane Prefill-Zeit.

| Prompt-Tokens | Frühere Prefill-Zeit | Prefill-Zeit mit Pipeline | Frühere Tokens/s | Tokens/s mit Pipeline | Änderung des Durchsatzes |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 2,048 | 2.089 s | 2.090 s | 980.6 | 979.8 | −0.09% |
| 8,192 | 5.487 s | 5.494 s | 1,493.0 | 1,491.2 | −0.13% |
| 16,384 | 8.828 s | 8.857 s | 1,855.9 | 1,849.8 | −0.33% |
| 32,768 | 15.668 s | 15.677 s | 2,091.4 | 2,090.2 | −0.06% |

Der Prefill neuer Prompts ist annähernd unverändert. Diese kleinen Unterschiede
belegen keine bedeutende Regression.

#### Decode in einer Unterhaltung

Der separate HTTP-Vergleich verwendet einen Anfangsprompt mit 8,828 Tokens und sechs
Folgerunden. Der Kontext erreicht etwa 11.9K Tokens. Jeder Wert
ist der Median aus vier Anfragen, wobei die Reihenfolge der Varianten in abwechselnden Runden umgekehrt wird,
mit einer Obergrenze von 256 Ausgabe-Tokens. Neue Prompt-Tokens sind die in dieser Runde verarbeiteten Tokens, nicht die gesamte Kontextgröße.

| Runde | Neue Prompt-Tokens | Frühere Decode-Tokens/s | Decode-Tokens/s mit Pipeline | Änderung beim Decode |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 121 | 70.20 | 79.45 | +13.18% |
| 2 | 321 | 70.60 | 82.05 | +16.22% |
| 3 | 1,086 | 76.60 | 79.00 | +3.13% |
| 4 | 128 | 73.00 | 79.90 | +9.45% |
| 5 | 313 | 74.35 | 86.55 | +16.41% |
| 6 | 1,089 | 74.15 | 80.75 | +8.90% |

Der Median der sechs Decode-Durchsatzverhältnisse ist **+11.31%**.
Das ist kein gemessener Decode-Gewinn bei jeder Kontextgröße bis 32K.
Erzeugte Längen und MTP-Arbeit unterscheiden sich durch automatische CPU-Aufteilung und
residentes Routing. Die Latenz ganzer Anfragen ist gemischt; ein HTTP-Median einer kurzen Runde
verschlechtert sich. Interpretiere den Decode-Gewinn nicht als garantierte
Beschleunigung ganzer Anfragen und nicht als Vergleich bei gleicher Ausgabe.

Eine separate Kontrolle mit festen Einstellungen stimmt bei Prompt-Größen von 2K, 8K und 32K mit allen drei vollständigen
Ausgabesequenzen von 16 Tokens überein. Sie beweist nicht, dass die Ausgabe bei jeder Produktionsanfrage identisch ist.

Siehe den [Bewertungsbericht](benchmarks/v100-upstream-pr-evaluation-2026-10-15.md)
für genaue Zeitmessungen, die Herkunft des Builds, verworfene Versuche und Befehle zur
Reproduktion. Siehe den [Leitfaden für das Dienstprofil](docs/NVIDIA_V100.md#dual-pcie-pipeline-service-profile),
bevor du die Option auf einer anderen Installation aktivierst.

### TurboQuant-KV-Cache — 25. Oktober 2026

TurboQuant ist ein optionales KV-Cache-Format. Es rotiert jeden Kopf mit 256 Werten mit einer Hadamard-Transformation und
quantisiert dann die rotierten Werte gegen ein Lloyd-Max-Codebook. Jeder Kopf behält eine FP16-Norm. Die Breiten sind
3, 4, 6 und 8 Bit pro Wert. Die Anpassung ist nur der minimale quadratische Fehler; dieser Stand hat kein QJL-Residuum.

Verwende `--kv tq6` (oder `tq3`, `tq4`, `tq8`). Der Cache ist nur resident: Er bleibt im VRAM, es gibt keine Kopie
im Hostspeicher, und `--kv-resident` wird abgelehnt. Die MTP-Draft-Schicht behält ihren INT8-Cache. Das Setup bietet diese
Werte nicht an. Nichts ist Standard, und der tägliche Dienst bleibt bei int8.

Bei der konfigurierten Kapazität von 524,288 Tokens und den 12 Attention-Schichten des Modells ist der KV-Cache:

| `--kv` | Bytes pro Zelle, eine Schicht | KV bei voller Kapazität, 12 Schichten | Änderung gegenüber int8 |
| --- | ---: | ---: | ---: |
| int8 | 1,056 | 6.1875 GiB | — |
| tq8 | 1,032 | 6.046875 GiB | −0.14 GiB |
| tq6 | 776 | 4.546875 GiB | −1.64 GiB |
| tq4 | 520 | 3.046875 GiB | −3.14 GiB |
| tq3 | 392 | 2.296875 GiB | −3.89 GiB |

Die Haupteinsparung von tq6 beträgt 1.64 GiB bei voller Kapazität. Das sind berechnete Kapazitäten, keine gemessenen
Reduktionen in `nvidia-smi`: Der Experten-Cache füllt den frei werdenden VRAM, deshalb bleibt der gesamte GPU-Speicher nahe 15,600 MiB.

Der V100-Prompt-Leser ist eine zweite Option: `STRATA_PROMPT_ATTN_TQ=1` auf sm_70. Er decodiert komprimierte Kacheln
für den Prompt-Pfad; nicht gesetzt oder `0` behält den Rückfall auf die Decode-Attention. Er ändert die KV-Bytes nicht.

Gemessen auf zwei Tesla V100 mit 16 GB bei 175 W mit IQ3_S, `--spec 4`, `--layer-split 16`,
`--pipeline-windows 2`, einer kalten Obergrenze von 2,048 Ausgabe-Tokens und einem Lauf je Größe und Variante. Bei der tq6-Variante war
der Prompt-Leser aktiv. Bei 32,768 Prompt-Tokens füllte tq6 mit 1,687 Tokens/s vor, gegenüber 2,117 bei int8,
und decodierte mit 120.2 Tokens/s gegenüber 125.5. Mit aktivem Leser sank die Laufzeit der 51,746-Token-Anfrage bei tq6
von 40.3–40.4 s auf 31.7–31.9 s, weiterhin langsamer als die 25.8 s von int8. Diese Messungen zeigen keinen
Geschwindigkeitsgewinn. Ein Nadel-Test fand in beiden Varianten 15 von 15 Zellen über 122,289 Prompt-Tokens; das ist kein
allgemeines Qualitätsergebnis.

Siehe den [vollständigen Bericht](benchmarks/v100-turboquant-2026-10-25.md) und
[Details](docs/DETAILS.md#tesla-v100-fork-benchmark).

### Gyro-L Pair-Trellis auf zwei V100 — 31. Oktober 2026

Dieser Branch liest agentionais `Qwen3.8-Flash-Next-Gyro-L-TQ3_0.gguf` (108,323,559,808 Bytes, Hugging-Face-
Revision `05892758a67efff0895d23c6df10a6fdb6be8cf4`). Seine gerouteten Experten sind als Pair-Trellis-Typen
TQP5, TQP6 und TQP7 (ggml-Typen 147 bis 149) in einer Hadamard-rotierten Basis gespeichert. Der Packschritt
(`tools/iq_pack.py --compat-bf16`) und die Kernel sind in [docs/GYRO.md](docs/GYRO.md) beschrieben. CPU und CUDA
werden unterstützt; Metal und SYCL haben keine Pair-Trellis-Kernel. Das benötigt Engine 0.1.45.6 oder neuer.

Zuerst mussten zwei Engine-Fehler behoben werden. Die Experten-Kernel selbst waren bereits bitgenau gegenüber dem
ggml der Autoren:

- **Das Decode-Fenster verlor die Rotation.** Bei einem gefalteten Pack rotierte das Fenster die Experten-Eingabe und
  überschrieb dann diese rotierte Kopie mit der unrotierten, bevor die Experten sie lasen. Dadurch liefen die gate/up-Projektionen
  bei jedem decodierten Token in der falschen Basis. Die Ausgabe einer einzelnen Anfrage sah plausibel aus und wiederholte sich dann; der
  Server war ab dem ersten Token falsch. Das Fenster behält jetzt eine Markierung "Bild geschrieben" pro Schicht.
- **Der Prompt-Helfer des Layer-Splits** (optional) rotierte nur seine SwiGLU-Ausgabe und schrieb die Zeilen einer Gruppe in einen
  Puffer, der für die WMMA-Kacheln der Stufe ausgelegt war. Jetzt rotiert er die Eingabe, die er erhält, und verwaltet sein H selbst.

Auf dem Paar dieses Forks gemessen: zwei Tesla V100-PCIE-16GB (GPU 0 an PCIe Gen3 x4, GPU 1 an x16) bei 175 W, ein
Ryzen 5 3600, Layer-Split 14, `--spec 4 --spec-min-p 0.70`, int8-KV mit KV-Wachstum, `--pipeline-windows 2`,
262,144-Token-Kontext, die eigene Umgebung des Dienstes, verschachtelte Wiederholungen:

| Prompt-Tokens | Prefill (Tokens/s) |
| ---: | ---: |
| 2,328 | 632 |
| 8,199 | 1,036-1,084 |
| 16,371 | 1,344-1,411 |
| 32,679 | 1,597-1,642 |

Antworten decodieren mit 68 bis 75 Tokens/s von 2K bis 32K Kontext (Mediane wiederholter Antworten mit 520 Tokens); der
Splitpunkt verschiebt das nicht über das Rauschen zwischen Läufen hinaus. Ein Lesevorgang ist durch die Experten begrenzt, die nicht im VRAM liegen: Die
x16-Karte überträgt 15.01 GiB über PCIe für einen Lesevorgang mit 2.3K Tokens (Profil bei Layer-Split 16), und die beiden Stufen laufen
nacheinander. Deshalb ist die x4-Karte etwa 700 ms jedes Lesevorgangs untätig.

Das Pack dient über die OpenAI-kompatible API: "Paris" auf die Hauptstadtfrage, eine Folgerunde, die
86 Prompt-Tokens wiederverwendet, eine Code-Anfrage, ein `get_weather`-Werkzeugaufruf mit korrekter Antwort aus einem Werkzeugergebnis
und eine Frage zu einer Passage mit 442 Tokens, die aus der Passage beantwortet wird.

**Layer-Split 14 statt 16.** Vierzehn Schichten auf der x4-Karte und 34 auf der x16-Karte lesen bei
2.3K Tokens 7.9% schneller (632 gegenüber 586 Tokens/s) und bei 8.2K 6% schneller, bei 16.3K gleich und bei 32.7K 1.5% langsamer.
15 und 13 liegen dazwischen, und 20 ist bei 2.3K 23% langsamer und bei 8.2K 13%. Ein agentischer Durchlauf verbringt seine Zeit mit einem
Lesevorgang, deshalb verwendet dieses Profil 14.

**Was auf diesem Paar gemessen wurde und sich nicht lohnte:** `--spec 6` und `--spec 8`, `--spec-min-p 0.50` und
`0.85`, ein fester `STRATA_PREFILL_CPU_SHARE`, `STRATA_PREFILL_CPU_SHARE_MAX` über 6,144, `--prefill-pipe`,
`--max-context 131072`, `STRATA_HC_Q8=1` und ein größerer `--vram-reserve-mib`. Der Prompt-Helfer selbst ist
hier langsamer (484 gegenüber 583 Tokens/s bei 2.3K): Er kann nur die Experten der ersten Stufe übernehmen, und auf diesem Paar
ist diese Stufe die x4-Karte, die 1.77 GiB der 16.8 GiB überträgt, die ein Lesevorgang bewegt. `--mtp` und der Standardwert
`STRATA_ROUTE_TAIL_SKIP=7` lohnen sich: Ohne sie fällt der Decode auf 35.5 und 48 Tokens/s.

## V100-Benchmark

Die Tabelle unten zeichnet einen Lauf mit einer Karte auf einer Tesla V100-PCIE-16GB über PCIe Gen3 x16 auf. Die Testmaschine hat außerdem eine zweite V100 an PCIe Gen3 x4; sie wurde für diesen Lauf nicht verwendet. Ergebnisse mit zwei Karten oder von einer Maschine mit zwei vollbreiten Verbindungen können abweichen und besser sein. Das ist eine reproduzierbare Messung, kein Versprechen für die Geschwindigkeit auf anderen Systemen.

| Ziel-Prompt | Mediane Prompt-Tokens | Prefill (Tokens/s) | Decode (Tokens/s) | Höchste GPU-Temperatur |
| ---: | ---: | ---: | ---: | ---: |
| 1K | 1,036 | 398.6 | 52.0 | 56 °C |
| 4K | 4,111 | 1,214.0 | 50.8 | 61 °C |
| 8K | 8,185 | 1,442.4 | 48.1 | 63 °C |
| 16K | 16,367 | 1,490.7 | 49.8 | 67 °C |
| 32K | 32,765 | 1,495.7 | 47.3 | 72 °C |
| 64K | 65,533 | 1,484.3 | 43.0 | 81 °C |
| 128K | 131,063 | 1,033.0 † | 40.9 | 84 °C |
| 256K | 256,080 | 617.8 † | 36.8 | 84 °C |

**Testaufbau:** Qwen3.8-Flash-Next Q2_0; nur eine V100 (GPU1, PCIe Gen3 x16); Ryzen 5 3600; 48 GB DDR4-3200; CUDA 12.8; 262,144-Token-Kontext; int8-KV-Cache; automatischer Prefill; MTP mit `--spec 8` und Draft-Untergrenze 0.70; paarige Expertenvariante; 700 MiB Vision-Reserve. Jeder Wert ist der Median aus drei frischen, nicht zwischengespeicherten Anfragen (`reused=0`) mit genau 256 Ausgabe-Tokens. Die Zeitmessungen stammen vom `/metrics`-Endpunkt der Engine.

Vor jeder Anfrage ruhte die passiv gekühlte Karte mindestens 120 Sekunden und erreichte 55 °C oder weniger ohne thermische Software-Drosselung für 15 Sekunden. Die Starttemperaturen lagen bei 51–55 °C. Alle Anfragen sind enthalten. Die Prompts mit 128K und 256K erreichten 84 °C und erfuhren während der Prompt-Verarbeitung eine thermische Software-Drosselung, trotz des gekühlten Starts. Lange Läufe können drosseln. † bedeutet Drosselung während des Prompts.

Gemessen am 2026-10-02. Engine: Fork-Version 0.1.31, Build-Commit `78417ea` (SHA-256 `512b1f25d60479a2ddb66fcf1ddca5407963e45378a3c8a463413ad159edae17`). Der Benchmark wurde vor der späteren Integration von Upstream v0.1.36 und den V100-Attention-PRs aufgezeichnet; er ist keine Messung des aktuellen `main`-Builds. Die historische Tabelle wird mit ihrer ursprünglichen Herkunft beibehalten. Interpretiere sie nicht als kontrollierten Vergleich mit früheren Tabellen oder aktuellen Builds.

Rohdaten: [`summary.json`](bench/results/2026-10-02-v100-single-cooled/summary.json), [`protocol.json`](bench/results/2026-10-02-v100-single-cooled/protocol.json), [`matrix.json`](bench/results/2026-10-02-v100-single-cooled/matrix.json), [`completed-requests.json`](bench/results/2026-10-02-v100-single-cooled/completed-requests.json) und [`GPU-Telemetrie`](bench/results/2026-10-02-v100-single-cooled/gpu.csv). Siehe die [vollständige Methode und Historie](docs/DETAILS.md#tesla-v100-fork-benchmark), die [Benchmark-Anweisungen](benchmarks/README.md) und die [V100-Leistungsberichte](benchmarks/).

Neuere V100-Pull-Requests enthalten gemessene Änderungen auf Kernel-Ebene und berichten ausdrücklich, wenn durchgängige Gewinne nicht belegt sind. Zum Beispiel meldete der Prompt-Attention-Port aus Upstream-PR #600 bei seiner gemessenen Form 27.9% weniger int8-Attention-Kernelzeit, während die Änderungen am Modell-Prefill in den berichteten Läufen +1.3% bis +3.0% betrugen. Er belegte keinen zuverlässigen Decode-Gewinn. Lies den [vollständigen Bericht](benchmarks/v100-q2_0-pr600-2026-10-03.md), bevor du Ergebnisse vergleichst.

## Leistungsarbeit für die V100

Die V100-Arbeit umfasst den Prompt-Pfad, den Decode-Pfad, die Speichernutzung, Lesevorgänge vom Datenträger und die Ausführung auf mehreren GPUs. Die Änderungen unten sind in der zusammengeführten Historie des Forks; jeder Link enthält seine eigene Testmethode, Hardware-Details und Grenzen. Verbesserungen der Kernel-Zeit erzeugen nicht zwangsläufig denselben prozentualen Gewinn beim Durchsatz des ganzen Modells.

### Prompt-Verarbeitung

- **BF16-zu-FP16-Tensor-Core-GEMMs auf Volta:** Volta hat keine native BF16-GEMM-Unterstützung. Der Fork konvertiert BF16-Gewichte exakt nach FP16 und verwendet V100-Tensor-Cores für Prompt-GEMMs anstelle des skalaren Rückfalls. Die zugehörigen FP16-Prompt-Aktivierungsbilder verwenden die richtige Bitdarstellung.
- **Schnellere PLE-Tabellenlesevorgänge:** POSIX-PLE-Lesevorgänge werden auf warteschlangenbasierte `io_uring`-O_DIRECT-Operationen umgestellt, mit einem synchronen Rückfall. Ein gemessener frischer 8K-PLE-Gather sank von etwa 6.9 Sekunden auf 7 Millisekunden.
- **Q2_0-MMQ-Prefill:** Der vorhandene quantisierte Q2_0-Matrix-Matrix-Pfad von llama.cpp wird für den internen Expertentyp von Strata aktiviert. Ein gemessener Prompt mit 14,486 Tokens verbesserte den Prefill-Durchsatz um 6.5% (926.1 auf 986.2–986.9 Tokens/s).
- **Nur lebende QSA-Blöcke bewerten:** Blöcke jenseits des aktuellen Bereichs des Prompts werden nicht bewertet. Bei einem gemessenen Prompt mit 15,423 Tokens sank die QSA-Auswahlzeit um 91.1%; der durchgängige Prefill-Durchsatz stieg in den berichteten Tests von 961.0 auf 1,111.3–1,112.9 Tokens/s.
- **Tensor-Core-Prompt-Attention für Volta:** Verwende `mma.m8n8k4` auf der V100 anstelle des FP32-Rückfalls. Ein früherer PR maß etwa 1.6–1.9× Geschwindigkeit im Attention-Kernel und einen höheren Prompt-Durchsatz in seinen Tests. Der spätere Port von Upstream-PR #600 reduzierte die int8-Attention-Kernelzeit bei einer gemessenen Form um 27.9%; die berichteten durchgängigen Prefill-Änderungen waren kleiner (+1.3% bis +3.0% in seinen 4K–32K-Tests).

### Decode und Ausführung auf mehreren GPUs

- **Gebatchtes Verify-Fenster:** Mehrere spekulative Decode-Positionen laufen zusammen und senken den Startaufwand pro Token. Gepaarte historische Tests berichteten eine schnellere Ausgabeerzeugung, doch die Ergebnisse hängen vom Prompt und den Laufzeitbedingungen ab.
- **Gekachelte QSA-Blockbewertung:** Ausgewählte Schlüsseldaten werden über Abfragekacheln wiederverwendet, statt sie wiederholt zu lesen. Der gemessene Kernel mit 256 Abfragen / 65K-Blöcken sank von 3.5 ms auf 1.3 ms; seine Arithmetik pro Bewertung blieb bitidentisch.
- **Fusioniertes Volta-GR und gebatchtes KV-Anhängen:** Arbeit über Verify-Positionen wird zusammengefasst, und KV-Daten werden in weniger Starts angehängt. Auf dem getesteten Dual-V100-Aufbau stieg der Decode über 4K–32K-Prompts um 5.6–17.3%, während der Prefill annähernd unverändert blieb.
- **Asymmetrisches Dual-V100-Tuning:** Die Behandlung von Host-Zeilen, der Dispatch des Geräteplans und das Speichereigentum werden für eine Konfiguration mit zwei Karten verbessert. Eine frühere installierte Konfiguration zeigte Prefill-Gewinne von 13.3–18.7% bei 4K–32K, doch die Decode-Ergebnisse waren gemischt und beide GPUs drosselten während dieses Laufs thermisch. Diese Werte isolieren nicht jede Codeänderung.
- **Aufwand für Gewichte und Dispatch senken:** Bei einem expliziten Split werden nur die Schichten geladen, die jede Karte besitzt. Unnötige CPU-Aktivierungsquantisierung und leere Dispatch-Barrieren entfallen, und paarige Experten-Kernel für Zeilen können optional laufen. Die gemessene native dichte Zuweisung von GPU0 sank von 1,376.20 MiB auf 541.27 MiB. Der paarige Zeilenmodus ist optional; die berichteten Ergebnisse für das ganze Modell belegten keine Decode-Verbesserung.

### Attention-Kernel und Speicherverkehr

- **Decode-Attention (Port von Upstream-PR #540):** Score-Shuffles und Verkehr im gemeinsamen Speicher werden reduziert, während die Akkumulationsreihenfolge erhalten bleibt. Die gemessene Kernelzeit sank auf GPU0 für M=1 von 52.90 auf 31.13 Mikrosekunden und für M=8 von 180.42 auf 111.85 Mikrosekunden. Die berichteten Decode-Änderungen am ganzen Modell waren klein; der Prefill war praktisch unverändert.
- **Prompt-Attention (Port von Upstream-PR #600):** Die Volta-Tensor-Core-Implementierung mit vier unabhängigen MMA-Berechnungen wird verwendet. Der PR berichtet Korrektheitsprüfungen und Messungen auf Kernel- und Modellebene; die Kernel-Reduktion von 27.9% darf nicht als 27.9% mehr Geschwindigkeit des ganzen Modells gelesen werden.
- **KV-Gather und GDN-Rekurrenz (Ports von Upstream-PR #627):** Breitere ausgerichtete int8-KV-Ladevorgänge mit einem sicheren Rückfall auf die ursprüngliche Breite und mehrere Akkumulatoren für die V100-Prompt-Rekurrenz werden verwendet. Die gemessenen Kernelzeiten verbesserten sich um etwa 10% für einen 32K-KV-Gather und um etwa 7% für eine Rekurrenz mit 4,096 Tokens. Der paarige Modell-Prefill unterschied sich um weniger als 0.2%; ein zuverlässiger durchgängiger Decode-Gewinn wurde nicht belegt.

### Build-Unterstützung und Belege

Der Fork hat außerdem die CUDA-Konfiguration und das Linken für `sm_70` repariert, die Zulassung von V100-Geräten durch Upstream-Zusammenführungen hindurch erhalten und Upstream-Änderungen angepasst, wenn ihre Architekturanforderungen Volta ausschließen. Diese Kompatibilitätskorrekturen halten die optimierten Pfade baubar; sie sind selbst keine Leistungsaussagen.

Für die vollständigen Belege und den Umfang der Implementierung siehe [PR #1](https://github.com/jmnargi/Strata-V100/pull/1), [#2](https://github.com/jmnargi/Strata-V100/pull/2), [#5](https://github.com/jmnargi/Strata-V100/pull/5), [#6](https://github.com/jmnargi/Strata-V100/pull/6), [#11](https://github.com/jmnargi/Strata-V100/pull/11), [#12](https://github.com/jmnargi/Strata-V100/pull/12), [#15](https://github.com/jmnargi/Strata-V100/pull/15), [#16](https://github.com/jmnargi/Strata-V100/pull/16) und [#17](https://github.com/jmnargi/Strata-V100/pull/17). Die Ergebnisse verwenden unterschiedliche Daten, Builds, Prompts, GPU-Anordnungen und Protokolle. Sie sind historische Messungen, kein einzelner kontrollierter Vorher-Nachher-Vergleich und keine Garantie für Gewinne auf jedem V100-System.

## Weitere Hardware

- **Ältere Grafikkarten** (Tesla P40 / V100, GTX 10, Radeon VII / MI50, RX 6700 XT, RX 5500 XT): [Ältere GPUs](docs/OLDER_GPUS.md).
- **Intel Arc**, unter Linux aus dem Quellcode gebaut: [Intel Arc](docs/INTEL_ARC.md).
- **AMD Ryzen AI Max (Strix Halo)**, unter Linux aus dem Quellcode gebaut: [Strix Halo](docs/STRIX_HALO.md).
- **Ältere Prozessoren ohne AVX2**: Sie funktionieren, aber langsam. [Ältere CPUs](docs/INSTALL.md#older-cpus-experimental).

## Installation und Start

Dieser Fork folgt der Installationserfahrung von Upstream. Für vollständige und aktuelle Plattformanforderungen, Modellauswahl und Optionen siehe die [Installationsanleitung](docs/INSTALL.md), den [Modellleitfaden](docs/MODELS.md) und die [Upstream-Setup-Anleitung](docs/AI_SETUP.md).

- **Windows:** Lade dieses Repository herunter oder klone es, und starte `START-HERE.bat`.
- **Linux:** Klone dieses Repository und starte `./setup.sh`.
- Folge den Eingabeaufforderungen, um ein Modell und eine Kontextgröße zu wählen. Das Setup lädt Modelldaten herunter und startet den lokalen Dienst.
- Öffne die Adresse, die das Setup ausgibt (normalerweise `http://127.0.0.1:8080`).

Für eine Tesla V100 verwende die CUDA-12-Engine mit `sm_70`-Unterstützung. Das Setup wählt für Volta die Engine für ältere GPUs; unter Linux wird sie lokal gebaut. Das Windows-CUDA-12-Archiv von Upstream enthält Volta-Code, wurde aber auf dieser Linux-Maschine nicht getestet. Siehe den [Leitfaden für ältere GPUs](docs/OLDER_GPUS.md) für Toolkit- und Treiberanforderungen. Befolge für Docker die [Docker-Anweisungen](docs/INSTALL.md) und baue für die Zielarchitektur der GPU.

## Strata verwenden

- **Web-App:** Verwende die lokale URL, die das Setup ausgibt, für den Chat und den Live-Monitor.
- **OpenAI-kompatible API:** Setze die Basis-URL deiner App auf `http://127.0.0.1:8080/v1`.
- **Anthropic-kompatible API:** Verwende `http://127.0.0.1:8080/v1/messages`.
- **OpenAI Responses API:** Verwende `http://127.0.0.1:8080/v1/responses`. Dieser zustandslose Endpunkt unterstützt Werkzeugaufrufe im Codex-Stil; er speichert `previous_response_id` nicht.
- **Bilder, MCP, mehrere GPUs und Konfiguration:** Siehe [Details](docs/DETAILS.md), [MCP-Server](docs/MCP_SERVER.md) und [Leitfaden für mehrere GPUs](docs/MULTI_GPU.md).

Der Server hört normalerweise auf localhost. Wenn du ihn für andere Maschinen freigibst, konfiguriere einen API-Schlüssel und verwende ein vertrauenswürdiges Netzwerk. Veröffentliche keine Geheimnisse oder private Konfiguration in Issues, Benchmark-Ergebnissen oder Pull-Requests.

### Gesprächs-Cache auf der Festplatte (optional)

Der Gesprächszustand hat drei Stufen. L1 ist die aktive Sitzung mit ihren residenten Prompt-Checkpoints. L2 ist der optionale Cache im Host-RAM (`--conversation-cache-mib`). L3 ist ein optionaler Speicher auf dem Datenträger, der gespeicherte Unterhaltungen und Checkpoint-Gruppen über Neustarts hinweg behält. Gestreamte KV-Schichten verwenden ihre maßgeblichen Host-Pools, wenn die Engine einen Zustand erfasst.

Um den Speicher auf dem Datenträger zu verwenden, füge `--conversation-cache-disk strata-conversations --conversation-cache-disk-gib 25` zu den Engine-Argumenten hinzu. Der Pfad und ein positives GiB-Budget sind gemeinsam erforderlich. Der Speicher ist standardmäßig aus. Er benötigt `--serve`, funktioniert mit `--conversation-cache-mib 0` und unterstützt `--layer-split` auf mehreren GPUs. Optional: `--conversation-cache-disk-slots N` begrenzt die Zahl der Datensätze, und `--conversation-cache-disk-min-free-mib N` hält freien Platz im Dateisystem.

Die Engine bewahrt einen aktiven Checkpoint, bevor die Aufbewahrung ihn entfernt oder eine Anfrage seinen Zweig überschreibt. Jeder gespeicherte Checkpoint enthält den GDN-Zustand, den PLE-Verlauf, den Zustand des QSA-Indexers und den erforderlichen KV-Unterbau. Token-KV-Daten allein reichen nicht aus, um dieses Modell fortzusetzen.

Verwandte Checkpoints teilen ein KV-Abbild je Archivgruppe. Die Engine verschiebt und erweitert dieses Abbild, statt für jeden Checkpoint das vollständige Präfix zu kopieren. Sie kopiert nur neue oder geänderte KV-Seiten, wenn der Unterbau noch gültig ist. Eine Gruppe wächst nicht weiter, bevor sie die Metadatengrenze des Datenträgers erreicht. Eine spätere Gruppe hat ihren eigenen Unterbau. Die Engine kombiniert einen gespeicherten Checkpoint nicht mit KV-Bytes aus einer anderen Berechnung des Präfix.

Archivgruppen verwenden dieselben Byte- und Slot-Grenzen von L2 wie gespeicherte Unterhaltungen. Opfer der L2-Kapazität wechseln nach L3, wenn die Datenträgerstufe aktiviert ist. Eine erfolgreiche Aufnahme in L2 schreibt das Abbild nicht zusätzlich auf den Datenträger. Wenn ein Abbild nicht in L2 passt, versucht die Engine, es in L3 zu speichern. Grenzen für physischen RAM und freien Speicherplatz gelten weiterhin. Die Einstellung für die Mindest-Tokenzahl gilt auch für Checkpoint-Archive.

Das Bewahren eines Checkpoints ist eine Optimierung, keine Voraussetzung. Wenn die Host-RAM-Untergrenze oder das Datenträgerbudget das Abbild ablehnt, verwirft die Engine den Checkpoint, protokolliert das Verwerfen und bedient weiter. Sie stoppt den Server nicht. Die Sitzung bleibt korrekt; nur die Wiederverwendung dieses Checkpoints geht verloren.

Die Suche wählt das längste kompatible Präfix über die aktive Sitzung, L2, L3 und inaktive Batch-Slots. Tokens, die Identität des Abbilds und der Modus des Steuervektors müssen übereinstimmen. Bei einer Bearbeitung einer früheren Nachricht endet die Wiederverwendung an einem gespeicherten Checkpoint vor dem ersten geänderten Token. Die Engine muss das Suffix erneut verarbeiten; sie kann nicht an einem beliebigen passenden Token ohne den rekurrenten Zustand an dieser Position fortsetzen. Ein größerer `--prompt-cache` ersetzt nicht das Bewahren von Checkpoints.

Ein L2-Archiv bleibt nach der Wiederherstellung verfügbar. Ein gültiges L3-Abbild wird nach L2 befördert, wenn das Bytebudget und die Untergrenze des physischen RAM es zulassen. Nur eine begrenzte Teilmenge seiner Checkpoints wird in der aktiven Sitzung resident. Die Wiederherstellung eines Zweigs entfernt keine anderen Checkpoints aus dem gespeicherten Archiv.

Der Speicher auf dem Datenträger unterstützt nur anhängbare Unterhaltungsketten. Eine fortlaufende Kette kann neue Token-IDs, Checkpoints, laufenden Zustand und KV-Seiten anhängen. Eine L2-Verdrängung schreibt ein vollständiges Startabbild; sie benötigt keine GPU-Erfassung. Ein Zurückspulen beginnt einen getrennten Zweig, statt die Quellkette zu ersetzen. Das Dateiformat ändert sich für Checkpoint-Archive nicht.

Datenträgerpaneele haben Prüfsummen und Versionsfelder. Ein Start verwendet eine temporäre Datei und eine Umbenennung. Ein unvollständiges Anhängen wird auf das letzte vollständige Paneel zurückgeführt. Ungültige oder inkompatible Datensätze werden entfernt. Der Speicher verdrängt die am längsten nicht verwendeten Ketten, um sein Bytebudget, seine Datensatzgrenze und seine Untergrenze für freien Platz durchzusetzen. Somit bietet weder L2 noch L3 unbegrenzte Historie. Wenn beide Stufen den erforderlichen Checkpoint verdrängt haben, muss die Anfrage das fehlende Präfix erneut verarbeiten.

Die Engine erfasst eine gemeinsame System-Prompt-Wurzel einmal in L2, wenn die Wurzelaufbewahrung aktiviert ist. L3 erhält die Wurzel bei einer L2-Verdrängung oder einer fehlgeschlagenen Aufnahme in L2. Bei deaktiviertem L2 kann eine aktivierte L3 die Wurzel direkt erhalten. `--conversation-cache-keep-root` und `--conversation-cache-disk-keep-root` steuern die Wurzelaufbewahrung in jeder Stufe. Ihre Varianten mit `--no-` deaktivieren sie. Wurzeln haben eine niedrigere Verdrängungspriorität als gewöhnliche Abbilder; ausdrückliche Checkpoint-Pins von Anfragen bleiben geschützt.

Die Datenträgerstufe fügt keine gleichzeitige Ausführung hinzu. Siehe [Batch-Slots](docs/BATCHING.md) für die Speicherkosten gleichzeitiger Anfragen. Die Erfassung muss abgeschlossen sein, bevor eine Anfrage die erforderlichen Sitzungsbytes überschreibt. Die Wiederherstellung ist synchron. Schreibvorgänge bei L2-Verdrängungen verwenden Host-Snapshots und können die Latenz der Datenträger-E/A erhöhen. Bemesse beide Cache-Budgets für die Unterhaltungen und die Checkpoint-Historie, die du brauchst. Siehe den [V100-Leitfaden](docs/NVIDIA_V100.md) für die Prüfung der Checkpoint-Aufbewahrung und den [Benchmark-Bericht zum Datenträger](benchmarks/v100-l3-conversation-cache-2026-10-03.md) für frühere Messungen.

#### Messergebnis auf der V100

In der Layer-Split-Konfiguration mit zwei V100 dieses Forks dauerte das Fortsetzen einer Unterhaltung mit 33,725 Tokens aus einem 25 GiB großen NVMe-Cache 1,614.6 ms zum Lesen und 315.8 ms zum Wiederherstellen. Die vollständige fortgesetzte Prompt-Phase einschließlich 22 neuer Tokens dauerte 3,015.1 ms. Ein kalter Lesevorgang desselben Präfix mit 33,725 Tokens dauerte 20,332.7 ms. Das ist eine **6.74-fache Beschleunigung** und eine **Reduktion der Prompt-Latenz um 85.2%** für die fortgesetzte Anfrage. Der Datensatz mit 943.6 MiB stellte den bytegenauen Zustand des Hauptmodells und identische Ausgaben auf beiden GPUs wieder her. Die Wiederherstellung nach einem Neustart bestand die bytegenaue Parität, und der Rückfall bei beschädigten Datensätzen erzeugte passende Ausgaben. Siehe den [Benchmark-Bericht](benchmarks/v100-l3-conversation-cache-2026-10-03.md).

Gemessen auf dem Dual-V100-Layer-Split-Aufbau dieses Forks (Qwen3.8-Flash-Next Q2_0, int8-KV): Zwei
Unterhaltungen wechselten sich durch einen Scratch-Cache ab und wurden über 8 Anfragen korrekt bedient, mit
`corruptions=0 evictions=0`. Eine Kettendatei je Unterhaltung wuchs über drei Parks von je +12 neuen Tokens
von 238 MiB -> 594 MiB (ein Speicher mit vollständigen Snapshots hätte bei 100k+ Tokens je Park eine neue Datei
von ~2.4 GiB geschrieben). Der erste Park eines frischen kurzen Chats legt die Kette an
(~118 MiB: eine K/V-Seite je Schicht plus der feste laufende Zustand des Modells). Danach ist jeder Park
das Delta (+12 Tokens ~ 118 MiB), und der Pfad zum Zurücklesen und Wiederherstellen ist unverändert (~0.4 s zum Lesen,
~0.04-0.08 s zum Wiederherstellen).

### Lazy Vision (optional)

Der Bildencoder (`strata-vision`) belegt etwa 1.74 GiB VRAM (1.43 GiB auf der ersten GPU, 0.31 GiB auf der zweiten), solange er läuft. Mit Lazy Vision läuft er zwischen Bildanfragen nicht: Dieser VRAM bleibt bei den Experten-Caches, sodass reine Textarbeit die volle Zahl residenter Experten behält.

Um es zu aktivieren, füge Folgendes zur Modellkonfiguration hinzu:

- `"vram_elastic": true` (das Engine-Flag `--vram-elastic`): Die Experten-Caches sind segmentiert, sodass die Engine zwischen Anfragen VRAM zurückgeben und wieder nehmen kann.
- `"vision": {"lazy": true, "idle_s": 900}`: Der Encoder startet bei der ersten Bildanfrage und stoppt nach `idle_s` Sekunden ohne Bildanfrage (900 = 15 Minuten). Der Encoder benötigt den Modus der Engine für jeweils eine Anfrage. Kombiniere ihn deshalb nicht mit `"parallel"` (oder gib dem Encoder seinen VRAM auf andere Weise, zum Beispiel mit `POST /v1/vram`).

Verhalten:

- Eine Textanfrage startet den Encoder nie. Eine Bildanfrage fordert die Engine zuerst auf, den VRAM der GPUs zurückzugeben - eine Reserve pro GPU, `[1700, 600]` bei einem Layer-Split und `[1800]` bei einer GPU, überschreibbar mit `"vram_mib"`. Dann startet sie den Encoder und codiert das Bild, während die Engine untätig ist.
- Nach `idle_s` Sekunden ohne Bildanfrage stoppt der Encoder, und die Experten-Caches wachsen wieder auf ihre volle Größe. Codierte Bilder bleiben auf dem Datenträger zwischengespeichert; dasselbe Bild wird deshalb nicht erneut codiert.
- Gemessen auf dem Layer-Split mit zwei V100 dieses Forks: Die Caches halten **14,005** residente Experten, während der Encoder untätig ist (12,688 mit residentem Encoder). Eine kalte Bildanfrage wurde in **5.9 s** beantwortet. Beide Caches kehrten nach dem Entladen des Encoders zu 14,005 zurück. Siehe [Details](docs/DETAILS.md).

## Wenn etwas schiefgeht

Die häufigen Probleme und ihre Korrekturen stehen in [TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md). Diese Seite behandelt
die Desktop-App, den Befehl `strata` und den Installer. Die vollständige Tabelle der Fehlermeldungen steht in
[DETAILS.md](docs/DETAILS.md#troubleshooting).

Wenn der Server startet, aber eine Anfrage fehlschlägt, führe `strata doctor` aus. Er liest die installierten Komponenten und
gibt aus, was fehlt oder veraltet ist; siehe [TERMINAL.md](docs/TERMINAL.md#doctor).

Wenn eine Antwort langsam statt fehlerhaft ist, lies [TUNING.md](docs/TUNING.md). Sie nennt die Einstellung, die
für jedes Ziel wichtig ist, und die gemessenen Kosten jeder Einstellung.

## Mitwirken

Beiträge sind willkommen. Du brauchst keine V100, um zu helfen: Dokumentation, Tests, Setup, Serververhalten, API-Kompatibilität und Verbesserungen für andere unterstützte Geräte sind nützlich. V100-spezifischer Code und Leistungsergebnisse profitieren von Validierung auf echter Volta-Hardware.

1. Prüfe [offene Issues](https://github.com/jmnargi/Strata-V100/issues) und [Pull-Requests](https://github.com/jmnargi/Strata-V100/pulls), um doppelte Arbeit zu vermeiden.
2. Prüfe bei breiten Upstream-Funktionen, ob die Änderung in [das Upstream-Repository von Niko](https://github.com/Niko1221/Strata) gehört. Dieser Fork integriert regelmäßig Upstream-Änderungen; gezielte V100-Korrekturen und Messungen können hier vorgeschlagen werden.
3. Mache eine gezielte Änderung, beschreibe die Hardware und die genauen Schritte zum Testen und füge relevante Tests hinzu.
4. Melde bei Leistungsaussagen die Basislinie und den Kandidaten, die vollständigen Laufzeiteinstellungen, die Last, die Zahl der Wiederholungen und die Grenzen. Füge nach Möglichkeit Rohdaten oder einen reproduzierbaren Befehl hinzu. Trenne die Kernel-Zeit vom Durchsatz des ganzen Modells und behaupte keinen Gewinn, den die Messungen nicht zeigen.
5. Öffne einen Pull-Request gegen den Branch `main` dieses Repositorys. Erkläre, ob die Änderung V100-spezifisch, eine Upstream-Integration oder eine allgemeine Strata-Verbesserung ist. Halte Anmeldedaten, persönliche Konfiguration und unabhängige lokale Dateien aus der Änderung heraus.

Siehe [`benchmarks/README.md`](benchmarks/README.md) und [`docs/DETAILS.md`](docs/DETAILS.md) für den aktuellen Messansatz. Beiträge unterliegen weiterhin der Projektlizenz und den Lizenzen der enthaltenen Komponenten und Modelldateien.

## Projektgeschichte

Dieser Fork entwickelt die V100-Unterstützung und V100-spezifische Arbeit weiter und behält dabei die Upstream-Entwicklung von Strata bei. Neuere Integrations- und Performance-Pull-Requests zeigen diesen Prozess:

- [#14 — Upstream v0.1.36 zusammenführen und dabei die V100-Unterstützung erhalten](https://github.com/jmnargi/Strata-V100/pull/14)
- [#15 — Gemessene Änderungen an V100-KV-Gather und GDN-Rekurrenz aus Upstream-PR #627 portieren](https://github.com/jmnargi/Strata-V100/pull/15)
- [#16 — Shuffles und Verkehr im gemeinsamen Speicher bei der V100-Decode-Attention reduzieren](https://github.com/jmnargi/Strata-V100/pull/16)
- [#17 — Den Volta-Prompt-Attention-Kernel aus Upstream-PR #600 portieren](https://github.com/jmnargi/Strata-V100/pull/17)
- [#88 — Die Pair-Trellis-Experten von Gyro-L lesen (TQP5/6/7), das gefaltete Decode-Fenster und den Prompt-Helfer des Layer-Splits korrigieren und das V100-Profil auf `layer_split 14` umstellen](https://github.com/jmnargi/Strata-V100/pull/88)

Diese Berichte beschreiben bestimmte getestete Änderungen; sie sind keine pauschale Aussage über schnellere Leistung bei allen Modellen oder GPUs. Für mehr Details durchsuche die [vollständige Historie der Pull-Requests](https://github.com/jmnargi/Strata-V100/pulls?q=is%3Apr+is%3Amerged) und die [Commit-Historie](https://github.com/jmnargi/Strata-V100/commits/main).

## Danksagung und Lizenz

Dieses Repository ist ein Fork von [Strata von Niko1221](https://github.com/Niko1221/Strata). **Wir sind Niko und allen Upstream-Mitwirkenden sehr dankbar.** Ihre ursprüngliche Arbeit ist der Grund, warum dieser Fork, seine V100-Unterstützung und diese lokale Modell-Laufzeit möglich sind. Wir möchten die Upstream-Arbeit würdigen und ihr folgen, während wir ihre Änderungen regelmäßig übernehmen.

Strata enthält [llama.cpp / ggml](https://github.com/ggml-org/llama.cpp) und andere Open-Source-Komponenten. Das Projekt ist unter der [MIT-Lizenz](LICENSE) lizenziert; Komponenten und Modelldateien können eigene Bedingungen haben. Siehe die [Hinweise zu Danksagungen und Lizenz](docs/HOW_IT_WORKS.md#credits).
