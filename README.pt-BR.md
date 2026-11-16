# Strata-V100

**Um fork do [Strata](https://github.com/Niko1221/Strata) mantido pela comunidade e focado na V100.** Este fork mantém o suporte à NVIDIA Volta (`sm_70`) funcionando. Ele também desenvolve e mede mudanças de desempenho em hardware Tesla V100, e integra periodicamente melhorias upstream para recursos e correções além da V100.

O Strata é um runtime local de código aberto para o [Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next). Ele combina GPU, memória do sistema e armazenamento para rodar esse modelo grande em um computador pessoal. Este fork preserva o projeto Strata mais amplo. Outro hardware NVIDIA e AMD compatível, a interface web, a API compatível com OpenAI, ferramentas, entrada de imagens e recursos upstream continuam importantes também.

> **Um grande crédito a Niko ([@Niko1221](https://github.com/Niko1221)) e aos contribuidores originais do Strata.** Este projeto se baseia no trabalho deles. O runtime do modelo, a arquitetura, os recursos do Strata e o trabalho que tornou possível rodar este modelo localmente vêm todos do projeto upstream. O fork para V100 não existiria sem essa base. Obrigado por tornar isso possível e por continuar a desenvolver o Strata.

[Strata upstream](https://github.com/Niko1221/Strata) · [Issues](https://github.com/jmnargi/Strata-V100/issues) · [Pull requests](https://github.com/jmnargi/Strata-V100/pulls) · [Discussões](https://github.com/jmnargi/Strata-V100/discussions)

**Idiomas:** [English](README.md) · [Deutsch](README.de.md) · [Español](README.es.md) · [Français](README.fr.md) · [日本語](README.ja.md) · [Português (Brasil)](README.pt-BR.md) · [简体中文](README.zh-CN.md)

## Conteúdo

- [Modelos e quants que rodam aqui](#modelos-e-quants-que-rodam-aqui)
- [No que este fork se concentra](#no-que-este-fork-se-concentra)
- [Documentação](#documentação)
- [Notas de versão](#notas-de-versão)
- [Benchmark da V100](#benchmark-da-v100)
- [Trabalho de desempenho na V100](#trabalho-de-desempenho-na-v100)
- [Outro hardware](#outro-hardware)
- [Instalar e iniciar](#instalar-e-iniciar)
- [Usar o Strata](#usar-o-strata)
- [Algo deu errado](#algo-deu-errado)
- [Como contribuir](#como-contribuir)
- [Histórico do projeto](#histórico-do-projeto)
- [Créditos e licença](#créditos-e-licença)

## Modelos e quants que rodam aqui

O Strata roda o **Qwen3.8-Flash-Next** a partir de sete famílias de releases quantizados, além de um **Qwen3.6-35B-A3B**
menor. Todas as linhas abaixo têm suporte do engine e do setup deste fork. A última coluna indica o que já foi
executado de ponta a ponta no par de V100 daqui. Tamanhos, qualidade e velocidade por tamanho: [docs/MODELS.md](docs/MODELS.md).

| Modelo / quant | Especialistas roteados | De onde vêm os pesos | Download | Execução neste par de V100 |
| --- | --- | --- | ---: | --- |
| **NVFP4 — `local-inference-lab` (QAD)** | NVFP4, grupo 16; atenção e especialistas compartilhados MXFP8; tabela n-gram NVFP4 | [`local-inference-lab/Qwen3.8-Flash-Next-NVFP4`](https://huggingface.co/local-inference-lab/Qwen3.8-Flash-Next-NVFP4), convertido no seu PC ([NVFP4.md](docs/NVFP4.md#the-qad-distillation-local-inference-labqwen38-flash-next-nvfp4)) | 106 GB | sim — 62.3 tokens/s de decode |
| **NVFP4 — NVIDIA, treinado para FP4 (`FP4-Local`, QAD)** | NVFP4, grupo 16 (escalas no formato CSF da NVIDIA) | [`nvidia/Qwen3.8-Flash-Next-FP4-Local`](https://huggingface.co/nvidia/Qwen3.8-Flash-Next-FP4-Local), convertido no seu PC ([NVFP4.md](docs/NVFP4.md#nvidias-fp4-local-checkpoint)) | 102 GB (+110 GB durante a preparação) | com suporte; não executado aqui |
| **NVFP4 — versão da NVIDIA** | NVFP4, grupo 16 | [`Maximilian228/Qwen3.8-Flash-Next-NVIDIA-NVFP4-Strata`](https://huggingface.co/Maximilian228/Qwen3.8-Flash-Next-NVIDIA-NVFP4-Strata) (pack pronto) | 128 GB | com suporte; não executado aqui |
| **NVFP4 — versão do Strata** | NVFP4, grupo 16 | [`Maximilian228/Qwen3.8-Flash-Next-NVFP4-GPTQ-Strata`](https://huggingface.co/Maximilian228/Qwen3.8-Flash-Next-NVFP4-GPTQ-Strata) (pack pronto, GPTQ) | 128 GB | com suporte; não executado aqui |
| **Unsloth Dynamic v3.1 (1-4 bits)** | UD-IQ1_M, UD-Q2_0, UD-IQ2_XS, UD-Q2_K_XL, UD-IQ3_XXS, UD-Q3_K_XL, UD-IQ4_XS-v31 | [`unsloth/Qwen3.8-Flash-Next-GGUF`](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF), revisão de 2026-11-04 | 68-94 GB | sim — UD-IQ4_XS-v31, 81 tokens/s |
| **Unsloth Dynamic (revisão anterior)** | UD-IQ4_XS, UD-Q4_K_XL, UD-Q5_K_XL, UD-Q6_K_XL, Q8_0 | [`unsloth/Qwen3.8-Flash-Next-GGUF`](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF) | 94-188 GB | com suporte; não executado aqui |
| **i-quants e Q2_0 do ISTA-DASLab (2-3.5 bits)** | Q2_0, IQ2_XS, IQ3_XXS, IQ3_S | [`ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF`](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF) | 66-84 GB | sim — IQ3_S, o perfil diário anterior |
| **Gyro (agentionai), trellis** | Gyro-S TQ1_0, Gyro-M TQ2_0 (pair trellis) | [`agentionai/Qwen3.8-Flash-Next-Gyro-GGUF`](https://huggingface.co/agentionai/Qwen3.8-Flash-Next-Gyro-GGUF) ([GYRO.md](docs/GYRO.md)) | 58.5 / 92 GB | sim — Gyro-M, cerca de 70 tokens/s |
| **Coder (ISTA-DASLab), metade dos especialistas** | IQ1_M, IQ2_S-IQ4_XS como o IQ3_S | [`ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-Coder-GGUF`](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-Coder-GGUF) | 58 GB | com suporte; não executado aqui |
| **OrcaRouter Uncensored** | IQ3_XXS | O Flash-Next Uncensored da `OrcaRouter` ([ORCA.md](docs/ORCA.md); uma etapa manual de empacotamento, não uma opção do instalador) | 84 GB | com suporte; não executado aqui |
| **Qwen3.6-35B-A3B** (um modelo menor) | UD-Q4_K_XL, UD-Q6_K_XL e Q8_0 da Unsloth, com a camada de rascunho MTP | [`unsloth/Qwen3.6-35B-A3B-MTP-GGUF`](https://huggingface.co/unsloth/Qwen3.6-35B-A3B-MTP-GGUF) ([MODELS.md](docs/MODELS.md#qwen36-35b-a3b)) | 23-38 GB | com suporte; não executado aqui |

As linhas NVFP4 precisam de placas NVIDIA: as placas RTX 50 multiplicam na largura FP4, e as placas mais antigas,
inclusive a V100, em precisão de 8 bits. As duas linhas QAD são preparadas no seu PC a partir dos safetensors dos
autores; nenhum peso quantizado deles é redistribuído aqui. As linhas Gyro e trellis precisam do engine 0.1.45.1 ou
mais novo; as linhas Unsloth com downs Q5_0 precisam do 0.1.45.7.

## No que este fork se concentra

- **Suporte a Tesla V100 / Volta (`sm_70`)**: manter as builds e o suporte em tempo de execução normais da V100. As builds para V100 usam um toolkit CUDA 12.x; o CUDA 13 não compila código `sm_70`.
- **Máquina de teste:** duas placas Tesla V100-PCIE-16GB; a GPU 0 usa PCIe Gen3 x4 e a GPU 1 usa Gen3 x16. As medições de pipeline abaixo usam as duas placas. O benchmark histórico de placa única usa apenas a placa x16. Um sistema com dois links Gen3 x16 completos não foi medido aqui.
- **Integração upstream**: este não é um substituto separado do Strata. As mudanças upstream são integradas periodicamente, com a compatibilidade com a V100 verificada e corrigida quando necessário. Novos recursos upstream podem ter requisitos de hardware acima de `sm_70`; eles não aceleram a V100 automaticamente.
- **A comunidade Strata mais ampla**: melhorias, relatos de bugs, testes, documentação e contribuições para outras plataformas compatíveis são bem-vindos.

O suporte à V100 não implica que todo recurso upstream ou toda outra configuração de GPU tenha sido testada na V100. Verifique o código, as notas de versão e os relatórios de medição relevantes antes de depender de um recurso específico de hardware.

## Documentação

O [índice da documentação](docs/README.md) lista todas as páginas e o propósito de cada uma. Estas páginas são a
referência:

- [Flags do engine](docs/ENGINE_FLAGS.md): toda opção de linha de comando, seu valor e seu padrão.
- [Variáveis de ambiente](docs/ENVIRONMENT.md): toda variável que o engine, o servidor e o
  terminal leem.
- [Servidor](docs/SERVER.md): opções do servidor, as chaves da configuração de execução e todos os endpoints da API.
- [Terminal](docs/TERMINAL.md): todo comando e subcomando `strata`.
- [Ajuste](docs/TUNING.md): as configurações por objetivo, as otimizações e os resultados medidos.
- [Como funciona](docs/HOW_IT_WORKS.md) e o [glossário](docs/GLOSSARY.md): as partes do
  engine e os termos que esta documentação usa.
- [Detalhes](docs/DETAILS.md): o registro de medições, os formatos de requisição e o histórico de configurações.

As páginas seguem o [estilo da documentação](docs/STYLE.md), que é o ASD-STE100 (Simplified Technical
English).

## Notas de versão

### Engine 0.1.45.8 — 16 de novembro de 2026

A [versão v0.1.45.8](https://github.com/jmnargi/Strata-V100/releases/tag/v0.1.45.8)
dá ao engine V100 atual uma versão distinta. Inclui a correção do deslocamento de
linhas no prefill FP16 dos especialistas, trabalho no decode NVFP4, economia
opcional de memória HC_Q8 entre placas, melhorias opcionais na pontuação de prompts
e no KV transferido, e progresso do prompt por slot. Esta mudança de versão não
altera o código de inferência nem as configurações padrão. As medições e os limites
estão em [NVIDIA_V100.md](docs/NVIDIA_V100.md).

### Conjuntos Unsloth Dynamic v3.1 (1-4 bits) — 8 de novembro de 2026

Resolve a [#95](https://github.com/jmnargi/Strata-V100/issues/95): o menu `--family unsloth` do setup parava em
UD-IQ4_XS (cerca de 4 bits). Agora ele também oferece os arquivos Dynamic v3.1 da Unsloth, de ~1 a ~4 bits, na revisão
que a Unsloth publicou em 2026-11-04 (commit `d9620119`):

| Tamanho | Download | Especialistas roteados | RAM com todos os especialistas na RAM | Observações |
| --- | ---: | ---: | ---: | --- |
| UD-IQ1_M | 68.0 GB | 32.4 GiB | ~48 GB | o menor; o mais fraco em trabalho cuidadoso ou longo |
| UD-Q2_0 | 70.5 GB | 34.8 GiB | ~51 GB | especialistas Q2_0, como o pack de 2 bits mais rápido |
| UD-IQ2_XS | 75.8 GB | 39.4 GiB | ~55 GB | 1 camada com downs Q5_0 |
| UD-Q2_K_XL | 79.6 GB | 42.8 GiB | ~59 GB | 1 camada com downs Q5_0 |
| UD-IQ3_XXS | 83.6 GB | 46.8 GiB | ~63 GB | sem Q5_0 |
| UD-Q3_K_XL | 89.5 GB | 52.3 GiB | ~68 GB | 20 camadas com downs Q5_0 |
| UD-IQ4_XS-v31 | 93.7 GB | 56.2 GiB | ~72 GB | 33 camadas com downs Q5_0; nome próprio no catálogo, porque os arquivos dele no Hub se chamam UD-IQ4_XS |

Os cinco arquivos que já eram oferecidos (UD-IQ4_XS, UD-Q4_K_XL, UD-Q5_K_XL, UD-Q6_K_XL, Q8_0) continuam fixados no
commit anterior. O segundo layout do UD-Q4_K_XL (quatro shards, tabela n-gram por último) no novo commit é aceito a
partir de arquivos que já estão no disco. O UD-IQ1_S continua recusado: ele mistura dois formatos de especialista em uma
camada.

Trabalho no engine que isso exigiu: o produto escalar do especialista Q5_0. O deslocamento de -16 por peso era aplicado
à soma das ativações originais, a forma CUDA do llama.cpp. Por isso, uma camada cuja projeção down é Q5_0 saía 2-3% menos
precisa por especialista do que qualquer outro formato down. Agora ele multiplica a soma de ativações quantizada, como
os caminhos Q5_1 e Q4_0 e o ggml-cpu (CUDA e SYCL). A medição nesta máquina usou a paridade sintética de especialistas
(`native_expert_parity --synthetic`, gate-up IQ2_XXS / IQ2_S / IQ2_XS / IQ3_XXS / IQ4_XS com downs Q5_0). O erro L1
relativo contra uma referência float foi **2.40e-2 .. 3.18e-2 antes, 1.25e-2 .. 1.30e-2 depois**. Isso é o mesmo dos downs
IQ4_NL e Q2_0 (1.29e-2 .. 1.31e-2) nas mesmas seeds. Os lançamentos de especialistas agrupados continuam iguais bit a
bit aos lançamentos anteriores à janela. A verificação `native_grouped_parity` adiciona Q5_0 a todo formato gate/up e usa
as formas Flash-Next que esses arquivos usam, com 0 falhas. Versão do engine: 0.1.45.7; o setup recusa um engine mais antigo para
estes sete conjuntos antes de baixar qualquer coisa.

Os cinco arquivos fixados, seus bytes e todo outro formato não mudam. Só os pesos de *especialista* Q5_0 leem este
kernel (o caminho Q5_0 denso é `native_mmvq`, intacto). O erro do especialista inteiro deles contra a referência
float passa de 2.4-3.2e-2 para 1.25-1.30e-2.

### Portes selecionados de desempenho do upstream v0.2.11 — 5 de novembro de 2026

Esta atualização porta otimizações upstream de forma seletiva; ela não é uma integração completa
do v0.2.11. Ela adiciona kernels de especialistas trellis mais rápidos, buffers de rascunho de prompt
menores e menos alocações do leitor de disco de PLE. Ela também traz uma correção de zero-spin de
CPU/CUDA no Linux. A versão do engine continua 0.1.45.6.

**Medido na máquina dual-V100 deste fork:** duas placas V100-PCIE-16GB a
175 W (GPU0 Gen3 x4, GPU1 x16), Ryzen 5 3600, e Gyro-M abliterated Huihui
TQ2_0. O perfil usou split 16, duas janelas de pipeline, MTP 4, KV int8
e contexto de 524,288 tokens com YaRN 2.

| Tokens do prompt | Tempo do prompt de controle | Group scratch + WMMA desligado |
| ---: | ---: | ---: |
| 2,048 | 2,360.8 ms | 2,214.3 ms |
| 8,192 | 6,932.7 ms | 6,541.0 ms |
| 32,768 | 18,008.6 ms | 17,636.7 ms |

Estes são valores médios de execuções frias e intercaladas A-B-A-B, quatro requisições por tamanho
e por braço. Os dois braços já incluíam os portes trellis, HC e PLE. Isto não é um ganho medido em
todo tamanho de contexto nem uma garantia de velocidade da requisição inteira.

A combinação opt-in medida é `STRATA_TQ_SUB8=1 STRATA_EXPERT_PAIR=0
STRATA_PF_HC_REGION=1 STRATA_PF_GROUP_SCRATCH=1 STRATA_PF_WMMA=0`.
O group scratch pode mudar o arredondamento de ponto flutuante. A mudança de alocação de PLE
não precisa de flag. A correção de zero-spin se aplica em `STRATA_POOL_SPIN_US=0`; ela não
estabelece economia no ajuste diário de 100.

Veja o [relatório do porte para V100](docs/NVIDIA_V100.md#selected-upstream-0211-ports-gyro-m-2026-11-05)
para medições individuais, guardas de opção, verificações numéricas, limites de build
e candidatos testados que não foram mantidos. Nenhum padrão do serviço muda.

### Sincronização seletiva com o upstream — 19 de outubro de 2026

Este branch importa mudanças compartilhadas do engine CUDA e do servidor do commit
`3d1fb480` do upstream. Ele mantém os kernels Volta do fork, o crescimento de K/V no pipeline
e as mudanças do cache de conversas. Ele exclui pacotes de desktop não relacionados e
atualizações separadas dos backends Metal, AMD e Intel. Veja o
[escopo da sincronização](improvements/RELEASE-2026-10-19-V100.md).
Use CUDA 12.8 com `sm_70`. Uma opção importada não estabelece um ganho de
desempenho na V100.

### Integração do upstream v0.1.39

Esta atualização integra o [Strata v0.1.39](https://github.com/Niko1221/Strata/releases/tag/v0.1.39), incluindo as releases intermediárias. Ela mantém os especialistas em tensor cores para V100, as mudanças de atenção, a divisão assimétrica de GPU e o cache de conversas em disco do fork.

- **Requisições paralelas:** defina `"parallel": 2` na configuração do modelo ou use `./setup.sh --parallel 2`. O setup grava esse padrão para uma configuração Volta nova e preserva uma configuração já existente. O engine pode decodificar conversas separadas juntas. Requisições extras esperam por um slot livre. Os slots usam memória de GPU adicional e podem reduzir a velocidade de uma requisição única. Se duas sessões de slot não couberem, o engine avisa e atende uma requisição por vez, com a configuração inalterada. Veja [slots de lote](docs/BATCHING.md) para ajustes de exatidão, custos de memória e limites.
- **APIs e interface:** a atualização adiciona a Responses API da OpenAI, ajustes de modelo, monitoramento do cache de conversas e métricas por slot. Veja [detalhes da API](docs/DETAILS.md#using-it).
- **Desempenho e memória:** a atualização inclui menos lançamentos na passagem de verificação, anéis de prompt dimensionados em bytes, páginas grandes no Linux e correções de orçamento de RAM. Ela também adiciona opções multi-GPU. Um `--layer-split` explícito agora carrega automaticamente só os pesos densos das camadas de cada estágio (a flag upstream `--trim-stage-weights` e `STRATA_STAGE_TRIM` não existem mais). A VRAM liberada vai para o cache de especialistas. Uma otimização para GPUs mais novas não implica ganho de velocidade na V100. Caminhos opcionais que mudam o arredondamento precisam ser medidos antes do uso.
- **Outras plataformas:** os caminhos upstream Intel Arc, CPU mais antiga e AMD adicional continuam disponíveis. Esta máquina não fornece validação de hardware para essas plataformas.

A V100 exige CUDA 12.x e uma build que inclua `sm_70`. Não instale o engine CUDA 13 em uma V100. A configuração paralela explícita é recomendada para instalações existentes; uma atualização não sobrescreve a configuração do modelo.

**Medido nas duas placas V100 (4 de outubro de 2026, build final).** Uma execução antes/depois
controlada da build integrada contra o runtime instalado antes, uma requisição por vez
(sem slots de lote), frio, Qwen3.8-Flash-Next Q2_0 com KV int8 no contexto de 524,288 tokens,
a divisão de camadas 20/28 e o MTP (`--spec 8`) inalterados; duas requisições novas por tamanho, medianas:

| Prompt alvo | Antes (tokens/s) | Integrado (tokens/s) | Mudança |
| ---: | ---: | ---: | ---: |
| 2K | 701.41 | 718.5 | +2.44 % |
| 8K | 1,402.82 | 1,458.0 | +3.93 % |
| 32K | 2,262.4 | 2,283.7 | +0.94 % |

Cada valor é a mediana de duas requisições novas (`reused=0`). Os ganhos são pequenos; duas repetições
por tamanho não são uma afirmação estatística. Todos os seis primeiros tokens pareados coincidiram com os da linha de base.
As respostas completas não são declaradas idênticas. O nível adaptativo de especialistas mantém uma mistura
diferente de especialistas na GPU e na CPU para a build integrada, e os dois caminhos arredondam
de forma diferente. Nenhuma mudança de decode ou de qualidade é declarada. Estes são os valores medidos
da build final (SHA-256 do engine `e9ad337a…`), relatados no
[relatório completo](benchmarks/v100-q2_0-upstream-v0139-2026-10-04.md).

Slots paralelos no contexto de 524,288 tokens dependem da divisão e da memória que sobra no
último estágio. O fallback medido (2 solicitados, uma requisição por vez atendida, configuração
intacta) está descrito em [docs/BATCHING.md](docs/BATCHING.md).

### Desempenho do pipeline dual-V100 — 15 de outubro de 2026

O perfil de serviço medido ativa `--pipeline-windows 2` com a ordem de GPUs
`0,1` e divisão por camadas `16`. Ele usa duas placas V100 de 16 GiB a 175 W,
um Ryzen 5 3600, o modelo nativo IQ3_S de formato misto, KV int8 e MTP
com `--spec 8`. A capacidade de contexto configurada continua 524,288 tokens.
O maior prompt medido é de 32,768 tokens.

**Segurança em respostas longas:** use uma build com a
[correção de crescimento de KV no pipeline](benchmarks/v100-pipeline-kv-growth-fix-2026-10-16.md)
ao combinar este perfil com `STRATA_KV_GROW=1`. O laço de pipeline anterior
podia acessar células de KV não mapeadas durante uma resposta longa. A correção
esvazia o trabalho especulativo antes de crescer o mapeamento e depois retoma
o pipeline. As medições de velocidade com saída curta abaixo não exercitaram
esse limite.

#### Prefill de prompt novo até 32K

Cada valor é a mediana de três requisições com tokens exatos, sem reuso de prompt
e com limite de saída de 16 tokens. O tempo de prefill exclui a geração da resposta.
O throughput é o número de tokens do prompt dividido pelo tempo mediano de prefill.

| Tokens do prompt | Tempo de prefill anterior | Tempo de prefill com pipeline | Tokens/s anteriores | Tokens/s com pipeline | Mudança de throughput |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 2,048 | 2.089 s | 2.090 s | 980.6 | 979.8 | −0.09% |
| 8,192 | 5.487 s | 5.494 s | 1,493.0 | 1,491.2 | −0.13% |
| 16,384 | 8.828 s | 8.857 s | 1,855.9 | 1,849.8 | −0.33% |
| 32,768 | 15.668 s | 15.677 s | 2,091.4 | 2,090.2 | −0.06% |

O prefill de prompt novo fica aproximadamente igual. Essas diferenças pequenas
não estabelecem uma regressão significativa.

#### Decode de conversa

A comparação HTTP separada usa um prompt inicial de 8,828 tokens e seis
turnos de continuação. O contexto chega a aproximadamente 11.9K tokens. Cada valor
é a mediana de quatro requisições, com a ordem dos braços invertida em rodadas alternadas
e limite de saída de 256 tokens. Os novos tokens de prompt são os tokens processados
naquele turno, não o tamanho total do contexto.

| Turno | Novos tokens do prompt | Tokens/s de decode anteriores | Tokens/s de decode com pipeline | Mudança no decode |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 121 | 70.20 | 79.45 | +13.18% |
| 2 | 321 | 70.60 | 82.05 | +16.22% |
| 3 | 1,086 | 76.60 | 79.00 | +3.13% |
| 4 | 128 | 73.00 | 79.90 | +9.45% |
| 5 | 313 | 74.35 | 86.55 | +16.41% |
| 6 | 1,089 | 74.15 | 80.75 | +8.90% |

A mediana das seis razões de throughput de decode é **+11.31%**.
Isto não é um ganho de decode medido em todo tamanho de contexto até 32K.
Os comprimentos gerados e o trabalho de MTP diferem com o compartilhamento automático de CPU
e o roteamento residente. A latência da requisição inteira é mista; uma mediana HTTP de turno curto
piora. Não interprete o ganho de decode como um ganho garantido de velocidade da requisição
inteira nem como uma comparação de saídas idênticas.

Um controle separado com ajustes fixos coincide em todas as três sequências completas de saída
de 16 tokens nos tamanhos de prompt 2K, 8K e 32K. Ele não prova saída idêntica
para toda requisição de produção.

Veja o [relatório de avaliação](benchmarks/v100-upstream-pr-evaluation-2026-10-15.md)
para tempos exatos, proveniência da build, tentativas rejeitadas e comandos de
reprodução. Veja o [guia do perfil de serviço](docs/NVIDIA_V100.md#dual-pcie-pipeline-service-profile)
antes de ativar a opção em outra instalação.

### Cache KV TurboQuant — 25 de outubro de 2026

O TurboQuant é um formato de cache KV opt-in. Ele rotaciona cada cabeça de 256 valores com uma transformada de
Hadamard e depois quantiza os valores rotacionados contra um codebook Lloyd-Max. Cada cabeça mantém uma norma FP16.
As larguras são 3, 4, 6 e 8 bits por valor. O ajuste é apenas de erro quadrático mínimo; este corte não tem resíduo QJL.

Use `--kv tq6` (ou `tq3`, `tq4`, `tq8`). O cache é somente residente: ele fica na VRAM, não há cópia
no host, e `--kv-resident` é recusado. A camada de rascunho MTP mantém o cache INT8 dela. O setup não oferece esses
valores. Nada é padrão, e o serviço diário continua em int8.

Na capacidade configurada de 524,288 tokens, com as 12 camadas de atenção do modelo, o cache KV é:

| `--kv` | Bytes por célula, uma camada | KV em capacidade total, 12 camadas | Mudança em relação ao int8 |
| --- | ---: | ---: | ---: |
| int8 | 1,056 | 6.1875 GiB | — |
| tq8 | 1,032 | 6.046875 GiB | −0.14 GiB |
| tq6 | 776 | 4.546875 GiB | −1.64 GiB |
| tq4 | 520 | 3.046875 GiB | −3.14 GiB |
| tq3 | 392 | 2.296875 GiB | −3.89 GiB |

A economia principal do tq6 é 1.64 GiB em capacidade total. Estas são capacidades calculadas, não reduções
medidas no `nvidia-smi`: o cache de especialistas preenche a VRAM liberada, então a memória total da GPU fica perto de 15,600 MiB.

O leitor de prompt da V100 é um segundo opt-in: `STRATA_PROMPT_ATTN_TQ=1` em sm_70. Ele decodifica tiles comprimidos
para o caminho de prompt; sem definir ou com `0`, o fallback de atenção de decode continua. Ele não muda os bytes de KV.

A medição usou duas placas Tesla V100 de 16 GB a 175 W com IQ3_S. O perfil usou `--spec 4`, `--layer-split 16`
e `--pipeline-windows 2`, com limite de saída frio de 2,048 tokens e uma execução por tamanho por braço. O braço tq6
tinha o leitor de prompt ligado. Com 32,768 tokens de prompt, o tq6 fez prefill a 1,687 tokens/s contra 2,117 do int8,
e decode a 120.2 tokens/s contra 125.5. Com o leitor ligado, o tempo total da requisição de 51,746 tokens do tq6
caiu de 40.3–40.4 s para 31.7–31.9 s. Ele ainda ficou mais lento que os 25.8 s do int8. Essas medições não mostram
ganho de velocidade. Um teste de agulha encontrou 15 de 15 células nos dois braços até 122,289 tokens de prompt; isso não é um
resultado geral de qualidade.

Veja o [relatório completo](benchmarks/v100-turboquant-2026-10-25.md) e
[os detalhes](docs/DETAILS.md#tesla-v100-fork-benchmark).

### Pair trellis Gyro-L em duas V100 — 31 de outubro de 2026

Este branch lê o `Qwen3.8-Flash-Next-Gyro-L-TQ3_0.gguf` da agentionai (108,323,559,808 bytes, revisão
`05892758a67efff0895d23c6df10a6fdb6be8cf4` no Hugging Face). Os especialistas roteados dele são armazenados como os tipos pair trellis
TQP5, TQP6 e TQP7 (tipos ggml 147 a 149) em uma base rotacionada por Hadamard. A etapa de pack
(`tools/iq_pack.py --compat-bf16`) e os kernels são descritos em [docs/GYRO.md](docs/GYRO.md). CPU e CUDA
têm suporte; Metal e SYCL não têm kernels pair-trellis. Isto precisa do engine 0.1.45.6 ou mais novo.

Dois defeitos do engine precisaram ser corrigidos primeiro. Os kernels de especialistas em si já eram exatos bit a bit contra o
ggml dos autores:

- **A janela de decode perdia a rotação.** Para um pack dobrado, a janela rotacionava a entrada dos especialistas e depois
  sobrescrevia essa cópia rotacionada com a não rotacionada antes de os especialistas a lerem. Assim, as projeções gate/up
  rodavam na base errada em todo token decodificado. A saída de tiro único parecia plausível e depois entrava em loop; o
  servidor estava errado desde o primeiro token. Agora a janela mantém uma marca de "imagem escrita" por camada.
- **O auxiliar de prompt da divisão por camadas** (opt-in) rotacionava apenas a saída SwiGLU dele. Ele também escrevia as
  linhas de um grupo em um buffer organizado para os tiles WMMA do estágio. Agora ele rotaciona a entrada que recebe e é
  dono do H dele.

Medido no par deste fork: duas Tesla V100-PCIE-16GB (GPU 0 em PCIe Gen3 x4, GPU 1 em x16) a 175 W, um
Ryzen 5 3600, divisão por camadas 14, `--spec 4 --spec-min-p 0.70`, KV int8 com crescimento de KV, `--pipeline-windows 2`,
contexto de 262,144 tokens, o próprio ambiente do serviço, repetições intercaladas:

| Tokens do prompt | Prefill (tokens/s) |
| ---: | ---: |
| 2,328 | 632 |
| 8,199 | 1,036-1,084 |
| 16,371 | 1,344-1,411 |
| 32,679 | 1,597-1,642 |

As respostas decodificam a 68 a 75 tokens/s de 2K a 32K de contexto (medianas de respostas repetidas de 520 tokens); o ponto
de divisão não muda isso além do ruído entre execuções. Uma leitura é limitada pelos especialistas que não estão na VRAM.
A placa x16 transmite 15.01 GiB por PCIe para uma leitura de 2.3K tokens, com o perfil em divisão por camadas 16.
Os dois estágios rodam um depois do outro. Por isso, a placa x4 fica ociosa cerca de 700 ms de cada leitura.

O pack atende pela API compatível com OpenAI: "Paris" para a pergunta da capital, um turno de acompanhamento que
reutiliza 86 tokens de prompt, uma requisição de código, uma chamada de ferramenta `get_weather` e uma resposta correta a partir de um resultado de ferramenta,
e uma pergunta sobre uma passagem de 442 tokens respondida a partir da passagem.

**Divisão por camadas 14 em vez de 16.** Quatorze camadas na placa x4 e 34 na placa x16 leem 7.9% mais rápido com
2.3K tokens (632 contra 586 tokens/s). A leitura é 6% mais rápida com 8.2K, igual com 16.3K e 1.5% mais lenta com 32.7K.
15 e 13 ficam entre esses valores, e 20 é 23% mais lento com 2.3K e 13% com 8.2K. Uma leitura é o que um turno agêntico consome
de tempo, então este perfil usa 14.

**O que foi medido e não compensou neste par:** `--spec 6` e `--spec 8`, `--spec-min-p 0.50` e
`0.85`, um `STRATA_PREFILL_CPU_SHARE` fixo, `STRATA_PREFILL_CPU_SHARE_MAX` acima de 6,144, `--prefill-pipe`,
`--max-context 131072`, `STRATA_HC_Q8=1` e um `--vram-reserve-mib` maior. O auxiliar de prompt em si é
mais lento aqui (484 contra 583 tokens/s com 2.3K). Ele só pode usar os especialistas do primeiro estágio. Neste par,
esse estágio é a placa x4, que transmite 1.77 GiB dos 16.8 GiB que uma leitura move. `--mtp` e o padrão
`STRATA_ROUTE_TAIL_SKIP=7` vale a pena manter: o decode cai para 35.5 e 48 tokens/s sem eles.

## Benchmark da V100

A tabela abaixo registra uma execução de placa única em uma Tesla V100-PCIE-16GB usando PCIe Gen3 x16. A máquina de teste também tem uma segunda V100 em PCIe Gen3 x4; ela não foi usada nesta execução. Resultados de uma configuração de duas placas ou de uma máquina com dois links de largura total podem diferir e podem ser melhores. É uma medição reproduzível, não uma promessa de velocidade em outros sistemas.

| Prompt alvo | Tokens do prompt (mediana) | Prefill (tokens/s) | Decode (tokens/s) | Temperatura máxima da GPU |
| ---: | ---: | ---: | ---: | ---: |
| 1K | 1,036 | 398.6 | 52.0 | 56 °C |
| 4K | 4,111 | 1,214.0 | 50.8 | 61 °C |
| 8K | 8,185 | 1,442.4 | 48.1 | 63 °C |
| 16K | 16,367 | 1,490.7 | 49.8 | 67 °C |
| 32K | 32,765 | 1,495.7 | 47.3 | 72 °C |
| 64K | 65,533 | 1,484.3 | 43.0 | 81 °C |
| 128K | 131,063 | 1,033.0 † | 40.9 | 84 °C |
| 256K | 256,080 | 617.8 † | 36.8 | 84 °C |

**Configuração do teste:** Qwen3.8-Flash-Next Q2_0; apenas uma V100 (GPU1, PCIe Gen3 x16); Ryzen 5 3600; 48 GB DDR4-3200; CUDA 12.8; contexto de 262,144 tokens; cache KV int8; prefill automático; MTP com `--spec 8` e piso de rascunho 0.70; variante de especialistas pareados; reserva de visão de 700 MiB. Cada valor é a mediana de três requisições novas, sem cache (`reused=0`), com exatamente 256 tokens de saída. Os tempos vêm do endpoint `/metrics` do engine.

Antes de cada requisição, a placa com refrigeração passiva ficou ociosa por pelo menos 120 segundos. Ela atingiu 55 °C ou menos sem redução térmica de software por 15 segundos. As temperaturas iniciais foram 51–55 °C. Todas as requisições estão incluídas. Os prompts de 128K e 256K atingiram 84 °C e sofreram redução térmica de software durante o processamento do prompt, apesar do início resfriado. Execuções longas podem limitar a velocidade. † indica redução de velocidade durante o prompt.

Medido em 2026-10-02. Engine: versão do fork 0.1.31, commit da build `78417ea` (SHA-256 `512b1f25d60479a2ddb66fcf1ddca5407963e45378a3c8a463413ad159edae17`). O benchmark foi registrado antes da integração posterior do upstream v0.1.36 e dos PRs de atenção para V100; ele não é uma medição da build atual do `main`. A tabela histórica é mantida com a proveniência original dela. Não a interprete como comparação controlada contra tabelas anteriores ou builds atuais.

Dados brutos: [`summary.json`](bench/results/2026-10-02-v100-single-cooled/summary.json), [`protocol.json`](bench/results/2026-10-02-v100-single-cooled/protocol.json), [`matrix.json`](bench/results/2026-10-02-v100-single-cooled/matrix.json), [`completed-requests.json`](bench/results/2026-10-02-v100-single-cooled/completed-requests.json) e [`telemetria da GPU`](bench/results/2026-10-02-v100-single-cooled/gpu.csv). Veja o [método completo e o histórico](docs/DETAILS.md#tesla-v100-fork-benchmark), as [instruções de benchmark](benchmarks/README.md) e os [relatórios de desempenho da V100](benchmarks/).

Pull requests recentes para V100 incluem mudanças medidas em nível de kernel e relatam explicitamente quando ganhos de ponta a ponta não são estabelecidos. Por exemplo, o porte de atenção de prompt do PR #600 do upstream relatou 27.9% menos tempo de kernel de atenção int8 na forma medida dele. As mudanças de prefill do modelo ficaram entre +1.3% e +3.0% nas execuções relatadas; ele não estabeleceu um ganho confiável de decode. Leia o [relatório completo](benchmarks/v100-q2_0-pr600-2026-10-03.md) antes de comparar resultados.

## Trabalho de desempenho na V100

O trabalho na V100 abrange o caminho de prompt, o caminho de decode, o uso de memória, as leituras de armazenamento e a execução multi-GPU. As mudanças abaixo estão no histórico integrado do fork; cada link inclui o próprio método de teste, detalhes de hardware e limitações. Melhorias de tempo de kernel não produzem necessariamente o mesmo ganho percentual no throughput do modelo completo.

### Processamento de prompt

- **GEMMs em tensor cores de BF16 para FP16 no Volta:** o Volta não tem suporte nativo a GEMM BF16. O fork converte pesos BF16 exatamente para FP16 e usa tensor cores da V100 para os GEMMs de prompt em vez do fallback escalar. As imagens de ativação de prompt FP16 associadas usam a representação de bits correta.
- **Leituras da tabela PLE mais rápidas:** troque as leituras PLE POSIX por operações `io_uring` O_DIRECT em fila, com fallback síncrono. Uma coleta PLE nova de 8K medida caiu de cerca de 6.9 segundos para 7 milissegundos.
- **Prefill MMQ Q2_0:** ative o caminho quantizado matriz-matriz Q2_0 do llama.cpp já existente para o tipo interno de especialista do Strata. Um prompt medido de 14,486 tokens melhorou 6.5% no throughput de prefill (926.1 para 986.2–986.9 tokens/s).
- **Pontuar apenas blocos QSA vivos:** evite pontuar blocos além do intervalo vivo do prompt atual. Em um prompt medido de 15,423 tokens, o tempo de seleção QSA caiu 91.1%. O throughput de prefill de ponta a ponta subiu de 961.0 para 1,111.3–1,112.9 tokens/s nos testes relatados.
- **Atenção de prompt em tensor cores para Volta:** use `mma.m8n8k4` na V100 em vez do fallback FP32. Um PR anterior mediu cerca de 1.6–1.9× de velocidade no kernel de atenção e maior throughput de prompt nos testes dele. O porte posterior do PR #600 do upstream reduziu o tempo de kernel de atenção int8 em 27.9% em uma forma medida. Os deltas de prefill de ponta a ponta relatados foram menores (+1.3% a +3.0% nos testes de 4K–32K dele).

### Decode e execução multi-GPU

- **Janela de verificação em lote:** execute várias posições especulativas de decode juntas, reduzindo a sobrecarga de lançamento por token. Testes históricos pareados relataram geração de saída mais rápida, mas os resultados variam com o prompt e as condições de execução.
- **Pontuação de blocos QSA em tiles:** reutilize os dados de chave selecionados entre tiles de consulta em vez de lê-los repetidamente. O kernel medido de 256 consultas / 65K blocos caiu de 3.5 ms para 1.3 ms; a aritmética por pontuação continuou idêntica bit a bit.
- **GR Volta fundido e append de KV em lote:** combine trabalho entre posições de verificação e anexe dados de KV em menos lançamentos. Na configuração dual-V100 testada, o decode aumentou 5.6–17.3% em prompts de 4K–32K, enquanto o prefill ficou aproximadamente igual.
- **Ajuste assimétrico dual-V100:** melhore o tratamento de linhas no host, o despacho do plano de dispositivos e a propriedade de memória para uma configuração de duas placas. Uma configuração instalada anterior mostrou ganhos de prefill de 13.3–18.7% em 4K–32K. No entanto, os resultados de decode foram mistos, e as duas GPUs reduziram a velocidade por temperatura durante aquela execução. Esses valores não isolam cada mudança de código.
- **Reduzir sobrecarga de pesos e despacho:** carregue apenas as camadas de cada placa ao usar um split explícito. Evite quantização desnecessária de ativações na CPU e barreiras de despacho vazias, e opcionalmente execute kernels de especialistas de linhas pareadas. A alocação densa nativa medida da GPU0 caiu de 1,376.20 MiB para 541.27 MiB. O modo de linhas pareadas é opt-in; os resultados relatados do modelo completo não estabeleceram melhoria de decode.

### Kernels de atenção e tráfego de memória

- **Atenção de decode (porte do PR #540 do upstream):** reduza trocas de pontuações e tráfego de memória compartilhada, preservando a ordem de acumulação. O tempo de kernel medido caiu de 52.90 para 31.13 microssegundos na GPU0 para M=1, e de 180.42 para 111.85 microssegundos para M=8. As mudanças relatadas de decode no modelo completo foram pequenas; o prefill ficou efetivamente igual.
- **Atenção de prompt (porte do PR #600 do upstream):** use a implementação em tensor cores do Volta com quatro computações MMA independentes. O PR relata verificações de correção e medições em nível de kernel e de modelo. A redução de 27.9% no kernel não deve ser lida como aumento de 27.9% na velocidade do modelo completo.
- **Coleta de KV e recorrência GDN (portes do PR #627 do upstream):** use cargas de KV int8 alinhadas mais largas, com fallback seguro para a largura original. Use múltiplos acumuladores para a recorrência de prompt da V100. Os tempos de kernel medidos melhoraram cerca de 10% para uma coleta de KV de 32K e cerca de 7% para uma recorrência de 4,096 tokens. O prefill pareado do modelo diferiu menos de 0.2%; nenhum ganho confiável de decode de ponta a ponta foi estabelecido.

### Suporte de build e evidências

O fork também corrigiu a configuração e a ligação CUDA para `sm_70` e manteve a admissão de dispositivos V100 através das integrações upstream. Ele também adaptou mudanças upstream quando os requisitos de arquitetura delas excluem o Volta. Essas correções de compatibilidade mantêm os caminhos otimizados compiláveis; elas não são, por si só, afirmações de desempenho.

Para as evidências completas e o escopo de implementação, veja o [PR #1](https://github.com/jmnargi/Strata-V100/pull/1), o [#2](https://github.com/jmnargi/Strata-V100/pull/2), o [#5](https://github.com/jmnargi/Strata-V100/pull/5), o [#6](https://github.com/jmnargi/Strata-V100/pull/6), o [#11](https://github.com/jmnargi/Strata-V100/pull/11), o [#12](https://github.com/jmnargi/Strata-V100/pull/12), o [#15](https://github.com/jmnargi/Strata-V100/pull/15), o [#16](https://github.com/jmnargi/Strata-V100/pull/16) e o [#17](https://github.com/jmnargi/Strata-V100/pull/17). Os resultados usam datas, builds, prompts, posicionamentos de GPU e protocolos diferentes. Eles são medições históricas, não uma única comparação controlada antes/depois nem uma garantia de ganhos em todo sistema V100.

## Outro hardware

- **Placas de vídeo mais antigas** (Tesla P40 / V100, GTX 10, Radeon VII / MI50, RX 6700 XT, RX 5500 XT): [Older GPUs](docs/OLDER_GPUS.md).
- **Intel Arc**, compilado a partir do código-fonte no Linux: [Intel Arc](docs/INTEL_ARC.md).
- **AMD Ryzen AI Max (Strix Halo)**, compilado a partir do código-fonte no Linux: [Strix Halo](docs/STRIX_HALO.md).
- **Processadores mais antigos sem AVX2**: funcionam, mas devagar. [Older CPUs](docs/INSTALL.md#older-cpus-experimental).

## Instalar e iniciar

Este fork acompanha a experiência de instalação do upstream. Para requisitos de plataforma completos e atuais, escolhas de modelo e opções, veja o [guia de instalação](docs/INSTALL.md), o [guia de modelos](docs/MODELS.md) e o [guia de setup do upstream](docs/AI_SETUP.md).

- **Windows:** baixe ou clone este repositório e execute `START-HERE.bat`.
- **Linux:** clone este repositório e execute `./setup.sh`.
- Siga as instruções para selecionar um modelo e um tamanho de contexto. O setup baixa os dados do modelo e inicia o serviço local.
- Abra o endereço impresso pelo setup (normalmente `http://127.0.0.1:8080`).

Para uma Tesla V100, use o engine CUDA 12 com suporte a `sm_70`. O setup seleciona o engine para placas mais antigas no caso do Volta; no Linux, ele compila localmente. O arquivo CUDA 12 do Windows do upstream inclui código Volta, mas não foi testado nesta máquina Linux. Veja o [guia para placas mais antigas](docs/OLDER_GPUS.md) para os requisitos de toolkit e driver. Para Docker, siga as [instruções de Docker](docs/INSTALL.md) e compile para a arquitetura de GPU de destino.

## Usar o Strata

- **Aplicativo web:** use a URL local impressa pelo setup para conversar e ver o monitor ao vivo.
- **API compatível com OpenAI:** defina a URL base do seu aplicativo como `http://127.0.0.1:8080/v1`.
- **API compatível com Anthropic:** use `http://127.0.0.1:8080/v1/messages`.
- **Responses API da OpenAI:** use `http://127.0.0.1:8080/v1/responses`. Este endpoint sem estado suporta chamadas de ferramenta no estilo Codex; ele não armazena `previous_response_id`.
- **Imagens, MCP, multi-GPU e configuração:** veja os [detalhes](docs/DETAILS.md), o [servidor MCP](docs/MCP_SERVER.md) e o [guia multi-GPU](docs/MULTI_GPU.md).

O servidor normalmente escuta em localhost. Se você o expuser a outras máquinas, configure uma chave de API e use uma rede confiável. Não publique segredos ou configuração privada em issues, resultados de benchmark ou pull requests.

### Cache de conversas em disco (opt-in)

O estado da conversa tem três níveis. O L1 é a sessão ativa e os checkpoints de prompt residentes dela. O L2 é o cache opcional na RAM do host (`--conversation-cache-mib`). O L3 é um armazenamento em disco opcional que mantém conversas salvas e grupos de checkpoints entre reinicializações. Camadas KV transmitidas em fluxo usam os pools autoritativos delas no host quando o engine captura o estado.

Para usar o armazenamento em disco, adicione `--conversation-cache-disk strata-conversations --conversation-cache-disk-gib 25` aos argumentos do engine. O caminho e um orçamento positivo em GiB são obrigatórios juntos. O armazenamento fica desligado por padrão, precisa de `--serve`, funciona com `--conversation-cache-mib 0` e suporta `--layer-split` em várias GPUs. Opcional: `--conversation-cache-disk-slots N` limita o número de registros, e `--conversation-cache-disk-min-free-mib N` mantém espaço livre no sistema de arquivos.

O engine preserva um checkpoint ativo antes de a retenção removê-lo ou de uma requisição sobrescrever o branch dele. Cada checkpoint salvo inclui o estado GDN, o histórico PLE, o estado do indexador QSA e o backing de KV necessário. Apenas os dados de KV de tokens não bastam para retomar este modelo.

Checkpoints relacionados compartilham uma imagem de KV por grupo de arquivo. O engine move e estende essa imagem em vez de copiar o prefixo completo para cada checkpoint. Ele copia apenas páginas de KV novas ou alteradas quando o backing ainda é válido. Um grupo para de crescer antes de atingir o limite de metadados do disco. Um grupo posterior tem o próprio backing. O engine não combina um checkpoint salvo com bytes de KV de outro cálculo do prefixo.

Os grupos de arquivo usam os mesmos limites de bytes e de slots do L2 que as conversas salvas. Vítimas de capacidade do L2 passam para o L3 quando o nível de disco está ativado. Uma admissão bem-sucedida no L2 não grava a imagem no disco também. Se uma imagem não couber no L2, o engine tenta armazená-la no L3. Os limites de RAM física e de espaço livre em disco continuam valendo. O ajuste de mínimo de tokens também vale para os arquivos de checkpoints.

Preservar um checkpoint é uma otimização, não um requisito. Se o piso de RAM do host ou o orçamento de disco recusar a imagem, o engine descarta o checkpoint, registra o descarte e continua atendendo. Ele não para o servidor. A sessão continua correta; apenas o reuso daquele checkpoint é perdido.

A consulta seleciona o prefixo compatível mais longo entre a sessão ativa, o L2, o L3 e slots de lote ociosos. Tokens, identidade da imagem e modo do vetor de controle precisam coincidir. Para uma edição em uma mensagem anterior, o reuso termina em um checkpoint salvo antes do primeiro token alterado. O engine precisa processar o sufixo de novo; ele não pode retomar em um token correspondente arbitrário sem o estado recorrente naquela posição. Aumentar `--prompt-cache` não substitui a preservação de checkpoints.

Um arquivo no L2 continua disponível depois da restauração. Uma imagem válida do L3 é promovida para o L2 quando o orçamento de bytes e o piso de RAM física permitem. Apenas um subconjunto limitado dos checkpoints dela fica residente na sessão ativa. Restaurar um branch não remove outros checkpoints do arquivo salvo.

O armazenamento em disco suporta cadeias de conversa somente de acréscimo. Uma cadeia em continuação pode acrescentar novos IDs de token, checkpoints, estado em execução e páginas de KV. Uma remoção do L2 grava uma imagem semente completa; ela não precisa de captura de GPU. Um retrocesso inicia um branch separado em vez de substituir a cadeia de origem. O formato de arquivo não muda para arquivos de checkpoints.

Os painéis de disco têm checksums e campos de versão. Uma semente usa um arquivo temporário e um rename. Um acréscimo incompleto é recuperado até o último painel completo. Registros inválidos ou incompatíveis são removidos. O armazenamento em disco remove as cadeias menos usadas recentemente para cumprir o orçamento de bytes, o limite de registros e o piso de espaço livre. Assim, nem o L2 nem o L3 fornecem histórico ilimitado. Se os dois níveis tiverem removido o checkpoint necessário, a requisição precisa processar o prefixo ausente de novo.

O engine captura uma raiz de prompt de sistema compartilhada uma vez no L2 quando a retenção de raiz está ativada. O L3 recebe a raiz na remoção do L2 ou em uma admissão falha no L2. Com o L2 desativado, um L3 ativado pode receber a raiz diretamente. `--conversation-cache-keep-root` e `--conversation-cache-disk-keep-root` controlam a retenção de raiz em cada nível. As variantes `--no-` delas desativam essa retenção. Raízes têm prioridade de remoção menor que imagens comuns; fixações explícitas de checkpoint de requisição continuam protegidas.

O nível de disco não adiciona execução concorrente. Veja [slots de lote](docs/BATCHING.md) para os custos de memória de requisições concorrentes. A captura precisa terminar antes de uma requisição sobrescrever os bytes de sessão necessários. A restauração é síncrona. Gravações de remoção do L2 usam snapshots do host e podem adicionar latência de I/O de disco. Dimensione os dois orçamentos de cache para as conversas e o histórico de checkpoints de que você precisa. Veja o [guia da V100](docs/NVIDIA_V100.md) para a verificação de retenção de checkpoints e o [relatório de benchmark de disco](benchmarks/v100-l3-conversation-cache-2026-10-03.md) para medições anteriores.

#### Resultado medido na V100

A configuração deste fork tem duas V100 com divisão por camadas. Retomar uma conversa de 33,725 tokens a partir de um cache NVMe de 25 GiB levou 1,614.6 ms para ler. A restauração levou 315.8 ms. A fase completa de prompt retomado, incluindo 22 tokens novos, levou 3,015.1 ms. Uma leitura fria do mesmo prefixo de 33,725 tokens levou 20,332.7 ms. Isso é um **ganho de velocidade de 6.74x** e uma **redução de 85.2% na latência do prompt** para a requisição retomada. O registro de 943.6 MiB restaurou o estado do modelo principal exato byte a byte e saída idêntica nas duas GPUs. A recuperação após reinício passou em paridade exata byte a byte, e o fallback de registro corrompido produziu saída correspondente. Veja [o relatório de benchmark](benchmarks/v100-l3-conversation-cache-2026-10-03.md).

Medido na máquina de duas V100 com divisão por camadas deste fork (Qwen3.8-Flash-Next Q2_0, KV int8): duas
conversas alternando por um cache de rascunho foram atendidas corretamente em 8 requisições com
`corruptions=0 evictions=0`. Um arquivo de cadeia por conversa cresceu de 238 MiB -> 594 MiB em
três parks de +12 tokens novos cada. Um armazenamento de snapshot completo teria gravado um arquivo novo de ~2.4 GiB
por park com 100k+ tokens. O primeiro park de um chat curto novo semeia a cadeia
(~118 MiB: uma página K/V por camada mais o estado em execução fixo do modelo). Depois disso, cada park
é o delta (+12 tokens ~ 118 MiB). O caminho de leitura e restauração não muda (~0.4 s para ler,
~0.04-0.08 s para restaurar).

### Visão preguiçosa (opt-in)

O codificador de imagens (`strata-vision`) ocupa cerca de 1.74 GiB de VRAM (1.43 GiB na primeira GPU, 0.31 GiB na segunda) enquanto roda. Com a visão preguiçosa ele não roda entre requisições de imagem. Essa VRAM fica com os caches de especialistas, então o trabalho só de texto mantém a contagem total de especialistas residentes.

Para ativar, adicione à configuração do modelo:

- `"vram_elastic": true` (a flag do engine `--vram-elastic`): os caches de especialistas são segmentados, então o engine pode devolver VRAM e retomá-la entre requisições.
- `"vision": {"lazy": true, "idle_s": 900}`: o codificador inicia na primeira requisição de imagem e para após `idle_s` segundos sem uma (900 = 15 minutos). O codificador precisa do modo de uma requisição por vez do engine. Por isso, não o combine com `"parallel"` (ou dê VRAM ao codificador de outra forma, por exemplo `POST /v1/vram`).

Como ele se comporta:

- Uma requisição de texto nunca inicia o codificador. Uma requisição de imagem primeiro pede ao engine que devolva a VRAM das GPUs. A reserva é uma por GPU, com `[1700, 600]` em uma divisão por camadas e `[1800]` em uma GPU; substitua com `"vram_mib"`. Depois ela inicia o codificador e codifica a imagem enquanto o engine está ocioso.
- Após `idle_s` segundos sem uma requisição de imagem, o codificador para, e os caches de especialistas crescem de volta ao tamanho total. As imagens codificadas ficam em cache no disco, então enviar a mesma imagem de novo não a codifica outra vez.
- Medido na divisão por camadas de duas V100 deste fork: os caches mantêm **14,005** especialistas residentes enquanto o codificador está ocioso (12,688 com o codificador residente). Uma requisição de imagem fria foi respondida em **5.9 s**, e os dois caches voltaram a 14,005 depois que o codificador foi descarregado. Veja os [detalhes](docs/DETAILS.md).

## Algo deu errado

Os problemas comuns e as correções deles estão em [TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md). Essa página cobre
o aplicativo de desktop, o comando `strata` e o instalador. A tabela completa de mensagens de erro está em
[DETAILS.md](docs/DETAILS.md#troubleshooting).

Se o servidor iniciar mas uma requisição falhar, execute `strata doctor`. Ele lê os componentes instalados e
imprime o que está faltando ou desatualizado; veja [TERMINAL.md](docs/TERMINAL.md#doctor).

Se uma resposta estiver lenta em vez de quebrada, leia [TUNING.md](docs/TUNING.md). Ele dá o ajuste que
importa para cada objetivo e o custo medido de cada um.

## Como contribuir

Contribuições são bem-vindas. Você não precisa de uma V100 para ajudar: documentação, testes, setup, comportamento do servidor, compatibilidade de API e melhorias para outros dispositivos compatíveis são úteis. Código e resultados de desempenho específicos da V100 se beneficiam de validação em hardware Volta real.

1. Verifique as [issues abertas](https://github.com/jmnargi/Strata-V100/issues) e os [pull requests](https://github.com/jmnargi/Strata-V100/pulls) para evitar trabalho duplicado.
2. Para recursos upstream amplos, verifique se a mudança pertence ao [repositório upstream do Niko](https://github.com/Niko1221/Strata). Este fork integra mudanças upstream periodicamente; correções e medições focadas na V100 podem ser propostas aqui.
3. Faça uma mudança focada, descreva o hardware e os passos exatos usados para testá-la, e inclua testes relevantes.
4. Para afirmações de desempenho, informe a linha de base e o candidato, as configurações completas de execução, a carga de trabalho, o número de repetições e os limites. Inclua dados brutos ou um comando reproduzível quando possível. Separe o tempo de kernel do throughput do modelo completo e não afirme um ganho que as medições não mostram.
5. Abra um pull request contra o branch `main` deste repositório. Explique se a mudança é específica da V100, uma integração upstream ou uma melhoria geral do Strata. Mantenha credenciais, configuração pessoal e arquivos locais não relacionados fora da mudança.

Veja [`benchmarks/README.md`](benchmarks/README.md) e [`docs/DETAILS.md`](docs/DETAILS.md) para a abordagem atual de medição. As contribuições continuam sujeitas à licença do projeto e às licenças dos componentes incluídos e dos arquivos de modelo.

## Histórico do projeto

Este fork desenvolve o suporte à V100 e o trabalho específico da V100 enquanto mantém o desenvolvimento upstream do Strata. Pull requests recentes de integração e de desempenho ilustram esse processo:

- [#14 — Integrar o upstream v0.1.36 preservando o suporte à V100](https://github.com/jmnargi/Strata-V100/pull/14)
- [#15 — Portar as mudanças medidas de coleta de KV e recorrência GDN para V100 do PR #627 do upstream](https://github.com/jmnargi/Strata-V100/pull/15)
- [#16 — Reduzir trocas na atenção de decode da V100 e o tráfego de memória compartilhada](https://github.com/jmnargi/Strata-V100/pull/16)
- [#17 — Portar o kernel de atenção de prompt Volta do PR #600 do upstream](https://github.com/jmnargi/Strata-V100/pull/17)
- [#88 — Ler os especialistas pair-trellis do Gyro-L (TQP5/6/7), corrigir a janela de decode dobrada e o auxiliar de prompt da divisão por camadas, e mover o perfil da V100 para `layer_split 14`](https://github.com/jmnargi/Strata-V100/pull/88)

Esses relatórios descrevem mudanças específicas testadas; eles não são uma afirmação genérica de desempenho mais rápido em modelos ou GPUs. Para mais detalhes, navegue pelo [histórico completo de pull requests](https://github.com/jmnargi/Strata-V100/pulls?q=is%3Apr+is%3Amerged) e pelo [histórico de commits](https://github.com/jmnargi/Strata-V100/commits/main).

## Créditos e licença

Este repositório é um fork do [Strata, de Niko1221](https://github.com/Niko1221/Strata). **Somos profundamente gratos ao Niko e a todos os contribuidores upstream.** O trabalho original deles é a razão de este fork, o suporte à V100 e este runtime local de modelo serem possíveis. Buscamos creditar e acompanhar o trabalho upstream ao trazer as mudanças dele periodicamente.

O Strata incorpora [llama.cpp / ggml](https://github.com/ggml-org/llama.cpp) e outros componentes de código aberto. O projeto está licenciado sob a [Licença MIT](LICENSE); componentes e arquivos de modelo podem ter termos separados. Veja as [notas de créditos e licença](docs/HOW_IT_WORKS.md#credits).
