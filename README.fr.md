# Strata-V100

**Un fork de [Strata](https://github.com/Niko1221/Strata) centré sur le V100 et maintenu par la communauté.** Ce fork garde le support de NVIDIA Volta (`sm_70`) fonctionnel. Il développe et mesure des changements de performance sur du matériel Tesla V100. Il fusionne aussi périodiquement les améliorations upstream pour les fonctionnalités et les corrections au-delà du V100.

Strata est un runtime local open source pour [Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next). Il combine le GPU, la mémoire système et le stockage pour faire tourner ce grand modèle sur un ordinateur personnel. Ce fork conserve le projet Strata au sens large. Les autres matériels NVIDIA et AMD pris en charge, l'interface web, l'API compatible OpenAI, les outils, l'entrée d'images et les fonctionnalités upstream restent importants eux aussi.

> **Un grand merci à Niko ([@Niko1221](https://github.com/Niko1221)) et aux contributeurs d'origine de Strata.** Ce projet s'appuie sur leur travail. Le runtime du modèle, l'architecture et les fonctionnalités de Strata, ainsi que le travail qui a rendu possible l'exécution locale de ce modèle, viennent tous du projet upstream. Le fork V100 n'existerait pas sans cette base. Merci de l'avoir rendue possible, et de continuer à développer Strata.

[Strata upstream](https://github.com/Niko1221/Strata) · [Tickets](https://github.com/jmnargi/Strata-V100/issues) · [Pull requests](https://github.com/jmnargi/Strata-V100/pulls) · [Discussions](https://github.com/jmnargi/Strata-V100/discussions)

**Langues :** [English](README.md) · [Deutsch](README.de.md) · [Español](README.es.md) · [Français](README.fr.md) · [日本語](README.ja.md) · [Português (Brasil)](README.pt-BR.md) · [简体中文](README.zh-CN.md)

## Sommaire

- [Modèles et quantifications qui tournent ici](#modèles-et-quantifications-qui-tournent-ici)
- [Ce sur quoi ce fork se concentre](#ce-sur-quoi-ce-fork-se-concentre)
- [Documentation](#documentation)
- [Notes de version](#notes-de-version)
- [Benchmark V100](#benchmark-v100)
- [Travaux de performance V100](#travaux-de-performance-v100)
- [Autre matériel](#autre-matériel)
- [Installation et démarrage](#installation-et-démarrage)
- [Utiliser Strata](#utiliser-strata)
- [Un problème est survenu](#un-problème-est-survenu)
- [Contribuer](#contribuer)
- [Historique du projet](#historique-du-projet)
- [Crédits et licence](#crédits-et-licence)

## Modèles et quantifications qui tournent ici

Strata fait tourner **Qwen3.8-Flash-Next** à partir de sept familles de versions quantifiées, plus un **Qwen3.6-35B-A3B** plus petit. Chaque ligne ci-dessous est prise en charge par le moteur et l'installateur de ce fork. La dernière colonne indique ce qui a réellement été exécuté de bout en bout sur la paire de V100 ici. Tailles, qualité et vitesses par taille : [docs/MODELS.md](docs/MODELS.md).

| Modèle / quantification | Experts routés | Origine des poids | Téléchargement | Exécution sur cette paire de V100 |
| --- | --- | --- | ---: | --- |
| **NVFP4 — `local-inference-lab` (QAD)** | NVFP4, groupe 16 ; attention et experts partagés MXFP8 ; table n-gram NVFP4 | [`local-inference-lab/Qwen3.8-Flash-Next-NVFP4`](https://huggingface.co/local-inference-lab/Qwen3.8-Flash-Next-NVFP4), converti sur votre PC ([NVFP4.md](docs/NVFP4.md#the-qad-distillation-local-inference-labqwen38-flash-next-nvfp4)) | 106 GB | oui — 62.3 tokens/s en decode |
| **NVFP4 — NVIDIA, entraîné pour FP4 (`FP4-Local`, QAD)** | NVFP4, groupe 16 (échelles sous la forme CSF de NVIDIA) | [`nvidia/Qwen3.8-Flash-Next-FP4-Local`](https://huggingface.co/nvidia/Qwen3.8-Flash-Next-FP4-Local), converti sur votre PC ([NVFP4.md](docs/NVFP4.md#nvidias-fp4-local-checkpoint)) | 102 GB (+110 GB pendant la préparation) | pris en charge ; non exécuté ici |
| **NVFP4 — version de NVIDIA** | NVFP4, groupe 16 | [`Maximilian228/Qwen3.8-Flash-Next-NVIDIA-NVFP4-Strata`](https://huggingface.co/Maximilian228/Qwen3.8-Flash-Next-NVIDIA-NVFP4-Strata) (pack prêt à l'emploi) | 128 GB | pris en charge ; non exécuté ici |
| **NVFP4 — version de Strata** | NVFP4, groupe 16 | [`Maximilian228/Qwen3.8-Flash-Next-NVFP4-GPTQ-Strata`](https://huggingface.co/Maximilian228/Qwen3.8-Flash-Next-NVFP4-GPTQ-Strata) (pack prêt à l'emploi, GPTQ) | 128 GB | pris en charge ; non exécuté ici |
| **Unsloth Dynamic v3.1 (1-4 bits)** | UD-IQ1_M, UD-Q2_0, UD-IQ2_XS, UD-Q2_K_XL, UD-IQ3_XXS, UD-Q3_K_XL, UD-IQ4_XS-v31 | [`unsloth/Qwen3.8-Flash-Next-GGUF`](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF), révision du 2026-11-04 | 68-94 GB | oui — UD-IQ4_XS-v31, 81 tokens/s |
| **Unsloth Dynamic (révision antérieure)** | UD-IQ4_XS, UD-Q4_K_XL, UD-Q5_K_XL, UD-Q6_K_XL, Q8_0 | [`unsloth/Qwen3.8-Flash-Next-GGUF`](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF) | 94-188 GB | pris en charge ; non exécuté ici |
| **i-quants et Q2_0 d'ISTA-DASLab (2-3.5 bits)** | Q2_0, IQ2_XS, IQ3_XXS, IQ3_S | [`ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF`](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF) | 66-84 GB | oui — IQ3_S, le profil quotidien antérieur |
| **Gyro (agentionai), treillis** | Gyro-S TQ1_0, Gyro-M TQ2_0 (treillis par paires) | [`agentionai/Qwen3.8-Flash-Next-Gyro-GGUF`](https://huggingface.co/agentionai/Qwen3.8-Flash-Next-Gyro-GGUF) ([GYRO.md](docs/GYRO.md)) | 58.5 / 92 GB | oui — Gyro-M, environ 70 tokens/s |
| **Coder (ISTA-DASLab), la moitié des experts** | IQ1_M, IQ2_S-IQ4_XS comme IQ3_S | [`ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-Coder-GGUF`](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-Coder-GGUF) | 58 GB | pris en charge ; non exécuté ici |
| **OrcaRouter Uncensored** | IQ3_XXS | le Flash-Next Uncensored d'`OrcaRouter` ([ORCA.md](docs/ORCA.md) ; une étape de pack manuelle, pas une option de l'installateur) | 84 GB | pris en charge ; non exécuté ici |
| **Qwen3.6-35B-A3B** (un modèle plus petit) | UD-Q4_K_XL, UD-Q6_K_XL et Q8_0 d'Unsloth, avec la couche de brouillon MTP | [`unsloth/Qwen3.6-35B-A3B-MTP-GGUF`](https://huggingface.co/unsloth/Qwen3.6-35B-A3B-MTP-GGUF) ([MODELS.md](docs/MODELS.md#qwen36-35b-a3b)) | 23-38 GB | pris en charge ; non exécuté ici |

Les lignes NVFP4 exigent des cartes NVIDIA (les cartes RTX 50 multiplient en largeur FP4 ; les cartes plus anciennes, dont la V100, en précision 8 bits). Les deux lignes QAD sont préparées sur votre PC à partir des safetensors des auteurs : aucun de leurs poids quantifiés n'est redistribué ici. Les lignes Gyro et treillis exigent le moteur 0.1.45.1 ou plus récent, les lignes Unsloth à down Q5_0 exigent la 0.1.45.7.

## Ce sur quoi ce fork se concentre

- **Prise en charge Tesla V100 / Volta (`sm_70`)** : préserver les compilations et le support d'exécution normaux du V100. Les compilations V100 utilisent une boîte à outils CUDA 12.x ; CUDA 13 ne compile pas le code `sm_70`.
- **Machine de test :** deux cartes Tesla V100-PCIE-16GB ; le GPU 0 utilise PCIe Gen3 x4 et le GPU 1 utilise Gen3 x16. Les mesures de pipeline ci-dessous utilisent les deux cartes. Le benchmark historique à une carte utilise uniquement la carte x16. Un système avec deux liens Gen3 x16 complets n'a pas été mesuré ici.
- **Intégration upstream** : ce projet n'est pas un remplacement séparé de Strata. Les changements upstream sont fusionnés périodiquement, avec la compatibilité V100 vérifiée et corrigée là où c'est nécessaire. Les nouvelles fonctionnalités upstream peuvent avoir des exigences matérielles supérieures à `sm_70` ; elles n'accélèrent pas automatiquement le V100.
- **La communauté Strata au sens large** : les améliorations, les rapports de bogues, les tests, la documentation et les contributions pour les autres plateformes prises en charge sont bienvenus.

Le support du V100 n'implique pas que chaque fonctionnalité upstream ou chaque autre configuration de GPU a été testée sur V100. Vérifiez le code, les notes de version et les rapports de mesure concernés avant de vous appuyer sur une fonctionnalité propre à un matériel.

## Documentation

L'[index de la documentation](docs/README.md) liste chaque page et son objet. Ces pages font référence :

- [Options du moteur](docs/ENGINE_FLAGS.md) : chaque option de ligne de commande, sa valeur et sa valeur par défaut.
- [Variables d'environnement](docs/ENVIRONMENT.md) : chaque variable que lisent le moteur, le serveur et le
  terminal.
- [Serveur](docs/SERVER.md) : les options du serveur, les clés de configuration d'exécution et tous les points
  d'accès de l'API.
- [Terminal](docs/TERMINAL.md) : chaque commande et sous-commande `strata`.
- [Réglage](docs/TUNING.md) : les réglages par objectif, les optimisations et les résultats mesurés.
- [Fonctionnement](docs/HOW_IT_WORKS.md) et le [glossaire](docs/GLOSSARY.md) : les parties du
  moteur et les termes que cette documentation utilise.
- [Détails](docs/DETAILS.md) : le registre des mesures, les formats de requête et l'historique des réglages.

Les pages suivent le [style de la documentation](docs/STYLE.md), qui est l'ASD-STE100 (anglais technique
simplifié).

## Notes de version

### Moteur 0.1.45.8 — 16 novembre 2026

La [version v0.1.45.8](https://github.com/jmnargi/Strata-V100/releases/tag/v0.1.45.8)
donne au moteur V100 actuel un numéro distinct. Elle comprend la correction du
décalage des lignes du prefill FP16 des experts, le travail sur le décodage NVFP4,
les économies de mémoire HC_Q8 optionnelles entre cartes, les améliorations
optionnelles du calcul des scores de prompt et du KV transféré, et la progression
du prompt par slot. Ce changement de version ne modifie ni le code d'inférence ni
les réglages par défaut. Les mesures et limites sont dans
[NVIDIA_V100.md](docs/NVIDIA_V100.md).

### Jeux Unsloth Dynamic v3.1 (1-4 bits) — 8 novembre 2026

Résout [#95](https://github.com/jmnargi/Strata-V100/issues/95) : le menu `--family unsloth` de l'installateur s'arrêtait à
UD-IQ4_XS (environ 4 bits). Il propose maintenant aussi les fichiers Dynamic v3.1 d'Unsloth, d'environ 1 à environ 4 bits, à la révision
qu'Unsloth a publiée le 2026-11-04 (commit `d9620119`) :

| Taille | Téléchargement | Experts routés | RAM avec tous les experts en RAM | Remarques |
| --- | ---: | ---: | ---: | --- |
| UD-IQ1_M | 68.0 GB | 32.4 GiB | ~48 GB | le plus petit ; le plus faible sur un travail exigeant ou long |
| UD-Q2_0 | 70.5 GB | 34.8 GiB | ~51 GB | experts Q2_0, comme le pack 2 bits le plus rapide |
| UD-IQ2_XS | 75.8 GB | 39.4 GiB | ~55 GB | 1 couche avec des down en Q5_0 |
| UD-Q2_K_XL | 79.6 GB | 42.8 GiB | ~59 GB | 1 couche avec des down en Q5_0 |
| UD-IQ3_XXS | 83.6 GB | 46.8 GiB | ~63 GB | pas de Q5_0 |
| UD-Q3_K_XL | 89.5 GB | 52.3 GiB | ~68 GB | 20 couches avec des down en Q5_0 |
| UD-IQ4_XS-v31 | 93.7 GB | 56.2 GiB | ~72 GB | 33 couches avec des down en Q5_0 ; son propre nom de catalogue, car ses fichiers Hub s'appellent UD-IQ4_XS |

Les cinq fichiers déjà proposés (UD-IQ4_XS, UD-Q4_K_XL, UD-Q5_K_XL, UD-Q6_K_XL, Q8_0) restent épinglés au
commit précédent, et la seconde disposition d'UD-Q4_K_XL (quatre fragments, table n-gram en dernier) au nouveau commit est acceptée depuis
les fichiers déjà présents sur le disque. UD-IQ1_S reste refusé : il mélange deux formats d'expert dans une seule couche.

Le travail moteur nécessaire : le produit scalaire d'expert Q5_0. Son décalage de -16 par poids était appliqué à la somme des
activations d'origine (la forme CUDA de llama.cpp), si bien qu'une couche dont la projection down est en Q5_0 sortait 2-3% moins précise par expert
que tout autre format down. Il multiplie maintenant la somme d'activations quantifiée, comme les chemins Q5_1 et Q4_0 et ggml-cpu
(CUDA et SYCL). Mesuré sur cette machine avec la parité d'expert synthétique (`native_expert_parity --synthetic`, gate-up IQ2_XXS /
IQ2_S / IQ2_XS / IQ3_XXS / IQ4_XS avec des down en Q5_0, erreur L1 relative par rapport à une référence float) : **2.40e-2 ..
3.18e-2 avant, 1.25e-2 .. 1.30e-2 après**, identique aux down IQ4_NL et Q2_0 (1.29e-2 .. 1.31e-2) sur les mêmes graines.
Les lancements d'experts groupés restent bit à bit égaux aux lancements d'avant la fenêtre (`native_grouped_parity` : Q5_0 ajouté à chaque
format gate/up, et les formes Flash-Next que ces fichiers utilisent, 0 échec). Version du moteur : 0.1.45.7 ; l'installateur refuse un
moteur plus ancien pour ces sept jeux avant de télécharger quoi que ce soit.

Les cinq fichiers épinglés, leurs octets et tous les autres formats sont inchangés : seuls les poids *experts* Q5_0 lisent ce noyau
(le chemin Q5_0 dense est `native_mmvq`, intact), et leur erreur sur l'expert entier par rapport à la référence float passe de
2.4-3.2e-2 à 1.25-1.30e-2.

### Ports de performance upstream v0.2.11 sélectionnés — 5 novembre 2026

Cette mise à jour porte sélectivement des optimisations upstream ; ce n'est pas une fusion
complète de la v0.2.11. Elle ajoute des noyaux d'expert treillis plus rapides, des tampons de travail de prompt
plus petits, moins d'allocations du lecteur disque PLE, et une correction du zéro-spin CPU/CUDA
sous Linux. La version du moteur reste 0.1.45.6.

**Mesuré sur la machine bi-V100 de ce fork :** deux cartes V100-PCIE-16GB à
175 W (GPU0 Gen3 x4, GPU1 x16), un Ryzen 5 3600, et Gyro-M abliterated Huihui
TQ2_0. Le profil utilisait la répartition 16, deux fenêtres de pipeline, MTP 4, KV int8,
et un contexte de 524,288 tokens avec YaRN 2.

| Tokens de prompt | Temps de prompt témoin | Group scratch + WMMA désactivés |
| ---: | ---: | ---: |
| 2,048 | 2,360.8 ms | 2,214.3 ms |
| 8,192 | 6,932.7 ms | 6,541.0 ms |
| 32,768 | 18,008.6 ms | 17,636.7 ms |

Ce sont des moyennes de passes A-B-A-B entrelacées à froid, quatre requêtes par taille
et par bras. Les deux bras incluaient déjà les ports treillis, HC et PLE. Ce n'est
pas un gain mesuré à chaque taille de contexte ni une garantie de vitesse sur la requête entière.

La combinaison optionnelle mesurée est `STRATA_TQ_SUB8=1 STRATA_EXPERT_PAIR=0
STRATA_PF_HC_REGION=1 STRATA_PF_GROUP_SCRATCH=1 STRATA_PF_WMMA=0`.
Group scratch peut changer l'arrondi en virgule flottante. Le changement d'allocation PLE
ne demande aucun drapeau. La correction du zéro-spin s'applique à `STRATA_POOL_SPIN_US=0` ; elle
n'établit pas une économie au réglage quotidien de 100.

Voir le [rapport des ports V100](docs/NVIDIA_V100.md#selected-upstream-0211-ports-gyro-m-2026-11-05)
pour les mesures individuelles, les gardes d'option, les vérifications numériques, les limites de compilation,
et les candidats testés qui n'ont pas été retenus. Aucune valeur par défaut du service n'est changée.

### Synchronisation upstream sélectionnée — 19 octobre 2026

Cette branche importe des changements partagés du moteur CUDA et du serveur depuis le commit
upstream `3d1fb480`. Elle conserve les noyaux Volta du fork, la croissance K/V du
pipeline et les changements de cache de conversation. Elle exclut les paquets de bureau sans rapport et les
mises à jour séparées des backends Metal, AMD et Intel. Voir la
[portée de la synchronisation](improvements/RELEASE-2026-10-19-V100.md).
Utilisez CUDA 12.8 avec `sm_70`. Une option importée n'établit pas un gain de
performance sur V100.

### Intégration upstream v0.1.39

Cette mise à jour intègre [Strata v0.1.39](https://github.com/Niko1221/Strata/releases/tag/v0.1.39), y compris les versions intermédiaires. Elle conserve les experts tensor core V100 du fork, les changements d'attention, la répartition asymétrique des GPU et le cache de conversation sur disque.

- **Requêtes parallèles :** définissez `"parallel": 2` dans la configuration du modèle, ou utilisez `./setup.sh --parallel 2`. L'installateur écrit cette valeur par défaut pour une nouvelle configuration Volta et préserve un réglage existant. Le moteur peut décoder des conversations distinctes ensemble. Les requêtes supplémentaires attendent une place libre. Les places utilisent de la mémoire GPU supplémentaire et peuvent réduire la vitesse d'une requête seule. Si deux sessions de place ne tiennent pas, le moteur le signale et sert une requête à la fois, sans modifier la configuration. Voir [places de traitement par lot](docs/BATCHING.md) pour les réglages d'exactitude, les coûts mémoire et les limites.
- **API et interface :** la mise à jour ajoute l'API OpenAI Responses, les réglages de modèle, la surveillance du cache de conversation et les métriques par place. Voir [détails de l'API](docs/DETAILS.md#using-it).
- **Performance et mémoire :** la mise à jour inclut des réductions de lancements dans la passe de vérification et des anneaux de prompt à taille d'octet. Elle ajoute aussi les huge pages de Linux, corrige le budget de RAM et propose des options multi-GPU. Un `--layer-split` explicite ne charge désormais automatiquement que les poids denses des couches de chaque étage (le drapeau upstream `--trim-stage-weights` et `STRATA_STAGE_TRIM` ont disparu). La VRAM libérée va au cache d'experts. Une optimisation pour les GPU plus récents n'implique pas une amélioration de vitesse sur V100. Les chemins optionnels qui changent l'arrondi doivent être mesurés avant utilisation.
- **Autres plateformes :** les chemins upstream Intel Arc, processeurs plus anciens et AMD supplémentaires restent disponibles. Cette machine ne fournit pas de validation matérielle pour ces plateformes.

V100 exige CUDA 12.x et une compilation qui inclut `sm_70`. N'installez pas le moteur CUDA 13 sur un V100. Une configuration parallèle explicite est recommandée pour les installations existantes ; une mise à jour n'écrase pas la configuration du modèle.

**Mesuré sur les deux cartes V100 (4 octobre 2026, compilation finale).** Une passe avant/après contrôlée
de la compilation intégrée par rapport au runtime installé précédemment, une requête à la fois
(pas de places de traitement par lot), à froid, Qwen3.8-Flash-Next Q2_0 avec KV int8 au contexte de 524,288 tokens,
la répartition 20/28 couches et MTP (`--spec 8`) inchangés ; deux requêtes neuves par taille, médianes :

| Prompt cible | Avant (tok/s) | Intégré (tok/s) | Changement |
| ---: | ---: | ---: | ---: |
| 2K | 701.41 | 718.5 | +2.44 % |
| 8K | 1,402.82 | 1,458.0 | +3.93 % |
| 32K | 2,262.4 | 2,283.7 | +0.94 % |

Chaque valeur est la médiane de deux requêtes neuves (`reused=0`). Les gains sont petits ; deux répétitions
par taille ne sont pas une affirmation statistique. Les six premiers tokens appariés correspondent à ceux de la référence.
Les réponses complètes ne sont pas déclarées identiques : le niveau d'expert adaptatif garde un mélange
d'experts différent sur le GPU et le CPU pour la compilation intégrée, et les deux chemins arrondissent
différemment. Aucun changement de decode ni de qualité n'est déclaré. Ce sont les valeurs mesurées de la compilation
finale (SHA-256 du moteur `e9ad337a…`), rapportées dans le
[rapport complet](benchmarks/v100-q2_0-upstream-v0139-2026-10-04.md).

Les places parallèles au contexte de 524,288 tokens dépendent de la répartition et de la mémoire restante sur
le dernier étage ; le repli mesuré (2 demandées, une requête servie à la fois, configuration
intacte) est décrit dans [docs/BATCHING.md](docs/BATCHING.md).

### Performance du pipeline bi-V100 — 15 octobre 2026

Le profil de service mesuré active `--pipeline-windows 2` avec l'ordre de GPU
`0,1` et la répartition par couches `16`. Il utilise deux cartes V100 de 16 GiB à 175 W,
un Ryzen 5 3600, le modèle natif IQ3_S en format mixte, le KV int8, et MTP
avec `--spec 8`. La capacité de contexte configurée reste 524,288 tokens.
Le plus grand prompt mesuré est 32,768 tokens.

**Sûreté des réponses longues :** utilisez une compilation avec la
[réparation de la croissance KV du pipeline](benchmarks/v100-pipeline-kv-growth-fix-2026-10-16.md)
quand vous combinez ce profil avec `STRATA_KV_GROW=1`. L'ancienne boucle de pipeline
pouvait accéder à des cellules KV non mappées pendant une réponse longue. La réparation
vide le travail spéculatif avant de faire croître le mapping, puis reprend
le pipeline. Les mesures de vitesse en sortie courte ci-dessous n'ont pas exercé
cette limite.

#### Prefill de prompt neuf jusqu'à 32K

Chaque valeur est la médiane de trois requêtes à tokens exacts sans réutilisation de prompt
et avec un plafond de sortie de 16 tokens. Le temps de prefill exclut la génération de la réponse.
Le débit est le nombre de tokens de prompt divisé par le temps de prefill médian.

| Tokens de prompt | Temps de prefill précédent | Temps de prefill du pipeline | Tokens/s précédents | Tokens/s du pipeline | Changement de débit |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 2,048 | 2.089 s | 2.090 s | 980.6 | 979.8 | −0.09% |
| 8,192 | 5.487 s | 5.494 s | 1,493.0 | 1,491.2 | −0.13% |
| 16,384 | 8.828 s | 8.857 s | 1,855.9 | 1,849.8 | −0.33% |
| 32,768 | 15.668 s | 15.677 s | 2,091.4 | 2,090.2 | −0.06% |

Le prefill de prompt neuf est à peu près inchangé. Ces petites différences
n'établissent pas une régression significative.

#### Decode de conversation

La comparaison HTTP séparée utilise un prompt initial de 8,828 tokens et six
tours de suite. Le contexte atteint environ 11.9K tokens. Chaque valeur
est la médiane de quatre requêtes, avec l'ordre des bras inversé un tour sur deux
et un plafond de sortie de 256 tokens. Les nouveaux tokens de prompt sont les tokens traités à
ce tour, pas la taille totale du contexte.

| Tour | Nouveaux tokens de prompt | Tokens/s de decode précédents | Tokens/s de decode du pipeline | Changement de decode |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 121 | 70.20 | 79.45 | +13.18% |
| 2 | 321 | 70.60 | 82.05 | +16.22% |
| 3 | 1,086 | 76.60 | 79.00 | +3.13% |
| 4 | 128 | 73.00 | 79.90 | +9.45% |
| 5 | 313 | 74.35 | 86.55 | +16.41% |
| 6 | 1,089 | 74.15 | 80.75 | +8.90% |

La médiane des six ratios de débit de decode est **+11.31%**.
Ce n'est pas un gain de decode mesuré à chaque taille de contexte jusqu'à 32K.
Les longueurs générées et le travail MTP diffèrent avec le partage CPU automatique et
le routage résident. La latence de la requête entière est mitigée ; une médiane HTTP
de tour court régresse. N'interprétez pas le gain de decode comme une accélération garantie
de la requête entière ni comme une comparaison à sortie appariée.

Un contrôle séparé à réglages fixes fait correspondre les trois séquences complètes de sortie de 16 tokens
aux tailles de prompt 2K, 8K et 32K. Il ne prouve pas une sortie identique
pour chaque requête de production.

Voir le [rapport d'évaluation](benchmarks/v100-upstream-pr-evaluation-2026-10-15.md)
pour les temps exacts, la provenance de compilation, les essais rejetés et les commandes
de reproduction. Voir le [guide du profil de service](docs/NVIDIA_V100.md#dual-pcie-pipeline-service-profile)
avant d'activer l'option sur une autre installation.

### Cache KV TurboQuant — 25 octobre 2026

TurboQuant est un format de cache KV optionnel. Il fait tourner chaque tête de 256 valeurs par une transformée de Hadamard, puis
quantifie les valeurs tournées contre un codebook Lloyd-Max. Chaque tête garde une norme FP16. Les largeurs sont
3, 4, 6 et 8 bits par valeur. L'ajustement est à erreur quadratique minimale seulement ; cette version n'a pas de résidu QJL.

Utilisez `--kv tq6` (ou `tq3`, `tq4`, `tq8`). Le cache est résident seulement : il reste en VRAM, il n'y a pas de copie
hôte, et `--kv-resident` est refusé. La couche de brouillon MTP garde son cache INT8. L'installateur ne propose pas ces
valeurs. Rien n'est une valeur par défaut, et le service quotidien reste en int8.

À la capacité configurée de 524,288 tokens, avec les 12 couches d'attention du modèle, le cache KV est :

| `--kv` | Octets par cellule, une couche | KV à capacité complète, 12 couches | Changement par rapport à int8 |
| --- | ---: | ---: | ---: |
| int8 | 1,056 | 6.1875 GiB | — |
| tq8 | 1,032 | 6.046875 GiB | −0.14 GiB |
| tq6 | 776 | 4.546875 GiB | −1.64 GiB |
| tq4 | 520 | 3.046875 GiB | −3.14 GiB |
| tq3 | 392 | 2.296875 GiB | −3.89 GiB |

L'économie principale de tq6 est 1.64 GiB à capacité complète. Ce sont des capacités calculées, pas des réductions
`nvidia-smi` mesurées : le cache d'experts remplit la VRAM libérée, donc la mémoire GPU totale reste près de 15,600 MiB.

Le lecteur de prompt V100 est une seconde option : `STRATA_PROMPT_ATTN_TQ=1` sur sm_70. Il décode des tuiles
compressées pour le chemin de prompt ; non défini ou `0` garde le repli de l'attention de decode. Il ne change pas les octets KV.

Mesuré sur deux cartes Tesla V100 de 16 GB à 175 W avec IQ3_S, `--spec 4`, `--layer-split 16`,
`--pipeline-windows 2`, un plafond de sortie de 2,048 tokens à froid, et une passe par taille et par bras. Le bras tq6 avait
le lecteur de prompt activé. À 32,768 tokens de prompt, tq6 a fait le prefill à 1,687 tokens/s contre 2,117 pour int8,
et le decode à 120.2 tokens/s contre 125.5. Avec le lecteur activé, le temps total de la requête de 51,746 tokens de tq6
est passé de 40.3–40.4 s à 31.7–31.9 s, toujours plus lent que les 25.8 s d'int8. Ces mesures ne montrent pas un
gain de vitesse. Un test d'aiguille a trouvé 15 cellules sur 15 dans les deux bras jusqu'à 122,289 tokens de prompt ; ce n'est pas un
résultat de qualité général.

Voir le [rapport complet](benchmarks/v100-turboquant-2026-10-25.md) et
[les détails](docs/DETAILS.md#tesla-v100-fork-benchmark).

### Treillis par paires Gyro-L sur deux V100 — 31 octobre 2026

Cette branche lit le `Qwen3.8-Flash-Next-Gyro-L-TQ3_0.gguf` d'agentionai (108,323,559,808 bytes, révision Hugging Face
`05892758a67efff0895d23c6df10a6fdb6be8cf4`). Ses experts routés sont stockés avec les types de treillis par paires
TQP5, TQP6 et TQP7 (types ggml 147 à 149) dans une base tournée par Hadamard. L'étape de pack
(`tools/iq_pack.py --compat-bf16`) et les noyaux sont décrits dans [docs/GYRO.md](docs/GYRO.md). CPU et CUDA
sont pris en charge ; Metal et SYCL n'ont pas de noyaux de treillis par paires. Cela exige le moteur 0.1.45.6 ou plus récent.

Deux défauts du moteur ont d'abord dû être réparés. Les noyaux d'expert eux-mêmes étaient déjà exacts au bit près par rapport au
ggml des auteurs :

- **La fenêtre de decode perdait la rotation.** Pour un pack replié, la fenêtre faisait tourner l'entrée d'expert, puis
  écrasait cette copie tournée par la version non tournée avant que les experts ne la lisent, si bien que les projections gate/up
  s'exécutaient dans la mauvaise base sur chaque token décodé. La sortie en un seul coup semblait plausible puis bouclait ; le
  serveur était faux dès le premier token. La fenêtre garde maintenant une marque « image écrite » par couche.
- **L'assistant de prompt de la répartition par couches** (en option) ne faisait tourner que sa sortie SwiGLU et écrivait les lignes d'un groupe dans un
  tampon disposé pour les tuiles WMMA de l'étage. Il fait maintenant tourner l'entrée qu'on lui donne et possède son H.

Mesuré sur la paire de ce fork : deux Tesla V100-PCIE-16GB (GPU 0 sur PCIe Gen3 x4, GPU 1 sur x16) à 175 W, un
Ryzen 5 3600, répartition par couches 14, `--spec 4 --spec-min-p 0.70`, KV int8 avec croissance KV, `--pipeline-windows 2`,
contexte de 262,144 tokens, l'environnement du service lui-même, passes entrelacées :

| Tokens de prompt | Prefill (tokens/s) |
| ---: | ---: |
| 2,328 | 632 |
| 8,199 | 1,036-1,084 |
| 16,371 | 1,344-1,411 |
| 32,679 | 1,597-1,642 |

Les réponses se décodent à 68 à 75 tokens/s de 2K à 32K de contexte (médianes de réponses répétées de 520 tokens) ; le point
de répartition ne déplace pas cela au-delà du bruit entre passes. Une lecture est limitée par les experts qui ne sont pas en VRAM : la
carte x16 transfère 15.01 GiB sur PCIe pour une lecture de 2.3K tokens (profil à répartition 16) et les deux étages s'exécutent
l'un après l'autre, donc la carte x4 reste inactive environ 700 ms à chaque lecture.

Le pack sert à travers l'API compatible OpenAI : « Paris » pour la question de capitale, un tour de suite qui
réutilise 86 tokens de prompt, une demande de code, un appel d'outil `get_weather` et une réponse correcte à partir d'un résultat d'outil,
et une question sur un passage de 442 tokens répondue depuis le passage.

**Répartition par couches 14 au lieu de 16.** Quatorze couches sur la carte x4 et 34 sur la carte x16 lisent 7.9% plus vite à
2.3K tokens (632 contre 586 tokens/s) et 6% plus vite à 8.2K, pareil à 16.3K, et 1.5% plus lentement à 32.7K.
15 et 13 se situent entre, et 20 est 23% plus lent à 2.3K et 13% à 8.2K. Une lecture est ce sur quoi un tour d'agent passe
son temps, donc ce profil utilise 14.

**Ce qui a été mesuré et n'a pas payé sur cette paire :** `--spec 6` et `--spec 8`, `--spec-min-p 0.50` et
`0.85`, un `STRATA_PREFILL_CPU_SHARE` fixe, `STRATA_PREFILL_CPU_SHARE_MAX` au-dessus de 6,144, `--prefill-pipe`,
`--max-context 131072`, `STRATA_HC_Q8=1`, et un `--vram-reserve-mib` plus grand. L'assistant de prompt lui-même est
plus lent ici (484 contre 583 tokens/s à 2.3K) : il ne peut prendre que les experts du premier étage, et sur cette paire
cet étage est la carte x4, qui transfère 1.77 GiB des 16.8 GiB qu'une lecture déplace. `--mtp` et le
`STRATA_ROUTE_TAIL_SKIP=7` par défaut valent la peine d'être gardés : le decode tombe à 35.5 et 48 tokens/s sans eux.

## Benchmark V100

Le tableau ci-dessous enregistre une passe à une seule carte sur une Tesla V100-PCIE-16GB en PCIe Gen3 x16. La machine de test a aussi une seconde V100 en PCIe Gen3 x4 ; elle n'a pas servi à cette passe. Les résultats d'une configuration à deux cartes ou d'une machine avec deux liens pleine largeur peuvent différer et être meilleurs. C'est une mesure reproductible, pas une promesse de vitesse sur d'autres systèmes.

| Prompt cible | Tokens de prompt médians | Prefill (tokens/s) | Decode (tokens/s) | Température GPU maximale |
| ---: | ---: | ---: | ---: | ---: |
| 1K | 1,036 | 398.6 | 52.0 | 56 °C |
| 4K | 4,111 | 1,214.0 | 50.8 | 61 °C |
| 8K | 8,185 | 1,442.4 | 48.1 | 63 °C |
| 16K | 16,367 | 1,490.7 | 49.8 | 67 °C |
| 32K | 32,765 | 1,495.7 | 47.3 | 72 °C |
| 64K | 65,533 | 1,484.3 | 43.0 | 81 °C |
| 128K | 131,063 | 1,033.0 † | 40.9 | 84 °C |
| 256K | 256,080 | 617.8 † | 36.8 | 84 °C |

**Configuration de test :** Qwen3.8-Flash-Next Q2_0 ; un seul V100 (GPU1, PCIe Gen3 x16) ; Ryzen 5 3600 ; 48 GB DDR4-3200 ; CUDA 12.8. Contexte de 262,144 tokens ; cache KV int8 ; prefill automatique ; MTP avec `--spec 8` et plancher de brouillon 0.70 ; variante d'expert appariée ; réserve de vision de 700 MiB. Chaque valeur est la médiane de trois requêtes neuves, non mises en cache (`reused=0`) avec exactement 256 tokens de sortie. Les temps viennent du point d'accès `/metrics` du moteur.

Avant chaque requête, la carte à refroidissement passif est restée inactive au moins 120 secondes. Elle est descendue à 55 °C ou moins sans ralentissement thermique logiciel pendant 15 secondes. Les températures de départ étaient 51–55 °C. Toutes les requêtes sont incluses. Les prompts 128K et 256K ont atteint 84 °C et subi un ralentissement thermique logiciel pendant le traitement du prompt, malgré le départ refroidi. Les longues passes peuvent limiter la fréquence. † indique un ralentissement pendant le prompt.

Mesuré le 2026-10-02. Moteur : version du fork 0.1.31, commit de compilation `78417ea` (SHA-256 `512b1f25d60479a2ddb66fcf1ddca5407963e45378a3c8a463413ad159edae17`). Le benchmark a été enregistré avant l'intégration upstream v0.1.36 ultérieure et les PR d'attention V100 ; ce n'est pas une mesure de la compilation `main` actuelle. Le tableau historique est conservé avec sa provenance d'origine. Ne l'interprétez pas comme une comparaison contrôlée avec des tableaux antérieurs ou des compilations actuelles.

Données brutes : [`summary.json`](bench/results/2026-10-02-v100-single-cooled/summary.json), [`protocol.json`](bench/results/2026-10-02-v100-single-cooled/protocol.json), [`matrix.json`](bench/results/2026-10-02-v100-single-cooled/matrix.json), [`completed-requests.json`](bench/results/2026-10-02-v100-single-cooled/completed-requests.json), et [`GPU telemetry`](bench/results/2026-10-02-v100-single-cooled/gpu.csv). Voir [la méthode complète et l'historique](docs/DETAILS.md#tesla-v100-fork-benchmark), [les instructions de benchmark](benchmarks/README.md), et [les rapports de performance V100](benchmarks/).

Les pull requests V100 récentes incluent des changements mesurés au niveau du noyau et indiquent explicitement quand les gains de bout en bout ne sont pas établis. Par exemple, le port de l'attention de prompt depuis la PR upstream #600 a rapporté 27.9% de temps de noyau d'attention int8 en moins à sa forme mesurée. Les changements de prefill du modèle allaient de +1.3% à +3.0% dans les passes rapportées ; il n'a pas établi de gain de decode fiable. Lisez le [rapport complet](benchmarks/v100-q2_0-pr600-2026-10-03.md) avant de comparer des résultats.

## Travaux de performance V100

Le travail V100 couvre le chemin de prompt, le chemin de decode, l'usage de la mémoire, les lectures de stockage et l'exécution multi-GPU. Les changements ci-dessous sont dans l'historique fusionné du fork ; chaque lien inclut sa propre méthode de test, les détails matériels et les limites. Les améliorations de temps de noyau ne produisent pas nécessairement le même gain en pourcentage sur le débit du modèle complet.

### Traitement du prompt

- **GEMM tensor core BF16-vers-FP16 sur Volta :** Volta n'a pas de support GEMM BF16 natif. Le fork convertit exactement les poids BF16 en FP16 et utilise les tensor cores du V100 pour les GEMM de prompt au lieu du repli scalaire. Les images d'activation de prompt FP16 associées utilisent la bonne représentation de bits.
- **Lectures de table PLE plus rapides :** passer les lectures PLE POSIX à des opérations `io_uring` O_DIRECT en file, avec un repli synchrone. Un gather PLE 8K neuf mesuré est passé d'environ 6.9 secondes à 7 millisecondes.
- **Prefill MMQ Q2_0 :** activer le chemin quantifié matrice-matrice Q2_0 de llama.cpp existant pour le type d'expert interne de Strata. Un prompt mesuré de 14,486 tokens a gagné 6.5% en débit de prefill (926.1 à 986.2–986.9 tokens/s).
- **Ne scorer que les blocs QSA vivants :** éviter de scorer des blocs au-delà de la plage vivante du prompt courant. Sur un prompt mesuré de 15,423 tokens, le temps de sélection QSA a chuté de 91.1%. Le débit de prefill de bout en bout est passé de 961.0 à 1,111.3–1,112.9 tokens/s dans les tests rapportés.
- **Attention de prompt tensor core pour Volta :** utiliser `mma.m8n8k4` sur le V100 au lieu du repli FP32. Une PR antérieure a mesuré environ 1.6–1.9× de vitesse dans le noyau d'attention et un débit de prompt plus élevé dans ses tests. Le port ultérieur de la PR upstream #600 a réduit le temps de noyau d'attention int8 de 27.9% à une forme mesurée. Les deltas de prefill de bout en bout rapportés étaient plus petits (+1.3% à +3.0% dans ses tests 4K–32K).

### Decode et exécution multi-GPU

- **Fenêtre de vérification par lots :** exécuter plusieurs positions de decode spéculatif ensemble, ce qui réduit la surcharge de lancement par token. Des tests historiques appariés ont rapporté une génération de sortie plus rapide, mais les résultats varient selon le prompt et les conditions d'exécution.
- **Scoring de blocs QSA par tuiles :** réutiliser les données de clé sélectionnées entre les tuiles de requête au lieu de les relire plusieurs fois. Le noyau mesuré de 256 requêtes / 65K blocs est passé de 3.5 ms à 1.3 ms ; son arithmétique par score est restée identique au bit près.
- **GR Volta fusionné et ajout KV par lots :** combiner le travail entre les positions de vérification et ajouter les données KV en moins de lancements. Sur la configuration bi-V100 testée, le decode a augmenté de 5.6–17.3% sur les prompts 4K–32K, tandis que le prefill est resté à peu près inchangé.
- **Réglage asymétrique bi-V100 :** améliorer la gestion des lignes hôtes, la répartition du plan d'appareils et la propriété de la mémoire pour une configuration à deux cartes. Une configuration installée antérieure a montré des gains de prefill de 13.3–18.7% à 4K–32K. Cependant, les résultats de decode étaient mitigés et les deux GPU ont ralenti thermiquement pendant cette passe. Ces valeurs n'isolent pas chaque changement de code.
- **Réduire la surcharge de poids et de répartition :** ne charger que les couches possédées par chaque carte lors d'une répartition explicite. Évitez aussi la quantification d'activation CPU inutile et les barrières de répartition vides, et exécutez en option des noyaux d'expert à lignes appariées. L'allocation dense native mesurée du GPU0 est passée de 1,376.20 MiB à 541.27 MiB. Le mode à lignes appariées est optionnel ; les résultats rapportés sur le modèle complet n'ont pas établi d'amélioration du decode.

### Noyaux d'attention et trafic mémoire

- **Attention de decode (port de la PR upstream #540) :** réduire les mélanges de scores et le trafic de mémoire partagée tout en préservant l'ordre d'accumulation. Le temps de noyau mesuré est passé de 52.90 à 31.13 microsecondes sur le GPU0 pour M=1, et de 180.42 à 111.85 microsecondes pour M=8. Les changements rapportés sur le decode du modèle complet étaient petits ; le prefill était pratiquement inchangé.
- **Attention de prompt (port de la PR upstream #600) :** utiliser l'implémentation tensor core Volta avec quatre calculs MMA indépendants. La PR rapporte des vérifications d'exactitude et des mesures au niveau du noyau et du modèle. La réduction de 27.9% du noyau ne doit pas être lue comme une augmentation de 27.9% de la vitesse du modèle complet.
- **Gather KV et récurrence GDN (ports de la PR upstream #627) :** utilisez des charges KV int8 alignées plus larges avec un repli sûr à la largeur d'origine. Utilisez plusieurs accumulateurs pour la récurrence de prompt V100. Les temps de noyau mesurés se sont améliorés d'environ 10% pour un gather KV 32K et d'environ 7% pour une récurrence de 4,096 tokens. Le prefill du modèle apparié a différé de moins de 0.2% ; aucun gain de decode de bout en bout fiable n'a été établi.

### Support de compilation et preuves

Le fork a aussi réparé la configuration et l'édition de liens CUDA pour `sm_70`, et maintenu l'admission des appareils V100 à travers les fusions upstream. Il a adapté les changements upstream quand leurs exigences d'architecture excluent Volta. Ces corrections de compatibilité gardent les chemins optimisés compilables ; elles ne sont pas en elles-mêmes des affirmations de performance.

Pour les preuves complètes et la portée de l'implémentation, voir [PR #1](https://github.com/jmnargi/Strata-V100/pull/1), [#2](https://github.com/jmnargi/Strata-V100/pull/2), [#5](https://github.com/jmnargi/Strata-V100/pull/5), [#6](https://github.com/jmnargi/Strata-V100/pull/6), [#11](https://github.com/jmnargi/Strata-V100/pull/11), [#12](https://github.com/jmnargi/Strata-V100/pull/12), [#15](https://github.com/jmnargi/Strata-V100/pull/15), [#16](https://github.com/jmnargi/Strata-V100/pull/16), et [#17](https://github.com/jmnargi/Strata-V100/pull/17). Les résultats utilisent des dates, compilations, prompts, placements de GPU et protocoles différents. Ce sont des mesures historiques, pas une seule comparaison avant/après contrôlée ni une garantie de gains sur chaque système V100.

## Autre matériel

- **Cartes graphiques plus anciennes** (Tesla P40 / V100, GTX 10, Radeon VII / MI50, RX 6700 XT, RX 5500 XT) : [Older GPUs](docs/OLDER_GPUS.md).
- **Intel Arc**, compilé depuis les sources sous Linux : [Intel Arc](docs/INTEL_ARC.md).
- **AMD Ryzen AI Max (Strix Halo)**, compilé depuis les sources sous Linux : [Strix Halo](docs/STRIX_HALO.md).
- **Processeurs plus anciens sans AVX2** : ils fonctionnent, mais lentement. [Older CPUs](docs/INSTALL.md#older-cpus-experimental).

## Installation et démarrage

Ce fork suit l'expérience d'installation upstream. Pour les exigences de plateforme complètes et actuelles, les choix de modèle et les options, voir le [guide d'installation](docs/INSTALL.md), le [guide des modèles](docs/MODELS.md) et le [guide d'installation upstream](docs/AI_SETUP.md).

- **Windows :** téléchargez ou clonez ce dépôt et lancez `START-HERE.bat`.
- **Linux :** clonez ce dépôt et lancez `./setup.sh`.
- Suivez les invites pour choisir un modèle et une taille de contexte. L'installateur télécharge les données du modèle et démarre le service local.
- Ouvrez l'adresse affichée par l'installateur (normalement `http://127.0.0.1:8080`).

Pour une Tesla V100, utilisez le moteur CUDA 12 avec le support `sm_70`. L'installateur choisit le moteur pour GPU plus anciens pour Volta ; sous Linux, il le compile localement. L'archive Windows CUDA 12 upstream inclut le code Volta, mais elle n'a pas été testée sur cette machine Linux. Voir le [guide des GPU plus anciens](docs/OLDER_GPUS.md) pour les exigences de boîte à outils et de pilote. Pour Docker, suivez les [instructions Docker](docs/INSTALL.md) et compilez pour l'architecture GPU cible.

## Utiliser Strata

- **Application web :** utilisez l'URL locale affichée par l'installateur pour le chat et le moniteur en direct.
- **API compatible OpenAI :** définissez l'URL de base de votre application sur `http://127.0.0.1:8080/v1`.
- **API compatible Anthropic :** utilisez `http://127.0.0.1:8080/v1/messages`.
- **API OpenAI Responses :** utilisez `http://127.0.0.1:8080/v1/responses`. Ce point d'accès sans état prend en charge les appels d'outils de style Codex ; il ne stocke pas `previous_response_id`.
- **Images, MCP, multi-GPU et configuration :** voir les [détails](docs/DETAILS.md), le [serveur MCP](docs/MCP_SERVER.md) et le [guide multi-GPU](docs/MULTI_GPU.md).

Le serveur écoute normalement sur localhost. Si vous l'exposez à d'autres machines, configurez une clé d'API et utilisez un réseau de confiance. Ne publiez pas de secrets ni de configuration privée dans les tickets, les résultats de benchmark ou les pull requests.

### Cache de conversation sur disque (en option)

L'état de conversation a trois niveaux. L1 est la session active et ses points de contrôle de prompt résidents. L2 est le cache optionnel en RAM hôte (`--conversation-cache-mib`). L3 est un stockage disque optionnel qui conserve les conversations enregistrées et les groupes de points de contrôle entre les redémarrages. Les couches KV diffusées en flux utilisent leurs pools hôtes faisant autorité quand le moteur capture l'état.

Pour utiliser le stockage disque, ajoutez `--conversation-cache-disk strata-conversations --conversation-cache-disk-gib 25` aux arguments du moteur. Le chemin et un budget GiB positif sont requis ensemble. Le stockage est désactivé par défaut, il exige `--serve`, il fonctionne avec `--conversation-cache-mib 0`, et il prend en charge `--layer-split` sur plusieurs GPU. En option : `--conversation-cache-disk-slots N` plafonne le nombre d'enregistrements, et `--conversation-cache-disk-min-free-mib N` garde de l'espace libre sur le système de fichiers.

Le moteur préserve un point de contrôle actif avant que la rétention ne le supprime ou qu'une requête n'écrase sa branche. Chaque point de contrôle enregistré inclut l'état GDN, l'historique PLE, l'état de l'indexeur QSA et le support KV requis. Les données KV de tokens seules ne suffisent pas à reprendre ce modèle.

Les points de contrôle liés partagent une image KV par groupe d'archives. Le moteur déplace et étend cette image au lieu de copier le préfixe complet pour chaque point de contrôle. Il ne copie que les pages KV nouvelles ou modifiées quand le support est encore valide. Un groupe cesse de grandir avant d'atteindre la limite de métadonnées du disque. Un groupe ultérieur a son propre support. Le moteur ne combine pas un point de contrôle enregistré avec des octets KV d'un autre calcul du préfixe.

Les groupes d'archives utilisent les mêmes limites d'octets et de places L2 que les conversations enregistrées. Les victimes de capacité L2 passent en L3 quand le niveau disque est activé. Une admission L2 réussie n'écrit pas aussi l'image sur le disque. Si une image ne tient pas en L2, le moteur tente de la stocker en L3. Les limites de RAM physique et d'espace disque libre s'appliquent toujours. Le réglage de tokens minimum s'applique aussi aux archives de points de contrôle.

Préserver un point de contrôle est une optimisation, pas une exigence. Si le plancher de RAM hôte ou le budget disque refuse l'image, le moteur abandonne le point de contrôle, journalise l'abandon et continue de servir. Il n'arrête pas le serveur. La session reste correcte ; seule la réutilisation de ce point de contrôle est perdue.

La recherche sélectionne le plus long préfixe compatible dans la session active, L2, L3 et les places de traitement par lot inactives. Les tokens, l'identité de l'image et le mode de vecteur de contrôle doivent correspondre. Pour une modification d'un message antérieur, la réutilisation s'arrête à un point de contrôle enregistré avant le premier token modifié. Le moteur doit traiter à nouveau le suffixe ; il ne peut pas reprendre à un token correspondant arbitraire sans l'état récurrent à cette position. Augmenter `--prompt-cache` ne remplace pas la préservation des points de contrôle.

Une archive L2 reste disponible après restauration. Une image L3 valide est promue en L2 quand le budget d'octets et le plancher de RAM physique le permettent. Seul un sous-ensemble borné de ses points de contrôle devient résident dans la session active. Restaurer une branche ne supprime pas les autres points de contrôle de l'archive enregistrée.

Le stockage disque prend en charge les chaînes de conversation en ajout seul. Une chaîne qui continue peut ajouter de nouveaux identifiants de tokens, des points de contrôle, l'état d'exécution et des pages KV. Une éviction L2 écrit une image de départ complète ; elle ne demande pas de capture GPU. Un retour en arrière démarre une branche séparée au lieu de remplacer la chaîne source. Le format de fichier ne change pas pour les archives de points de contrôle.

Les panneaux disque ont des sommes de contrôle et des champs de version. Une image de départ utilise un fichier temporaire et un renommage. Un ajout incomplet est récupéré jusqu'au dernier panneau complet. Les enregistrements invalides ou incompatibles sont supprimés. Le stockage disque évince les chaînes les moins récemment utilisées pour respecter son budget d'octets, sa limite d'enregistrements et son plancher d'espace libre. Ainsi, ni L2 ni L3 ne fournit un historique illimité. Si les deux niveaux ont évincé le point de contrôle requis, la requête doit traiter à nouveau le préfixe manquant.

Le moteur capture une racine de prompt système partagée une fois en L2 quand la rétention de racine est activée. L3 reçoit la racine lors d'une éviction L2 ou d'une admission L2 échouée. Avec L2 désactivé, un L3 activé peut recevoir la racine directement. `--conversation-cache-keep-root` et `--conversation-cache-disk-keep-root` contrôlent la rétention de racine dans chaque niveau. Leurs variantes `--no-` la désactivent. Les racines ont une priorité d'éviction plus faible que les images ordinaires ; les épinglages explicites de points de contrôle de requête restent protégés.

Le niveau disque n'ajoute pas d'exécution concurrente. Voir [places de traitement par lot](docs/BATCHING.md) pour les coûts mémoire des requêtes concurrentes. La capture doit se terminer avant qu'une requête n'écrase les octets de session requis. La restauration est synchrone. Les écritures d'éviction L2 utilisent des instantanés hôtes et peuvent ajouter de la latence d'E/S disque. Dimensionnez les deux budgets de cache pour les conversations et l'historique de points de contrôle dont vous avez besoin. Voir [le guide V100](docs/NVIDIA_V100.md) pour la vérification de la rétention de points de contrôle et [le rapport de benchmark disque](benchmarks/v100-l3-conversation-cache-2026-10-03.md) pour les mesures antérieures.

#### Résultat V100 mesuré

Sur la configuration bi-V100 à répartition par couches de ce fork, une conversation de 33,725 tokens a été reprise depuis un cache NVMe de 25 GiB. La lecture a pris 1,614.6 ms et la restauration a pris 315.8 ms. La phase de prompt reprise complète, y compris 22 nouveaux tokens, a pris 3,015.1 ms. Une lecture à froid du même préfixe de 33,725 tokens a pris 20,332.7 ms. C'est une **accélération de 6.74x** et une **réduction de 85.2% de la latence de prompt** pour la requête reprise. L'enregistrement de 943.6 MiB a restauré un état du modèle principal exact au bit près et une sortie identique sur les deux GPU. La récupération après redémarrage a passé la parité exacte au bit près, et le repli sur enregistrement corrompu a produit une sortie identique. Voir [le rapport de benchmark](benchmarks/v100-l3-conversation-cache-2026-10-03.md).

Mesuré sur la machine bi-V100 à répartition par couches de ce fork (Qwen3.8-Flash-Next Q2_0, KV int8) : deux
conversations alternant dans un cache de travail ont été servies correctement sur 8 requêtes avec
`corruptions=0 evictions=0` ; un fichier de chaîne par conversation est passé de 238 MiB -> 594 MiB sur
trois sauvegardes de +12 nouveaux tokens chacune (un stockage à instantané complet aurait écrit un nouveau fichier de ~2.4 GiB
par sauvegarde à 100k+ tokens). La première sauvegarde d'un court chat neuf amorce la chaîne
(~118 MiB : une page K/V par couche plus l'état d'exécution fixe du modèle) ; ensuite chaque sauvegarde
est le delta (+12 tokens ~ 118 MiB) et le chemin de relecture/restauration est inchangé (~0.4 s en lecture,
~0.04-0.08 s en restauration).

### Vision à la demande (en option)

L'encodeur d'images (`strata-vision`) occupe environ 1.74 GiB de VRAM (1.43 GiB sur le premier GPU, 0.31 GiB sur le second) aussi longtemps qu'il tourne. Avec la vision à la demande, il ne tourne pas entre les requêtes d'images. Cette VRAM reste avec les caches d'experts, donc le travail texte seul garde le nombre complet d'experts résidents.

Pour l'activer, ajoutez à la configuration du modèle :

- `"vram_elastic": true` (le drapeau de moteur `--vram-elastic`) : les caches d'experts sont segmentés, donc le moteur peut rendre la VRAM et la reprendre entre les requêtes.
- `"vision": {"lazy": true, "idle_s": 900}` : l'encodeur démarre à la première requête d'image et s'arrête après `idle_s` secondes sans requête d'image (900 = 15 minutes). L'encodeur a besoin du mode « une requête à la fois » du moteur, donc ne le combinez pas avec `"parallel"` (ou donnez-lui sa VRAM autrement, par exemple `POST /v1/vram`).

Comportement :

- Une requête texte ne démarre jamais l'encodeur. Une requête d'image demande d'abord au moteur de rendre la VRAM des GPU. Elle utilise une réserve par GPU, `[1700, 600]` sur une répartition par couches et `[1800]` sur un seul GPU, à remplacer avec `"vram_mib"`. Puis elle démarre l'encodeur et encode l'image pendant que le moteur est inactif.
- Après `idle_s` secondes sans requête d'image, l'encodeur s'arrête et les caches d'experts reprennent leur taille complète. Les images encodées restent en cache sur le disque, donc renvoyer la même image ne la réencode pas.
- Mesuré sur la répartition bi-V100 de ce fork : les caches contiennent **14,005** experts résidents pendant que l'encodeur est inactif (12,688 avec l'encodeur résident). Une requête d'image à froid a été répondue en **5.9 s**, et les deux caches sont revenus à 14,005 après le déchargement de l'encodeur. Voir les [détails](docs/DETAILS.md).

## Un problème est survenu

Les problèmes courants et leurs corrections sont dans [TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md). Cette page couvre
l'application de bureau, la commande `strata` et l'installateur. Le tableau complet des messages d'erreur est dans
[DETAILS.md](docs/DETAILS.md#troubleshooting).

Si le serveur démarre mais qu'une requête échoue, lancez `strata doctor`. Il lit les composants installés et
indique ce qui manque ou n'est pas à jour ; voir [TERMINAL.md](docs/TERMINAL.md#doctor).

Si une réponse est lente plutôt que cassée, lisez [TUNING.md](docs/TUNING.md). Il donne le réglage qui
compte pour chaque objectif, et le coût mesuré de chacun.

## Contribuer

Les contributions sont bienvenues. Vous n'avez pas besoin d'un V100 pour aider. La documentation, les tests, l'installation, le comportement du serveur, la compatibilité de l'API et les améliorations pour les autres appareils pris en charge sont utiles. Le code et les résultats de performance spécifiques au V100 profitent d'une validation sur du matériel Volta réel.

1. Consultez les [tickets ouverts](https://github.com/jmnargi/Strata-V100/issues) et les [pull requests](https://github.com/jmnargi/Strata-V100/pulls) pour éviter le travail en double.
2. Pour les grandes fonctionnalités upstream, vérifiez si le changement appartient au [dépôt upstream de Niko](https://github.com/Niko1221/Strata). Ce fork intègre périodiquement les changements upstream ; les corrections et mesures ciblées V100 peuvent être proposées ici.
3. Faites un changement ciblé, décrivez le matériel et les étapes exactes utilisées pour le tester, et incluez les tests pertinents.
4. Pour les affirmations de performance, rapportez la référence et le candidat, tous les réglages d'exécution, la charge de travail, le nombre de répétitions et les limites. Incluez les données brutes ou une commande reproductible quand c'est possible. Séparez le temps de noyau du débit du modèle complet et ne revendiquez pas un gain que les mesures ne montrent pas.
5. Ouvrez une pull request contre la branche `main` de ce dépôt. Expliquez si le changement est spécifique au V100, une intégration upstream ou une amélioration générale de Strata. Gardez les identifiants, la configuration personnelle et les fichiers locaux sans rapport hors du changement.

Voir [`benchmarks/README.md`](benchmarks/README.md) et [`docs/DETAILS.md`](docs/DETAILS.md) pour l'approche de mesure actuelle. Les contributions restent soumises à la licence du projet et aux licences des composants inclus et des fichiers de modèle.

## Historique du projet

Ce fork développe le support V100 et le travail spécifique au V100 tout en conservant le développement upstream de Strata. Les pull requests d'intégration et de performance récentes illustrent ce processus :

- [#14 — Fusionner la v0.1.36 upstream en préservant le support V100](https://github.com/jmnargi/Strata-V100/pull/14)
- [#15 — Porter les changements mesurés de gather KV V100 et de récurrence GDN depuis la PR upstream #627](https://github.com/jmnargi/Strata-V100/pull/15)
- [#16 — Réduire les mélanges et le trafic de mémoire partagée de l'attention de decode V100](https://github.com/jmnargi/Strata-V100/pull/16)
- [#17 — Porter le noyau d'attention de prompt Volta de la PR upstream #600](https://github.com/jmnargi/Strata-V100/pull/17)
- [#88 — Lire les experts en treillis par paires Gyro-L (TQP5/6/7). Corriger la fenêtre de decode repliée et l'assistant de prompt de la répartition par couches. Passer le profil V100 à `layer_split 14`](https://github.com/jmnargi/Strata-V100/pull/88)

Ces rapports décrivent des changements testés précis ; ce n'est pas une affirmation générale de performance plus rapide sur tous les modèles ou GPU. Pour plus de détails, parcourez l'[historique complet des pull requests](https://github.com/jmnargi/Strata-V100/pulls?q=is%3Apr+is%3Amerged) et l'[historique des commits](https://github.com/jmnargi/Strata-V100/commits/main).

## Crédits et licence

Ce dépôt est un fork de [Strata par Niko1221](https://github.com/Niko1221/Strata). **Nous sommes profondément reconnaissants envers Niko et tous les contributeurs upstream.** Leur travail d'origine est la raison pour laquelle ce fork, son support V100 et ce runtime de modèle local sont possibles. Nous visons à créditer et à suivre le travail upstream en intégrant périodiquement ses changements.

Strata incorpore [llama.cpp / ggml](https://github.com/ggml-org/llama.cpp) et d'autres composants open source. Le projet est sous [licence MIT](LICENSE) ; les composants et les fichiers de modèle peuvent avoir des conditions séparées. Voir les [notes de crédits et de licence](docs/HOW_IT_WORKS.md#credits).
