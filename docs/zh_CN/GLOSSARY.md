# docs/zh_CN 翻译规范与术语表

本目录是 `docs/` 下英文文档的简体中文译文。**英文文档始终是权威版本**：两者冲突时以英文为准；
新特性、实测数字和参数名只以英文文档为来源。

本文件是全部译文的统一标尺：写新译文、改旧译文、review PR 都按这里执行。

## 1. 结构：与英文原文 1:1

- 标题层级、段落切分、列表、代码块、表格、图片、引用块的数量与顺序**必须**与原文一致，不合并、不省略、不增补。
- 表格行数、分隔行、列顺序、每格的数值照抄；数字、单位、百分比、时间、温度、token 数与原文逐一相同。
- 代码块内容（含其中的英文注释）原样保留，一个字符都不改。
- 原文的 issue/PR 引用（`#1415`、`PR #600`）、commit、版本号（`0.1.45`）、型号（`RTX 3090`、`gfx1100`）、
  参数（`--prefill`、`STRATA_ROUTE_TAIL_SKIP=1`）、路径（`bench/results/…`）一律不改。

## 2. 文件与链接

- 一篇英文文档对应一篇同名译文：`docs/FOO.md` → `docs/zh_CN/FOO.md`。不使用 `.p1.md` 之类的分块文件名。
- **标题译成中文，但锚点保留英文原 slug**。做法：在译后标题行**下面紧跟一行** HTML 锚点，`slug` 取自英文原标题：

  ```markdown
  ## 使用方式
  <a id="using-it"></a>
  ```

  这样站内目录跳转、以及 README / 其它文档里 `DETAILS.md#using-it` 这类跨文件锚点在改指中文版后仍然有效。
  slug 生成规则与 GitHub 一致：英文标题转小写，去掉标点，空格换 `-`（例如 `What you need` → `what-you-need`）。
- **指向 docs 内其它文档的链接**改写为同目录相对路径，锚点照原样：
  `[模型](MODELS.md#pick-by-memory)`（不要写 `../MODELS.md`）。
- **指向 docs 之外**的链接要多加一层 `../`，因为译文位于 `docs/zh_CN/`，比英文原文深一层：

  | 英文原文里 | 译文里 |
  | --- | --- |
  | `../README.md` | `../../README.md` |
  | `../SECURITY.md` | `../../SECURITY.md` |
  | `../bench/results/x.json` | `../../bench/results/x.json` |
  | `../packaging/UPDATES.md` | `../../packaging/UPDATES.md` |

  仓库根的 `https://…`、issue/PR 链接、`benchmarks/…`、`improvements/…` 这类**不带 `../`** 的链接照原样保留。
  层级写错时可运行 `python tools/check_docs_zh.py --fix-links` 自动修正（它只改「浅一层不存在、深一层存在」的链接）。
  英文原文里本来就失效的目标（如 `MAC.md`、`../TERMINAL_LAUNCH.md`）照原样保留，只留校验 warning。

## 3. 不翻译的内容

| 类别 | 示例 | 处理 |
| --- | --- | --- |
| 代码块 | `cmake --build build-hip --target strata -j2` | 原样，含注释 |
| 命令行/参数/环境变量 | `STRATA_PREFILL_CPU_SHARE=1`、`--vram-reserve-mib 3072` | 原样 |
| 文件名/路径/JSON 字段 | `experts.bin`、`"parallel": 2`、`id_slot` | 原样 |
| 型号/架构/版本 | `Tesla V100`、`sm_70`、`wave32`、`CUDA 12.8` | 原样 |
| 量化格式 | `Q2_0`、`IQ3_XXS`、`NVFP4` | 原样 |
| 界面原文串 | `"Windows protected your PC"` | 保留英文，必要时随后括注中文 |
| 日志/错误串 | `verify: timed out at layer N` | 原样（可加引号） |
| 表格数值与单位 | `+18.7%`、`66.9 tok/s`、`84 C` | 原样 |

## 4. 术语表（关键术语保留英文 + 中文说明）

正文里的**术语保留英文原词**，首次出现时用中文点明含义；其余句子用通顺的中文，不逐词硬译。

| 英文 | 译文用法 | 说明 |
| --- | --- | --- |
| prefill | `prefill`（预填充） | 首次出现写「prefill（预填充）」，之后写 `prefill` |
| decode | `decode`（解码） | 同上 |
| expert | 专家层（expert） | 泛指模型专家时用「专家层」，指缓存条目/槽位时可写「专家」 |
| expert cache | 专家缓存 | |
| layer / layer split | 层 / 层切分 | |
| offload | 卸载 | |
| KV cache | KV 缓存 | |
| token | `token` | **不译「词元」** |
| MTP / draft layer | MTP / MTP 草稿层 | |
| draft | 草稿 | `--spec` 语境下保留 `draft` |
| speculative / spec | 投机（spec） | |
| slot | 槽位 | |
| resident | 常驻 | |
| pack | 模型包 | |
| host memory | 宿主内存 | 与「显存」相对 |
| VRAM | 显存 | |
| GTT / pinned / page lock | 原样 / 页锁定 | |
| kernel | 内核 | |
| quantize / quant | 量化 / 保持格式名 | 格式名一律原样 |
| bench / baseline | 基准 / 原始（baseline） | |
| throughput | 吞吐 | |
| prompt | 提示词 | |
| context | 上下文 | |
| engine | 引擎 | |
| setup | setup | 指安装程序时不译 |

## 5. 文体与排版

- 用简体中文、书面语；保持原文的语气（原文直白时可直白，不要添加宣传性修饰）。
- 中英文之间加一个半角空格：`GPU 未容纳的专家`、`见 DETAILS.md`。
- 中文句子用全角标点（，。；：、！？），括注用全角括号 `（）`；英文串、代码、链接保持半角。
- 表格单元格内的数值、单位、代码写法不动，只译说明性文字。
- 行宽不必与英文一致（原文多为 100 列硬换行），按中文习惯自然断行即可；
  但**列表项的层级、序号、缩进**要与原文一致。
- 不添加译者注、不添加「本文译自…」之类的页眉，除原文本身有的提示框外不新增提示框。

## 6. 禁止事项

- 不改写、不简化、不「顺便改进」原文内容；原文写错的地方照译（可在 PR 里单独指出，不在译文里修）。
- 不把 `prefill` 译成「预填充」后又在同篇里写回 `prefill` 造成混用——按第 4 节：首次括注、之后统一用英文原词。
- 不删除或改写原文的 HTML 锚点、`<details>` 折叠块、图片引用。
- 不批量替换术语表以外的词，避免破坏示例串。

## 7. 校验

翻译完成后运行：

```sh
python tools/check_docs_zh.py
```

脚本逐篇比对译文与英文原文的代码块数、表格行数、链接数、标题数、数字个数，检测残留的整段英文，
并检查站内锚点是否可解析、同目录链接的目标文件是否存在。全部通过才算完成。
