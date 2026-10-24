# 属主本地的管理桥，版本 1
<a id="owner-local-management-bridge-version-1"></a>

Strata CLI 与本地 Desktop host 使用同一套已保存 profile 存储，并共用 `strata_control.management.runtime_action()` 执行运行时操作。受信任的 Desktop host 可以调用一次性的 `python -m strata_control.management` 进程，来完成运行时操作、组件/模型检查、经复核的校准以及 profile 编辑。在 stdin 上发送一个 UTF-8 JSON 对象，关闭 stdin，然后在 stdout 上读取一个 JSON 对象。命令失败时以响应的 `exit_code` 退出；正常成功退出 0。诊断信息写到 stderr。输入上限为 64 KiB。

这是**属主本地的进程间 IPC**。它没有 HTTP 监听端口，也不接受推理或 monitor 的 API key。被导入的 Node host 为一个由 host 配置的已保存 profile 提供了有限的、需显式启用的回环 HTTP 接口：用于 Serving 的 `/api/native/runtime`、用于 Monitor 的 `/api/native/telemetry`、用于组件检查的 `/api/native/components`、用于已安装 run config 的 `/api/native/models`、用于详情的 `/api/native/models/inspect?id=MODEL_ID`、用于显式只读 GGUF 预检的 `/api/native/models/import-plan`、用于已保存设置的 `/api/native/profile`，以及用于经复核的校准、有界进度和精确的停止当前测量请求的 `/api/native/calibration/review`、`/execute`、`/job?id=JOB_ID` 与 `/cancel`。Serving、Profile 与 Calibration 在写入变更之前都使用短时、一次性的服务端复核。Node 路由会为动作类 POST 检查本地的 Host/Origin 和一个自定义的 JSON action 头；Python 会重新检查已复核的前置条件。这份本地预览并非经过审计的 Desktop 认证，也不是通用的浏览器管理 API。其余 UI 界面仍在使用原型数据；native 模式会拒绝原型数据的写入和 `/v1` 推理。共享的 Desktop 日志与完整的操作 API 仍属待集成工作。

## 请求与响应
<a id="request-and-response"></a>

```json
{"schema_version":1,"action":"runtime.start.plan","profile":"four-gpu","params":{"shutdown_timeout_s":30}}
```

`schema_version` 必须是整数 `1`。`action` 与 `params` 必须与下表中的某一行完全匹配。`profile` 是一个已保存的小写 profile ID。未知字段、重复的 JSON key、非有限数字、数字字段里的布尔值以及预期之外的 action，都会在操作开始前被拒绝。运行时错误返回 `{"schema_version":1,"ok":false,"error":"...","exit_code":N}`。成功的结果包含 `schema_version:1` 以及属主运行时的结果。记入日志的动作还会返回一个 `operation_id`，UI 和 CLI 都能用 `strata ops result` 查看它。

| Action | 精确的 `params` 键 | 作用 |
| --- | --- | --- |
| `calibration.plan` | 无 | 只读的 decode 扫描预检，包含已保存 profile 的修订号、来源摘要、所选设备与组件状态。不启动引擎，也不写日志条目。 |
| `calibration.job.create` | `job_id`、`action`、`path`、`exclusive_ack`、`if_profile_revision`、`if_source_sha256`、`if_report_sha256` | 在启动 worker 之前，为一次经复核的 Desktop 测量/应用/回滚动作创建一份仅属主可见的凭据（receipt）。measure 场景下 `if_report_sha256` 为 `null`。仅有凭据并不能证明 worker 已启动。 |
| `calibration.job` | `job_id` | 在 Desktop host 重启后读取一份有界凭据。带有该确切 job ID 和经复核的 profile/来源标识的测量报告，可以终结一份在写入报告后仍处于 running 状态的凭据。一份确切的 worker 租约可以证明工作仍在进行；若缺少租约、缺少最终日志结果或缺少匹配的报告，凭据会在一个短暂的启动窗口之后变为 `completion_unknown`。响应中还带有有界的阶段进度与停止请求状态。 |
| `calibration.cancel` | `job_id` | 仅对那个确切的、当前活动的测量 worker 请求停止。仅属主可见的标记由该 worker 轮询，并传入引擎代次（generation）；该请求只确认意图，最终凭据才会说明是否真的取消成功。Apply 与 rollback 在飞行途中无法取消。 |
| `calibration.measure` | `path`、`exclusive_ack`、`if_profile_revision`、`if_source_sha256`；Desktop 还会发送 `job_id` | 要求显式的独占访问确认，拒绝过期的已保存 profile 或来源，然后测量并写入一个新的绝对本地报告路径，并把结果记入日志。在校准锁内重新检查来源。带有 job ID 时，每一次预检与执行失败都会终结匹配的凭据。 |
| `calibration.apply.plan` | `path` | 复核一份已保存报告的新鲜度与适用性，钉住报告正文摘要、profile 修订号与来源摘要。不写入，也不写日志条目。 |
| `calibration.apply` | `path`、`exclusive_ack`、`if_profile_revision`、`if_source_sha256`、`if_report_sha256`；Desktop 还会发送 `job_id` | 重新检查已复核的 profile、来源与报告，然后对候选配置做冒烟测试并应用，同时保留私有的回滚凭据；把结果记入日志。 |
| `calibration.rollback.plan` | `path` | 复核确切已应用的状态与为回滚保存的备份；钉住报告、profile 与当前来源。不写入，也不写日志条目。 |
| `calibration.rollback` | `path`、`exclusive_ack`、`if_profile_revision`、`if_source_sha256`、`if_report_sha256`；Desktop 还会发送 `job_id` | 重新检查已复核的状态，对上一份配置做冒烟测试，并恢复 config/settings；把结果记入日志。 |
| `components.inspect` | `live` | 读取已保存 profile 的已安装组件报告。`live: true` 还会运行有界的 host GPU 探测。检查本身可能成功（`ok: true`）而 `report.ok` 为 false，原因是缺少必需文件或所选显卡。不会创建操作日志条目。 |
| `models.installed` | 无 | 为模型选择列出至多 128 个由 setup 写入的本地 run config；存在性与执行兼容性仍是两件事。要求所配置的已保存 profile 存在。 |
| `models.inspect` | `model_id` | 检查一个已安装的 run config 及其组件报告。模型文件完整性与引擎兼容性仍未经验证。不写日志条目。 |
| `models.import.plan` | `path` | 使用 CLI 的有界预检检查一个绝对本地 GGUF 文件或分片目录。它从不复制、收编或加载模型。不写日志条目。 |
| `profile.inspect` | 无 | 读取已保存的 profile、修订号、来源摘要与生效的服务参数值。不做变更，也不写日志条目。 |
| `profile.change.plan` | `overrides`、`source_config`、`if_revision` | 针对编辑器中已加载的修订号，校验带类型的覆盖项与一个绝对的 run config；展示变更前/后、生效值、当前 profile 修订号与拟议的来源摘要。不写文件，也不重启。 |
| `profile.change` | `overrides`、`source_config`、`if_revision`、`if_source_sha256` | 通过与 CLI 相同的版本化 profile 存储保存已复核的候选。profile 修订号或来源摘要发生变化则拒绝。运行中的实例不受影响；重启由操作者另行复核。Profile 历史会记录旧修订号，但这次保存不是一次运行时日志操作。 |
| `runtime.status` | 无 | 读取属主当前的实例状态。 |
| `runtime.start.plan` | `shutdown_timeout_s` | 校验已保存的 profile 并展示生效的启动参数、`launch.profile_revision` 与 `source_sha256`；不启动，也不写日志条目。 |
| `runtime.start` | `wait_s`、`shutdown_timeout_s`、`if_profile_revision`、`if_source_sha256` | 带着两项已复核的前置条件通过属主 host 启动，并把结果记入日志。Desktop 桥要求这两项前置条件都必须提供。 |
| `runtime.restart.plan` | `allow_source_switch`、`allow_source_migration` | 比较当前持有的启动与已保存的 profile，并报告来源、属主与实例的前置条件。 |
| `runtime.restart` | `wait_s`、`startup_wait_s`、`if_profile_revision`、`if_owner_id`、`if_instance_id`、`allow_source_switch`、`allow_source_migration`、`if_source_sha256` | 带着已复核的 profile、来源、属主与实例前置条件，通过属主 host 排空并重启，并留下日志记录。Desktop 桥要求这四项前置条件都必须提供。 |
| `runtime.verify.plan` | `load` | 读取就绪的属主与实例 ID、模型加载状态，以及本次检查是否会加载模型。它不执行探测，也不创建日志条目。 |
| `runtime.verify` | `wait_s`、`load`、`if_owner_id`、`if_instance_id` | 在属主 host 内重新检查已复核的属主与实例，然后检查 API 标识、启动证明（attestation），并在已加载时检查代次。它可能显式加载一个模型，并会记入日志。Desktop 桥要求这两个 ID 都必须提供；CLI 直接验证时可以省略。 |
| `runtime.stop.plan` | 无 | 读取当前属主 ID，以及可获得时的引擎实例 ID；不排空，也不写日志条目。 |
| `runtime.stop` | `wait_s`、`if_owner_id`、`if_instance_id` | 仅排空并停止已复核的属主与实例；把结果记入日志。Desktop 桥要求提供 `if_owner_id`。 |

时间单位是秒。`if_profile_revision` 是一个 16 字符的小写 profile 修订号；`if_source_sha256` 是一个 64 字符的小写摘要。start 时，从 `runtime.start.plan` 取 `plan.launch.profile_revision` 与 `plan.source_sha256`。restart 时，从 `runtime.restart.plan` 取 `candidate_profile_revision`、`candidate_source_sha256`、`current_owner_id` 与 `current_instance_id`。stop 时，从 `runtime.stop.plan` 取 `plan.owner_id` 与 `plan.instance_id`；实例 ID 在引擎正在启动或启动失败时可能是 `null`。属主与实例 ID 存在时是 32 字符的小写值。restart 计划要求实例已就绪，因此两个标识都存在。桥会在写日志之前拒绝过期的 restart 与 stop 复核；属主 host 在排空前会再检查一次。start 计划若已过期，会尽可能在日志开始前被拒绝；属主 host 在启动引擎前一刻会再次检查同样的前置条件。一份计划不会预留 profile、来源文件或实例。被中断或超时的写入型调用，其结果可能无法确定。重试前请先检查操作日志与运行时状态；绝不要假设进程退出已经撤销了某个副作用。

## Desktop 适配要求
<a id="desktop-adapter-requirements"></a>

Desktop host 现已把一个已配置 profile 的 Start、Stop、Restart 与 Verify 控件映射到这些带类型的动作，并展示返回的操作 ID 与属主状态。来源配置发生变化时，第一次 Restart 复核会拒绝该动作并展示一次显式的来源变更复核；第二次计划只设置由操作者选择的来源切换/迁移许可，并在排空前展示候选路径、摘要与回滚要求。它的 Components 界面读取与 CLI 相同的 `doctor.inspect` 报告；可选的实时显卡探测只在操作者请求后才运行。Profile 展示带类型的覆盖字段与已安装的模型配置选项，在服务端保存一份已复核的候选 90 秒，并通过共享的 profile 存储保存。Models 列出与 CLI 相同的、由 setup 写入的 run config，按需检查一个配置及其组件报告，并提供一次显式的只读 GGUF 文件或目录预检。Calibration 通过与 CLI 相同的 Python 校准模块，复核一次 decode 扫描与已保存的 apply/rollback 候选。浏览器只会收到有界的、所选的设置与发现项；argv 与原始测量报告留在 host 上。一次只允许运行一个 host job。浏览器轮询一个属主日志的 job ID 以获取 launching、running、succeeded、failed、cancelled 或不确定的状态，同时 Python 动作写入有界进度和最终的共享操作记录。Calibration 作业拥有仅属主可见的日志凭据，可在 Desktop host 重启后按 ID 恢复。一份确切的已保存测量报告可以核对一份在写入报告后仍处于 running 的凭据。测量会写入有界的阶段/样本进度；一个实时的确切 worker 租约可以区分"仍在活动"与"成为孤儿的凭据"。操作者可以为一次活动测量请求取消，其引擎代次会收到停止事件。最终凭据只在 worker 退出后才记录 `cancelled`；在重试一次结果不确定的操作之前，请检查 `strata ops runs` 与报告路径。浏览器只收到所选的、有界的检查字段；一套看似合理的模型文件仍需引擎、模型包、完整性与硬件检查。host 在发给浏览器之前会过滤并有界化每一份报告。Verify 使用一次短时、一次性的 90 秒复核，并钉住属主与实例 ID；当模型未加载时会展示一次显式的 Load + Verify 选择。一次成功的单 token 探测并不能确立回答质量或持续吞吐。它必须保留 plan/execute 的复核步骤，为一个 profile 只保留一个生命周期权威，并且绝不从浏览器内容拼接 shell 命令字符串。Harness 界面仍需要它那份共享的带类型映射。校准进度、确切 worker 存活性与测量取消已在本地实现；长时间运行的 GPU 行为、worker 突然死亡后的清理，以及 unmanaged/Windows 引擎的排除，仍需要针对目标环境做验证。当前一次性的桥不提供通用的写租约、流式进度，也无法对含糊的 start/stop/restart 结果做恢复。

在生产使用之前，请在 Windows/Linux 上验证这座桥及其属主传输方式，测试一次真实的引擎启动，并针对同一个 profile 与日志运行 UI/CLI 一致性场景。来源运行的 macOS 假引擎测试检查了：由桥启动的实例能从 CLI 看到并停止；过期的 start、stop 与 restart 复核不会影响另一次启动；格式错误的请求不会创建日志条目。
