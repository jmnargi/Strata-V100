# The `strata` command

This page documents the Terminal: `strata.py` and the `strata_cli/` package. The page covers
engine version 0.1.45.8. Every command below exists in `strata_cli/cli.py` in this tree.

The Terminal starts the server, edits saved settings, manages an owned server instance, reads
telemetry, and runs the tuner. It does not contain engine code. The engine is the `strata` binary in
`engine/`, and the server is `serve/server.py`. Read [HOW_IT_WORKS.md](HOW_IT_WORKS.md) for the
engine, and [SERVER.md](SERVER.md) for the server.

## Run the terminal

A source checkout runs the Terminal as a Python script:

```
python strata.py --version
```

The terminal archive runs it as a command:

```
strata --version
```

`--version` prints one line: the product version, the build number from `CMakeLists.txt`, and the
Python interpreter this terminal uses (from the code, `strata_cli/cli.py`, class `_VersionAction`).

```
strata 0.2.0-beta (build 0.1.45.8; python 3.12.3 at /usr/bin/python3)
```

The product version comes from `strata_cli.PRODUCT_VERSION`. The build number comes from `VERSION`
or from the `project(strata VERSION ...)` line.

### Output and exit rules

These rules apply to every command.

| Rule | Detail |
|---|---|
| `--output human\|json` | `human` (default) prints prose. `json` prints machine-readable JSON; most commands print `{"schema_version":1,"ok":...}`. |
| `--json` | An alias for `--output json`. The commands that declare it are `serve`, `status`, `monitor`, `logs`, `connection`, `config show`, `profile`, `doctor`, `components verify`, `components update`, `components rollback`, `models`, `engine calibrate`, `runtime`, `ops`, `code sessions` and `update`. The other commands take `--output` only, and refuse `--json` as an unknown argument. `install` takes neither: it passes its arguments to `setup.py`. |
| `--quiet` | Suppresses human informational output. Where `main()` captures that output, the last 64 KiB is written to stderr when the command fails. |
| Errors | Human mode writes `strata: <message>` to stderr. JSON mode writes `{"schema_version":1,"ok":false,"error":{"code":N,"message":"..."}}`. |
| Exit codes | See [Exit codes](#exit-codes). |

`serve`, `logs` and `update` also accept `--output jsonl` for one JSON event per line.

### What needs the server packages

`strata.py` defines `SERVER_PACKAGES = ("jinja2", "yaml", "regex", "psutil", "requests", "numpy", "PIL")`.
The preflight checks them for the `serve` command only, before any import of the server. A missing
package is a typed error with exit code 5. The Terminal never installs a package.

The same message names the commands that work without those packages: `status`, `monitor`,
`timeskip` and `tune plan`. `status` and the shared monitor grammar use only the standard library
(from the code, `strata_cli/monitor_commands.py`). The interactive monitor imports the terminal UI
package only when it starts; `--once` prints a snapshot without it. `time_skip` is standard-library
only (from the code, `strata_control/time_skip.py`).

## The model of the tool

Four things, in order of lifetime: a run config, a saved profile, a runtime instance, and the owner
bridge.

### Run config

A run config is a JSON file that `setup.py` writes. It names the engine, the tokenizer, the model
files and the engine arguments. `strata_control/profiles.py:read_run_config` requires an `args`
array of strings, a nonempty `exe`, and a nonempty `tokenizer`. Other keys are optional, for example
`model_name`, `port`, `parallel`, `log`, `api_key`, `mcp_servers` and `before_load`.

The setup installer owns this file. The Terminal reads it and never rewrites it by itself. Only
`strata engine calibrate --apply-result` and `strata tune apply` change it, and each keeps rollback
evidence.

### Saved profile

A saved profile is a small JSON file that belongs to you, not to the installer. It holds an ID, the
absolute path of one run config, and typed serving overrides. The schema is
`{"schema_version":1,"id":...,"source_config":...,"overrides":{...}}`.

| Property | Value |
|---|---|
| Profile folder | `$STRATA_PROFILE_DIR`, else `~/.config/strata/profiles` (POSIX) or `%APPDATA%\Strata\profiles` (Windows) |
| Allowed overrides | `host`, `port`, `parallel`, `batch_groups`, `max_queue`, `queue_timeout`, `offline`, `batch_telemetry` |
| Bounds | `port` 0..65535; `parallel` 1..1024; `batch_groups` `auto` or 1..1024; `max_queue` 0..4096; `queue_timeout` 0.1..600 s; `batch_telemetry` and `offline` true or false |
| Revision | 16 lowercase hex characters, a digest of the profile bytes |
| History | One file per earlier revision in `<profile folder>/.history/<name>/` |

A command that takes `--profile` accepts a saved profile name or a path to a run config JSON file.
A value with `.json`, a path separator, or an absolute path is a path (from the code,
`strata_control/profiles.py:resolve`). A bare lowercase name is a saved profile. A typo in a name
never falls back to a path.

The effective serving value comes from the first source that sets it: the command flag, then the
saved profile override, then the run config, then the environment (`STRATA_BATCH_TELEMETRY` is the
only variable this rule reads), then the default (from the code, `strata_cli/cli.py:_effective`).

Four combinations are refused. `queue_timeout` needs `max_queue`. A non-loopback host needs an API
key in the run config or `STRATA_API_KEY`. A `parallel` override conflicts with `--batch` or `--slots`
in the run config arguments, and a `batch_groups` override conflicts with `--batch-groups` there.
`offline` requires the run config to have no `mcp_servers` and no `before_load`.

### Runtime instance

A runtime instance is one server process that the Terminal started and still holds. The supervisor
in `strata_control/runtime.py` keeps the process handle. It never signals a PID it found in a file;
it compares the PID (and on Windows the creation time) to decide whether the recorded child is
alive. A client finds the instance from its ready file, and the owner holds the control channel.

| State | Folder |
|---|---|
| Runtime state | `$XDG_STATE_HOME/strata/runtime`, else `~/.local/state/strata/runtime`; `%LOCALAPPDATA%\Strata\runtime` on Windows |
| Ready file | Written when the server binds a port. `serve --ready-file` writes one at an absolute path you choose. |

`strata serve` is the foreground form: the server ends with the terminal. `strata runtime start` is
the background form: the server keeps serving until `strata runtime stop`, or until the process
named by `--die-with-pid` ends.

### Owner bridge

The Terminal and a local Desktop host share the saved-profile store and one function,
`strata_control.management.runtime_action()`. A trusted host runs
`python -m strata_control.management`, writes one JSON request on stdin, and reads one JSON reply.
The reply carries `schema_version: 1`, and the process exits with the reply's `exit_code` on
failure. The bridge has no HTTP listener and takes no inference key. Read
[MANAGEMENT_PROTOCOL.md](MANAGEMENT_PROTOCOL.md) for the action table and the required preconditions.

The same module is the reason a reviewed step exists for every write. A plan action reads and
validates, and it writes nothing. An execute action rechecks the reviewed identities before it acts.
Journaled actions return an `operation_id`; `strata ops result` reads it back.

| Store | Location |
|---|---|
| Recipes | `<profile folder>/.ops/<profile>/` |
| Operation journal | `<profile folder>/.ops/<profile>/.runs/` |
| Tuning jobs | `$STRATA_TUNE_DIR`, else `$XDG_STATE_HOME/strata/tuning` |
| Tuning presets from `tune auto` | `$STRATA_PRESET_DIR`, else `<profile folder>/../presets` |
| Time-skip runs | `$STRATA_TIMESKIP_DIR`, else `$XDG_STATE_HOME/strata/time-skip` |
| Monitor connections | `$STRATA_CONNECTION_DIR`, else `~/.config/strata/connections` (POSIX) or `%APPDATA%\Strata\connections` (Windows) |

The code creates the profile, runtime, connection, recipe and journal folders as owner-only, and
checks the owner and the permission bits before use (from the code,
`strata_control/private_files.py`). The tuning and preset folders are created with the default
permissions.

## Command summary

| Command | Purpose |
|---|---|
| [serve](#serve) | Start the server in the foreground. |
| [launch](#launch) | Guided setup, serve and monitor, or the same with flags. |
| [status](#status) | Print one telemetry snapshot. |
| [monitor](#monitor) | Open the read-only terminal monitor, or print one snapshot. |
| [logs](#logs) | Print bounded request activity. |
| [connection](#connection) | Save a monitor endpoint without the credential value. |
| [config](#config) | Show the effective client connection settings. |
| [profile](#profile) | Create, edit and version saved profiles. |
| [doctor](#doctor) | Read-only check of the installed components. |
| [components](#components) | Component inventory, manifest check, backend matrix, engine choice, update. |
| [models](#models) | Inspect installed model run configs, or preview a GGUF set. |
| [engine calibrate](#engine-calibrate) | Measure, review, apply and roll back engine calibration. |
| [install](#install) | Hand over to the installer in a full Strata folder. |
| [runtime](#runtime) | Start, check, stop, restart and clean up an owned instance. |
| [ops](#ops) | Edit and run typed local operator recipes. |
| [timeskip](#timeskip) | Checkpoint a workspace, try a bounded change, and decide from evidence. |
| [tune](#tune) | Bounded engine-configuration search with retained evidence. |
| [presets](#presets) | List, apply, undo and re-check saved tuning presets. |
| [connect](#connect) | Print or write the setup for an app that uses the server. |
| [code](#code) | Coding client for the shared Desktop harness. Not available in this tree. |
| [repair-access](#repair-access) | Take back Strata files that an elevated run locked on Windows. |
| [update](#update) | Install the engine for this PC from the signed feed. |

## serve

Start the existing Strata inference server in the foreground. It ends with this terminal.

```
strata serve --profile CONFIG.json [options]
```

| Option | Default and effect |
|---|---|
| `--profile`, `--config CONFIG.json` | required. A saved profile name or a run config path. |
| `--engine strata\|mock` | `strata` (default). `mock` runs a scripted engine for client tests: no GPU and no model. |
| `--host ADDR` | `127.0.0.1` (default). A non-loopback host needs an API key in the run config or `STRATA_API_KEY`. |
| `--port N` | `8095` (default). Range 0..65535. |
| `--parallel N` | not fixed in code. Range 1..1024. The run config decides when the flag is absent. |
| `--batch-groups VALUE` | not fixed in code. `auto` or 1..1024. |
| `--max-queue N` | not fixed in code. Range 0..4096. Maximum pending calls beyond the effective slots. |
| `--queue-timeout S` | not fixed in code. Range 0.1..600 s. Requires `--max-queue`. |
| `--batch-telemetry` / `--no-batch-telemetry` | off (default). Samples engine decode-window rows for the monitor. `STRATA_BATCH_TELEMETRY=1` also turns it on. |
| `--offline` | off (default). Disables remote image fetches and MCP/tool subprocesses. |
| `--ready-file PATH` | none. Writes a new instance record at an absolute path. |
| `--shutdown-timeout S` | `30` (default). Range 0..300 s. Time to drain owned calls on stop. |
| `--plan` | off (default). Validates and prints the launch, and starts nothing. |
| `--output human\|json\|jsonl` | `human` (default). `json` needs `--plan`. `jsonl` is the foreground lifecycle and cannot be combined with `--plan`. |

Read [SERVER.md](SERVER.md) for the server options and the request format, and
[ENGINE_FLAGS.md](ENGINE_FLAGS.md) for the engine flags that the run config carries.

What it changes: it starts the server and the engine. It writes the ready file when `--ready-file`
is given. It changes no configuration file.

```
strata serve --profile four-gpu --plan --output json
```

| Exit code | Meaning |
|---|---|
| 0 | The server stopped. Owned requests were drained. |
| 2 | An option is outside its range, or the option combination is not allowed. |
| 5 | The API key in the run config is invalid. |
| 6 | Startup failed or the server raised an unexpected error. |
| 8 | The shutdown drain deadline expired; owned requests were cancelled. |
| 130 | Interrupted. |

## launch

Set up a model, tune it, start serving and watch it in one place. With no flags it opens the guided
screens. With flags it does the same without a question.

```
strata launch [options]
```

| Group | Options and defaults |
|---|---|
| What to run | `--preset NAME`; `--model NAME`; `--family NAME`; `--model-path PATH`; `--data-dir DIR`; `--context 32k\|64k\|128k\|256k\|N`; `--kv int8` (default), `q4_0`, `k8v4`, `fp16`; `--vision off\|on\|gpu\|cpu` (default on when the model and the engine can read pictures); `--parallel 1..8` (default: fit to the card); `--gpu LIST`; `--tune quick\|thorough\|skip` (default `skip` here); `--tune-allow-small`; `--set KEY=VALUE` (repeatable) |
| Who can use it | `--access local\|lan\|cloudflare\|tailscale`; `--port N` (`8080` default); `--api-key KEY`; `--api-key-env NAME`; `--new-key` |
| Running | `--save-preset NAME`; `--no-tui`; `--open`; `--yes`; `--plan`; `--feed URL\|DIR`; `--output human\|json` |
| Manage | `--list-presets`; `--delete-preset NAME`; `--status`; `--stop`; `--show-key` |

What it changes: it can download and prepare a model, write `strata-*.json` run configs, save a
preset, and start the server with the monitor. A preset is shared with the Desktop app's profiles
and with the tuning presets.

```
strata launch --model q2_0 --context 64k --parallel 4 --no-tui --port 8080
```

Read [INSTALL.md](INSTALL.md) for the first setup, and [MODELS.md](MODELS.md) for the model choices.

| Exit code | Meaning |
|---|---|
| 0 | Done. With no engine for this PC it still exits 0 and reports `"serving": false` (from the code, `strata_cli/launch/runner.py`). |
| 2 | A flag, a preset or a model name is wrong. |
| 5 | The engine or the model files cannot be used. |
| 6 | The server could not start. |
| 130 | Stopped at a question or by Ctrl+C. |

## status

Print one read-only telemetry snapshot.

```
strata status [connection options]
```

| Option | Default and effect |
|---|---|
| `--url URL` | `http://127.0.0.1:8095` (default). The server root. A remote URL needs HTTPS. |
| `--connection NAME` | none. A saved connection from `strata connection`. |
| `--credential-env NAME` | `STRATA_API_KEY` (default). The environment variable that holds the key. |
| `--timeout S` | `3.0` (default). Range above 0 and at most 30 s. |

The client reads `/v1/telemetry` first. If that endpoint is absent, it tries `/metrics`; if that is
also absent, it reads `/v1/status` (from the code, `strata_cli/client.py:snapshot`). A response
larger than 2 MiB is refused. `--output json` prints the normalized snapshot.

What it changes: nothing. The command is read only.

```
strata status --connection lab --output json
```

## monitor

Open the read-only terminal monitor, or print one snapshot with `--once`. The monitor has seven
views: Overview through Logs and Connect. `strata launch` opens the same monitor.

```
strata monitor [connection options] [options]
```

It takes the same connection options as `status`, plus:

| Option | Default and effect |
|---|---|
| `--once` | off (default). Prints one snapshot and returns, without the terminal UI. |
| `--theme auto\|dark\|light\|terminal` | `auto` (default). |
| `--color auto\|always\|never` | `auto` (default). |
| `--glyphs auto\|unicode\|braille\|ascii` | `auto` (default). |
| `--interval S` | `1.0` (default). Range 0.5..30 s. |
| `--requests N` | `12` (default). Range 1..50. |

The interactive monitor needs a terminal. JSON output needs `--once`. `q` detaches; the command
starts nothing, so it stops nothing.

What it changes: nothing.

```
strata monitor --once --url http://127.0.0.1:8095
```

## logs

Print bounded, redacted request activity. The feed is read only.

```
strata logs [connection options] [options]
```

| Option | Default and effect |
|---|---|
| `--source runtime\|harness` | `runtime` (default). `harness` returns exit code 5: shared local harness activity is unavailable. |
| `--follow` | off (default). Polls for new transitions until Ctrl+C. |
| `--interval S` | `2.0` (default). Range 0.5..30 s. |
| `--output human\|json\|jsonl` | `human` (default). `--json` is one-shot only; `jsonl` requires `--follow`. |

A gap in the sequence is printed as a gap, never filled (from the code, `strata_cli/logs.py`).

What it changes: nothing.

```
strata logs --follow --output jsonl
```

## connection

Save a monitor endpoint. The store keeps the name of the credential environment variable, never
the credential value.

```
strata connection {create,show,list,set} ...
```

| Subcommand | Usage and options |
|---|---|
| `create` | `strata connection create NAME --url URL [--prefix PATH] [--credential-env NAME]`. `--prefix` is `""` (default). `--credential-env` is `STRATA_TELEMETRY_KEY` (default). |
| `show` | `strata connection show NAME`. Shows the saved connection without its credential. |
| `list` | `strata connection list`. |
| `set` | `strata connection set NAME KEY=VALUE [--plan] [--if-revision REVISION]`. Edits one URL, prefix or credential reference. `--if-revision` fails if the connection changed since the preview. |

A connection name uses lowercase `a-z`, `0-9`, `_` or `-` and starts with a letter. The folder must
be owner-only.

What it changes: it writes one JSON file in the connections folder. `--plan` writes nothing.

```
strata connection create lab --url https://gpu.example.net --prefix /strata
```

## config

Show the effective client connection settings and where each value came from.

```
strata config show [--effective] [connection options]
```

`--effective` is the default behavior. The output names the origin of each value: `flag`,
`environment`, `connection` or `default` (from the code, `strata_cli/connections.py:resolve`).

What it changes: nothing.

```
strata config show --json
```

## profile

Edit saved serving overrides outside the run config that setup writes. An installer update cannot
erase these values.

```
strata profile {create,list,show,validate,set,use-config,edit,history,rollback} ...
```

| Subcommand | Usage and behavior |
|---|---|
| `create` | `strata profile create NAME --from-config CONFIG.json`. Saves a profile with no overrides. It refuses an existing name. It writes `<profile folder>/<name>.json`. |
| `list` | `strata profile list`. One line per profile: ID and revision, or the read error. |
| `show` | `strata profile show NAME [--effective]`. `--effective` adds the launch plan: effective values, origins, server arguments and engine arguments (secrets redacted). |
| `validate` | `strata profile validate NAME`. Checks the profile and the run config it references, and prints the launch plan. Exit code 0 means valid, 5 means the run config is missing or unreadable, and 2 means an override or the profile schema is invalid. |
| `set` | `strata profile set NAME ASSIGNMENT... [--plan]`. Assignments are `key=value` with the keys and bounds in [Saved profile](#saved-profile). It saves a new revision and keeps the previous one in history. |
| `use-config` | `strata profile use-config NAME --from-config CONFIG.json [--plan] [--if-revision REVISION]`. Points the profile at another run config. A running instance keeps its launch. |
| `edit` | `strata profile edit NAME`. Opens a temporary JSON copy in `$VISUAL` or `$EDITOR` and saves it after validation. Exit code 6 means the editor exited with an error; the draft stays on disk. |
| `history` | `strata profile history NAME`. Lists revisions, newest first, and marks the current one. |
| `rollback` | `strata profile rollback NAME --to REVISION`. Restores a revision from history. The target is a full 16-character revision. Running instances are unchanged. |

`profile show --effective` and `profile validate` build the same launch plan as `serve`, so they
read `serve/server.py` and `serve/ready.py` (from the code, `strata_cli/cli.py:_launch_plan`).

What it changes: the profile JSON and its history. `--plan` writes nothing.

```
strata profile set four-gpu parallel=8 port=8096 --plan
strata profile validate four-gpu
```

| Exit code | Meaning |
|---|---|
| 0 | Done. |
| 2 | An assignment or a name is invalid, or a plan precondition failed before work. |
| 5 | The profile or its run config is missing or invalid. |
| 6 | The profile or the run config changed during a reviewed save; inspect it again. |

## doctor

Read-only check of the installed engine components for one profile.

```
strata doctor [--profile NAME|CONFIG.json] [--components] [--live]
```

| Option | Default and effect |
|---|---|
| `--profile` | none. A saved profile name or a run config path. Without it the check covers the Terminal's own modules only, and the report says so. |
| `--components` | on (default). Includes the component inventory. |
| `--live` | off (default). Queries the host GPUs and compares the selected device indices. It needs a profile. |

The report names each component, its path or reason, and the issues. `--live` adds one block per
card: name, memory in MiB, driver version and UUID. Read the component rules in
[HOW_IT_WORKS.md](HOW_IT_WORKS.md) and the file list in [INSTALL.md](INSTALL.md).

What it changes: nothing.

```
strata doctor --profile four-gpu --live
```

| Exit code | Meaning |
|---|---|
| 0 | Every required component is present. |
| 5 | A required component is missing or unusable. |

## components

Read-only component inventory, manifest check, backend matrix and engine choice. `components update`
and `components rollback` change an installation; both are the same code as `strata update`.

```
strata components {verify,plan,backends,engine,update,rollback} ...
```

| Subcommand | Usage and behavior |
|---|---|
| `verify` | `strata components verify --profile NAME [--live]`. Checks the required files for one installed profile. Same report as `doctor --profile`. Exit code 0 or 5. |
| `plan` | `strata components plan MANIFEST [--backend FAMILY] [--target KEY]`. Reads a component manifest and says whether this machine may use it. `--target` overrides the platform key, to check another machine's manifest here. Exit code 2 when the manifest path does not exist, 6 when the manifest is invalid; otherwise 0, also when the combination is unusable (the answer is in `usable`). |
| `backends` | `strata components backends`. The OS and backend combinations the project commits to, and which one this machine is. Exit code 0. |
| `engine` | `strata components engine [--backend FAMILY] [--components-dir DIR] [--full]`. Which engine package in the local component folder this PC would start. `--full` hashes every file, not only the entrypoint. This command never downloads. Exit code 0 when a usable engine is present, 5 when one is missing, 2 for a manifest problem. |
| `update` | The same as `strata update`. See [update](#update). |
| `rollback` | `strata components rollback`. Returns to the previous engine version that the last update kept. Exit code 0 on success. |

What it changes: `plan`, `backends`, `engine` and `verify` change nothing. `update` and `rollback`
install a version and keep the version they replace.

```
strata components engine --backend cuda --full
strata components plan ~/.local/share/strata/components/engine/0.1.45-linux-x86_64-cuda/engine-component.json
```

## models

Inspect the run configs that setup wrote, or preview an existing GGUF file set. This command never
adopts, copies or loads a model.

```
strata models {list,inspect,import} ...
```

| Subcommand | Usage and options |
|---|---|
| `list` | `strata models list [--installed\|--served] [--install-dir DIR] [--url URL] [--connection NAME] [--credential-env NAME] [--timeout S]`. `--installed` and `--served` exclude each other. The default lists local installed configs. `--served` asks one server which model it serves now; the connection options require it. `--install-dir` applies to local listing only. |
| `inspect` | `strata models inspect MODEL_ID [--install-dir DIR]`. Reports the component status for one run config. Engine compatibility stays unverified. |
| `import` | `strata models import PATH --plan`. `--plan` is required, so the command only inspects. It accepts one `.gguf` file or a folder of shards. It reports the file set status, the candidate selection and the adoption status, and it changes nothing. |

Discovery is read only and bounded: at most 128 configs of at most 1 MiB each (from the code,
`strata_control/models.py`). Read [MODELS.md](MODELS.md) for the model families and
[QUANTS.md](QUANTS.md) for the quantizations.

What it changes: nothing.

```
strata models list --install-dir /opt/strata
strata models import /data/models/qwen-q2_0.gguf --plan
```

| Exit code | Meaning |
|---|---|
| 0 | Done. |
| 2 | An option combination is wrong, or the model ID is malformed. |
| 5 | A run config or GGUF file set is unreadable or unusable (import preflight issues). |

## engine calibrate

Measure, review, apply and roll back engine calibration for one saved profile. The calibration
suite is `legacy-decode-v1`: it measures decode speed, and it is not a multi-GPU serving benchmark.

```
strata engine calibrate --profile NAME (--plan | --measure | --review-result REPORT.json |
                         --apply-result REPORT.json | --review-rollback REPORT.json |
                         --rollback-result REPORT.json) [options]
```

| Option | Default and effect |
|---|---|
| `--profile NAME` | required. The saved profile to calibrate. A bare run config path keeps the older behavior without a journal entry. |
| action | exactly one of the six flags above is required. |
| `--out REPORT.json` | none. The new report path. It is required with `--measure` and refused for every other action. |
| `--exclusive` | off (default). Asserts that every model instance that uses these devices is stopped. Required with `--measure`, `--apply-result` and `--rollback-result`. |

The report binds a candidate to the run config, the engine binary and the local hardware. A report
is at most 2 MiB and is fresh for 30 days (from the code, `strata_control/calibration.py`). Only
`--pcie-frac`, `--spec-min-p`, `--pool-workers`, `--adapt-every`, `--adapt-swaps` and
`--adapt-decay` can be applied. An apply smoke-tests the candidate and keeps a private rollback
copy. A saved profile writes a journal entry that `strata ops result` can read.

What it changes: `--measure` writes one report. `--apply-result` writes the run config and rollback
evidence. `--rollback-result` restores the earlier configuration. `--plan`, `--review-result` and
`--review-rollback` write nothing.

```
strata engine calibrate --profile four-gpu --measure --exclusive --out /tmp/cal-2026-11.json
strata engine calibrate --profile four-gpu --review-result /tmp/cal-2026-11.json
```

| Exit code | Meaning |
|---|---|
| 0 | Done, or the reviewed report is applicable. |
| 2 | An action flag or a precondition flag is wrong. |
| 5 | A required component is missing, or the report is invalid or unsupported. |
| 6 | The reviewed report is not applicable, or the measurement failed. |

## install

The explicit installer. It hands over to `setup.py` in a full Strata folder, which is where pip, the
engine and the models come from. Starting and serving never install anything.

```
strata install [--update] [setup.py options...]
```

| Option | Default and effect |
|---|---|
| `--update` | off (default). Updates an existing install (what `UPDATE.bat` and `update.sh` run) instead of setting one up. |
| setup.py options | passed through unchanged, e.g. `--family qwen --model Q2_0 --yes`. |

The installer is read from `$STRATA_INSTALL_DIR`, else the Terminal's own folder. The Terminal
archive has no `setup.py`; it returns exit code 5 and names the two alternatives: a full Strata
folder, or `strata update`.

What it changes: whatever `setup.py` writes. Read [INSTALL.md](INSTALL.md) for the steps and
[SAFETY.md](SAFETY.md) for what is downloaded.

```
strata install --family qwen --model Q2_0 --yes
```

| Exit code | Meaning |
|---|---|
| 5 | This folder has no `setup.py`. |
| other | The exit code of `setup.py` (0 on success). |

## runtime

Manage an owned local server instance. `runtime start` starts a background service: the engine
keeps serving after this terminal closes and until `runtime stop`, or until the process named by
`--die-with-pid` ends.

```
strata runtime {start,status,verify,stop,cleanup,restart} ...
```

| Subcommand | Usage and options |
|---|---|
| `start` | `strata runtime start --profile NAME [--die-with-pid PID] [--plan] [--wait S] [--shutdown-timeout S]`. `--wait` is `300` (default), 0..600 s. `--shutdown-timeout` is `30` (default), 0..300 s. `--plan` prints the launch and starts nothing. |
| `status` | `strata runtime status --profile NAME`. Reads the owner's current instance state. |
| `verify` | `strata runtime verify --profile NAME [--wait S] [--load] [--plan] [--if-owner-id ID] [--if-instance-id ID]`. Rechecks the owned API and a one-token generation. `--wait` is `300` (default), 1..600 s. `--load` loads an unloaded model for the generation check. The two `--if-*` flags require the owner and the instance shown by `verify --plan`. |
| `stop` | `strata runtime stop --profile NAME [--plan] [--wait S] [--if-owner-id ID] [--if-instance-id ID]`. Drains and stops the owned instance. `--wait` is `330` (default), 1..600 s. |
| `cleanup` | `strata runtime cleanup [--profile NAME]`. Ends servers and engines that Strata started whose owner is gone. Each process is checked by PID, start time and command line first. A running instance is never touched. |
| `restart` | `strata runtime restart --profile NAME [--plan] [--allow-source-switch] [--allow-source-migration] [--wait S] [--startup-wait S] [--if-profile-revision REV] [--if-owner-id ID] [--if-instance-id ID] [--if-source-sha256 SHA256]`. `--wait` is `1000` (default), 1..1800 s. `--startup-wait` is `300` (default), 1..600 s. `--allow-source-switch` permits a new source config when the old file stays available. `--allow-source-migration` permits edited run config bytes with a private rollback snapshot. |

The owner holds the process handle. It never sends a signal to a PID it found in a file. The plan
and precondition flags exist so a reviewed action cannot hit a different instance. Read
[MANAGEMENT_PROTOCOL.md](MANAGEMENT_PROTOCOL.md) for the identity fields and their formats.

What it changes: `start` and `restart` start a server and write an instance record. `stop` drains
and ends it. `cleanup` ends leftovers. `status` and `--plan` change nothing. Journaled actions write
an operation record.

```
strata runtime start --profile four-gpu --plan
strata runtime start --profile four-gpu --wait 300
strata runtime verify --profile four-gpu --load
```

| Exit code | Meaning |
|---|---|
| 0 | Done; the instance is in the requested state. |
| 2 | An option or a precondition format is invalid. |
| 4 | The endpoint refused the credential during verification. |
| 5 | The profile preflight failed; a component or the run config is unusable. |
| 6 | The owner or the instance changed after the plan, the instance is not owned or not ready, or the action failed. |
| 8 | A wait deadline expired (startup wait, or the stop drain deadline). |

## ops

Edit and run typed local operator recipes for one saved profile. A recipe contains no command text;
each kind routes to an existing Strata operation.

```
strata ops {init,list,create,show,edit,set,history,rollback,plan,run,runs,result} --profile NAME ...
```

Every subcommand requires `--profile NAME`.

| Subcommand | Usage |
|---|---|
| `init` | `strata ops init --profile NAME`. Creates the three default recipes: `restart` (`runtime.restart`), `calibrate` (`engine.calibrate.measure`) and `check` (`components.verify`). |
| `list` | `strata ops list --profile NAME`. |
| `create` | `strata ops create NAME --profile NAME --kind KIND`, where `KIND` is `runtime.restart`, `engine.calibrate.measure` or `components.verify`. |
| `show` | `strata ops show NAME --profile NAME`. |
| `edit` | `strata ops edit NAME --profile NAME`. Opens a temporary JSON copy in `$VISUAL` or `$EDITOR`. |
| `set` | `strata ops set NAME --profile NAME KEY=VALUE [--plan]`. |
| `history` | `strata ops history NAME --profile NAME`. |
| `rollback` | `strata ops rollback NAME --profile NAME --to REVISION`. |
| `plan` | `strata ops plan NAME --profile NAME`. Prints the operation the recipe would run. |
| `run` | `strata ops run NAME --profile NAME [--exclusive] [--if-revision REV] [--if-profile-revision REV] [--if-owner-id ID] [--if-instance-id ID] [--if-source-sha256 SHA256]`. `--exclusive` applies to a calibration recipe only. |
| `runs` | `strata ops runs --profile NAME [--limit N]`. `--limit` is `20` (default). |
| `result` | `strata ops result OPERATION_ID --profile NAME`. Reads one journal record. |

Recipes use the profile folder: `<profile folder>/.ops/<profile>/`, and the journal is in `.runs/`
below it. A journal record stays `completion_unknown` when the caller died; it is not proof that a
worker runs.

What it changes: `init`, `create`, `edit`, `set` and `rollback` write recipe files. `run` performs the
operation and writes a journal record. `list`, `show`, `history`, `plan`, `runs` and `result` write
nothing.

```
strata ops init --profile four-gpu
strata ops run restart --profile four-gpu --if-owner-id 8f14e45fceea167a5a36dedd4bea2543
```

| Exit code | Meaning |
|---|---|
| 0 | Done. |
| 2 | A name, a revision or an option is invalid. |
| 5 | The plan is not usable (a missing component, or a failed preflight). |
| 6 | The run failed, or the recipe changed since the preview. |

## timeskip

Retain a good result, try a bounded change, check it, and decide from evidence. No timer decides
anything: a check that did not run is `unknown`, never `passed`.

```
strata timeskip {run,status,list,apply} ...
```

| Subcommand | Usage and options |
|---|---|
| `run` | `strata timeskip run --workspace DIR --prompt TEXT [--attempts N] [--revise-command CMD] [--state-dir DIR]`. `--attempts` is `2` (default); 0 checkpoints and checks the baseline only. `--revise-command` performs one revision attempt with the candidate directory as the working directory, the prompt in `STRATA_TIMESKIP_PROMPT`, and the attempt number in `STRATA_TIMESKIP_ATTEMPT`. Without it, the run reports `inconclusive` instead of an improvement. |
| `status` | `strata timeskip status RUN_ID [--state-dir DIR]`. Prints the durable state, the checkpoints with their check outcomes, the attempts and the decision. |
| `list` | `strata timeskip list [--state-dir DIR]`. |
| `apply` | `strata timeskip apply RUN_ID [--workspace DIR] [--attempt N] [--dry-run] [--restore] [--state-dir DIR]`. Refuses if the destination changed. `--restore` puts the original back. `--attempt` selects one revision instead of the retained checkpoint. |

The checks come from `.strata/checks.json` in the workspace. Without that file the run prints a
warning, checkpoints, and ends inconclusive. The state folder is `--state-dir`, else
`$STRATA_TIMESKIP_DIR`, else `$XDG_STATE_HOME/strata/time-skip` (`%LOCALAPPDATA%\Strata\time-skip`
on Windows).

What it changes: `run` writes checkpoints and candidate directories in the state folder, and it
changes nothing in the workspace. `apply` writes the workspace and keeps a rollback copy.
`--dry-run` reports the change and writes nothing.

```
strata timeskip run --workspace ~/project --prompt "make the parser reject empty input" --attempts 2
strata timeskip apply 4f2a1c8e9b3d --dry-run
```

| Exit code | Meaning |
|---|---|
| 0 | Done. |
| 2 | The workspace, the run ID, the checks file or the request is unusable. |
| 6 | The run failed or was interrupted, or the apply met a conflict or did not complete. |

## tune

Bounded engine-configuration search with retained evidence. The search returns a recommendation; it
never changes a profile by itself.

```
strata tune {plan,run,apply,rollback,status,list,auto} ...
```

| Subcommand | Usage and options |
|---|---|
| `plan` | `strata tune plan --engine PATH --backend FAMILY --model ID [--objective generation\|prefill\|latency] [--trials N] [--time S] [--repeats N] [--hold DIM]`. Shows every dimension the sweep would search and why the rest were excluded. It measures nothing. |
| `run` | `strata tune run --engine PATH --backend FAMILY --model ID [options]`. Measures the baseline and the candidates, and writes a job record. |
| `apply` | `strata tune apply JOB_ID [--config RUN_CONFIG.json] [--model-path PATH] [--context N]`. Applies a reviewed recommendation, after the identity checks and a diff. |
| `rollback` | `strata tune rollback APPLICATION_ID`. Restores the configuration bytes from before an apply. |
| `status` | `strata tune status JOB_ID`. Reads a job's durable state and report. |
| `list` | `strata tune list`. |
| `auto` | `strata tune auto TARGET [options]`. The real tuner: measures your configuration, tries one-knob changes, and saves a gain as a preset. |

`--engine` is a path, or `component:strata-engine` for the engine package installed for this PC.
`--backend` is one of `metal`, `cuda`, `hip`, `sycl`, `cpu`.

`plan` takes `--objective`, `--trials`, `--time`, `--repeats` and `--hold`. `run` adds the workload,
identity and state options below. Both commands require `--engine`, `--backend` and `--model`.

| Option | Default and effect |
|---|---|
| `--objective generation\|prefill\|latency` | `generation` (default). The metric the sweep ranks. |
| `--trials N` | `16` (default). The bounded number of candidates. |
| `--time S` | `1800.0` (default). The total-time bound that the plan reports. |
| `--repeats N` | `3` (default). Paired repeats per candidate. |
| `--hold DIM` | none. Holds one dimension constant instead of searching it; repeatable. |
| `--prompt-tokens N` | `512` (default). Fixed for the whole sweep. |
| `--max-new N` | `128` (default). Fixed for the whole sweep. |
| `--mode protocol\|api` | `protocol` (default). `protocol` measures the engine's own serve protocol. `api` drives the real HTTP API. |
| `--api-prompt TEXT` | `In one short sentence, what is a hash table?` (default). |
| `--api-expect REGEX` | none. The answer must match, or the run is not counted as correct. Required with `--mode api`, together with `--config`. |
| `--api-port N` | `8123` (default). |
| `--config PATH` | none. The run config the `api` worker starts from. |
| `--boot-timeout S` | `600.0` (default). Time for one candidate to become ready. |
| `--request-timeout S` | `600.0` (default). Time for one measured request. |
| `--model-path PATH` | none. Adds the model's content digest to the record. Without it the record is not comparable, and `apply` refuses it. |
| `--context N` | none. The context the measurements were taken at. |
| `--full-digests` | off (default). Hashes the model and the engine completely instead of sampling. |
| `--state-dir DIR` | none. The state folder; see [Evidence](#evidence-of-a-sweep). |

`tune auto` takes its own options:

| Option | Default and effect |
|---|---|
| `TARGET` | required. A saved profile name or a run config path. |
| `--mode quick\|thorough` | `quick` (default). The help text says quick takes about 20 minutes and thorough up to 2 hours with a 5-pair confirmation. |
| `--goal speed\|room\|room:MiB` | `speed` (default). `room` leaves memory free; `room:MiB` leaves that many MiB free. |
| `--leave-ram PERCENT` | `0` (default). Range 0..50. A nonzero value switches the goal to `room`. |
| `--use chat\|coding` | `chat` (default). `coding` also judges long-prompt speed. |
| `--allow-small-differences` | off (default). Also tries arms of the `placement` class. |
| `--allow-not-repeatable` | off (default). Also tries arms of the `nonrepeatable` class. |
| `--name NAME` | none. The preset name; the default is model, context and goal. |
| `--time MINUTES` | none. The time to spend, in minutes. |
| `--no-save` | off (default). Measures and reports, and saves no preset. |
| `--gpu-lock PATH` | `$STRATA_TUNE_GPU_LOCK`. A lock file held around every engine run. |
| `--no-quiet-check` | off (default). Skips the wait for other engines and a busy CPU. |
| `--serve-dir DIR` | the Terminal's own folder. The folder that holds `serve/`. |
| `--work-dir DIR` | a temporary folder. Where the run's configs and logs go. |
| `--status-file PATH` | none. Progress as JSON, rewritten at every step. |
| `--boot-timeout S` | `900.0` (default). |

What it changes: `plan` writes nothing. `run` writes a job record. `apply` writes the run config and
keeps the previous bytes. `rollback` restores them and marks the application rolled back. `auto`
writes a preset when it finds a gain, unless `--no-save` is given.

```
strata tune plan --engine engine/strata --backend cuda --model qwen3.8-flash-next --trials 16
strata tune run --engine engine/strata --backend cuda --model qwen3.8-flash-next --model-path /data/q2_0.gguf
strata tune apply 8f14e45fceea167a --config strata-qwen.json
```

| Exit code | Meaning |
|---|---|
| 0 | Done. |
| 2 | A required option is absent, the engine binary is absent, or the job, config or identity cannot be matched. |
| 5 | No engine at `--engine` (`tune auto`), or a preset cannot be used. |
| 6 | An apply was refused (`ApplyError` default), or the sweep failed. |
| 130 | The sweep was cancelled. |
| 404, 409 | `tune rollback` found no application record, or the config changed after the apply. |

Read [TUNING.md](TUNING.md) for the measured knobs and their gains on named hardware.

### The arm model

A candidate is one knob changed from your own configuration (Rule 1, from the code,
`strata_control/tune_knobs.py`). The sweep never replaces your settings with the first value of every
dimension. A boolean flag is present or absent; `--kv-grow 0` is not a value (Rule 3). A value
outside the bounds, or on the never list, is not tried (Rule 2).

The per-request settings are tried before any restart, because they need none. A per-request win
needs at least 3% in every round (from the code, `strata_control/tune_knobs.py:PER_REQUEST`).

The machine class decides the order and the list. The classes are A to G: A is RAM-bound resident
mode, B is a card with at least 24 GB, C is a card below 24 GB, D and E are the Qwen3.6 hybrid cases,
F is a CPU with performance and efficiency cores, and G is Intel Arc under SYCL. `quick` keeps the
first arms the time budget allows. `thorough` keeps all arms, and the flat knobs come last.

### Answer classes

Each arm declares one answer class. The class says what the setting does to the answers.

| Class | Meaning for you | Notice text (from the code) |
|---|---|---|
| `exact` | The answers are byte-identical to the standard setup. The tuner may pick it alone. | `Answers are identical to the standard setup.` |
| `placement` | The math is the same, but the CPU/GPU split changes, so near-ties can flip. The tuner needs your tick. | `Answers may differ slightly from the standard setup.` |
| `nonrepeatable` | The answers are not bit-exact from run to run. The tuner needs your tick. | `Answers are not repeatable run to run.` |
| `quality` | The setting changes the numbers on purpose. The tuner never picks it. | `Includes a quality setting that changes the numbers.` |

`strata tune auto --allow-small-differences` permits the `placement` class.
`--allow-not-repeatable` permits the `nonrepeatable` class. Nothing selects `quality`
(from the code, `tune_knobs.allowed`).

On some machines the standard setup is not repeatable. The code records that the RTX 3060 in
resident-RAM mode and the Arc A750 gave 2 and 3 identical gate prompts out of 10 between two runs of
the default. There the `exact` notice changes to: `Answers can differ from run to run on this PC
even with the standard setup; this setting does not add to that.` (from the code,
`strata_control/tune_knobs.py`).

### Bounds

A value outside these bounds is not tried, and a preset that carries one is refused.

| Flag | Safe values |
|---|---|
| `--spec` | 3 or 4 |
| `--spec-min-p` | 0.3 to 0.7 |
| `--mtp` | 1, 2 or 3 (Qwen3.6 only; the Flash-Next `--mtp` is a path) |
| `--pcie-frac` | 0.0 to 0.55 |
| `--adapt-every` | 1 to 4 |
| `--adapt-swaps` | 80 to 160 |
| `--adapt-decay` | 0.7 to 0.97 |
| `--hybrid-rows` | 64 to 256 |
| `--host-core` | `first` or `last` |
| `--kv-resident` | 32768, 49152 or 65536 |
| `--prefill` | `auto`, `auto:16384`, `auto:32768`, 2048 or 4096 |

### The never-pick list

These settings are refused with a reason. The table is the rule set from
`strata_control/tune_knobs.py:NEVER`.

| Setting | Condition | Reason (from the code) |
|---|---|---|
| `STRATA_SH_STREAM` | gfx12 cards | it halves decode speed on this graphics card |
| `STRATA_RDNA2_SGEMM=all` | gfx12 cards | it makes long prompts five times slower |
| `STRATA_SELECT_SIMT` | always | its gain is below the 3% margin |
| `STRATA_HIP_WMMA` | always | it changes the numbers (a quality setting) |
| `--pool-affinity auto\|p-cores` | a CPU with performance and efficiency cores | it made answers 47% slower |
| `--pool-workers` above the engine default | always | more helper threads than the engine chose made it 36% slower |
| `--ple-io ram\|mmap` | RAM-bound machine | it needs the table in memory, which this PC does not have to spare (-61% speed) |
| `--adapt-every` at or above 100000 | always | turning the expert re-balancing off cost 3% to 44% |
| `--kv-resident` below the context | unless at 128k or more, where 49152 and 65536 were flat or better | it made answers 22% to 34% slower and changed them |
| `--pcie-frac` at or above 0.75 | a card below 24 GB | a high PCIe share made answers 14% to 37% slower |
| `--prefill` fixed below `auto` | Flash-Next, or a Qwen3.6 chunk of 512 or less | smaller prompt chunks made long prompts 11% to 75% slower |
| `--expert-vram` without `--prefill` | Qwen3.6 | without a matching prompt chunk it made prompts 44% slower |
| `--spec` at or above 6 | always | wider speculative windows were 4% to 16% slower |
| `--spec` at or below 2 | Flash-Next | a 2-token window was 4% to 11% slower |
| `--lookup-chain` | always | it was 3% to 8.5% slower |
| `--vram-reserve-mib` | a card of 8400 MiB or less | this card cannot start with a reserve |
| `--expert-cache` zero or very small | always | turning the GPU expert cache off or very small costs speed |
| `--mtp` at 0 or at 4 and above | Qwen3.6 | that draft window was 10% to 28% slower |

Each reason follows the code's own wording in `NEVER`. The measurement behind each row is in
[DETAILS.md](DETAILS.md); the gains that the tuner may pick are in [TUNING.md](TUNING.md).

### Evidence of a sweep

`strata tune run` writes durable evidence in the state folder:

| Item | Detail |
|---|---|
| Job record | `<state>/<job id>/job.json`. It holds the plan, every candidate, the events, and the report. |
| Exclusive lease | `<state>/<job id>/tuning.lease`, an `O_EXCL` marker. A crashed sweep leaves a stale marker that the next run reports. |
| History | `<state>/history`. A previous sweep there is labelled comparable or not, with the reason. |
| Application | `<state>/applications/<application id>/application.json` and `before.json`. |

The state folder is `--state-dir`, else `$STRATA_TUNE_DIR`, else `$XDG_STATE_HOME/strata/tuning`
(`%LOCALAPPDATA%\Strata\tuning` on Windows).

A record is comparable only when the engine digest, the model digest, the backend, the workload, the
context and the machine all match (from the code, `strata_control/tune_history.py`). A model is
identified by a content digest, not by its name. The default digest samples the head, the middle and
the tail plus the exact size; `--full-digests` hashes the whole file.

`strata tune auto` keeps its evidence in the preset folder: one JSON file per preset, `active.json`
for the active choice, and `history.jsonl` with one row per trial (from the code,
`strata_control/presets.py`). `strata presets recheck` adds a recheck block to the preset.

`apply` refuses a recommendation when the sweep did not complete, when it recommended no change,
when the machine drifted during the sweep, when the configuration already carries the flags, or when
the recorded identity cannot be matched to the installed bytes. The CLI also refuses a sweep that
recorded no model digest (from the code, `strata_cli/cli.py:_tune_apply`).

## presets

List and manage the presets that `strata tune auto` saves.

```
strata presets {list,status,show,apply,rename,delete,undo,export,import,effective,off,recheck} ...
```

| Subcommand | Usage |
|---|---|
| `list` | `strata presets list [--model TEXT]`. `--model` filters on the model or quant name. |
| `status` | `strata presets status TARGET`. Which preset a configuration would start with, and whether it is still valid. |
| `show` | `strata presets show PRESET`. One preset in full. |
| `apply` | `strata presets apply PRESET`. Makes the preset active; the previous one is kept for undo. |
| `rename` | `strata presets rename PRESET NAME`. |
| `delete` | `strata presets delete PRESET`. |
| `undo` | `strata presets undo PRESET [--config TARGET]`. Goes back to the preset that was active before, or to the standard settings. |
| `export` | `strata presets export PRESET [--file PATH]`. Writes the preset to a file, or prints it. |
| `import` | `strata presets import FILE`. |
| `effective` | `strata presets effective TARGET`. What a start with these choices runs: the saved tuning on top of the configuration, and the list of changed flags. |
| `off` | `strata presets off TARGET`. Switches the saved tuning off for these choices. The preset is kept. |
| `recheck` | `strata presets recheck PRESET TARGET [--gpu-lock PATH] [--serve-dir DIR] [--work-dir DIR] [--status-file PATH] [--boot-timeout S] [--no-quiet-check]`. A short check that the preset is still as fast. `--boot-timeout` is `900.0` (default). |

`TARGET` is a saved profile name or a run config path. `presets.find` accepts a full preset ID, a
prefix of at least 4 characters, or the exact name (case-insensitive). An ambiguous value is
refused.

What it changes: `apply`, `rename`, `delete`, `undo`, `import`, `off` and `recheck` write the preset
store. `list`, `status`, `show`, `export` and `effective` write nothing.

```
strata presets list --model qwen
strata presets effective four-gpu
```

| Exit code | Meaning |
|---|---|
| 0 | Done. |
| 2 | A target, a file or an argument is invalid. |
| 5 | The preset cannot be used, or the PC changed too much for a re-check. |

## connect

Print the settings file and the environment that an app needs to reach the running server. With no
app name it lists the apps.

```
strata connect [APP] [options]
```

| Option | Default and effect |
|---|---|
| `APP` | none. `opencode`, `codex`, `claude-code`, `omp`, `pi`, `hermes`, `openclaw` or `deepseek-harness`. |
| `--url URL` | the running launch, else `http://127.0.0.1:8080`. |
| `--key KEY` | the running launch's key. |
| `--key-env NAME` | none. Reads the key from this environment variable. |
| `--model NAME` | the running launch's model. |
| `--context SIZE` | the running launch's context, else 32k. |
| `--preset NAME` | none. Takes the address, the key and the model from this saved preset. |
| `--check` | off (default). Sends one prompt and one tool call the way this app sends them, and reports each. |
| `--write` | off (default). Writes the file or files, after a question. A replaced file is backed up first. |
| `--yes` | off (default). Skips the question for `--write`. |
| `--show-key` | off (default). Prints the real key instead of a masked value. |

What it changes: nothing without `--write`. With `--write` it writes the app's settings file and
keeps a backup of a replaced file.

```
strata connect opencode --check
strata connect codex --write --yes
```

| Exit code | Meaning |
|---|---|
| 0 | Done. |
| 1 | `--check` reached the server, but a check failed, or the server did not answer. |
| 2 | The app name is unknown, or `--write` ran without a terminal and without `--yes`. |

## code

Coding client for the shared Desktop harness. The parser exists in `strata_cli/cli.py`, but the
dispatch answers with exit code 5 in this tree: `the shared Desktop harness component is not
installed; coding sessions are unavailable`.

```
strata code [--workspace DIR] [--resume ID] [run|sessions] ...
```

The declared shape is `code run --prompt-file PATH --policy PATH` and `code sessions list`. Do not
rely on either in this build.

## repair-access

Take back Strata files that a run as administrator locked on Windows. The command checks the known
Strata folders, repairs what it can, and reports the count of checked, repaired and failed entries.

```
strata repair-access [--output human|json]
```

The same repair runs automatically, once, when a command meets a permission error in a Strata folder
(from the code, `strata_cli/cli.py:_access_retry`).

What it changes: the owner and the permissions of files in Strata's own per-user folders. It changes
no file content.

```
strata repair-access
```

| Exit code | Meaning |
|---|---|
| 0 | Every entry was checked and repaired where needed. |
| 5 | At least one entry could not be repaired. |

## update

Download and install the engine for this PC, and Control, from the signed feed. It switches only
after a test start, and it keeps the previous version.

```
strata update [options]
```

| Option | Default and effect |
|---|---|
| `--check` | off (default). Reads the feed and says what would change, and installs nothing. |
| `--app` | off (default). Also downloads and verifies the Desktop app's installer. |
| `--app-version V` | `$STRATA_APP_VERSION` (default when the variable is set). |
| `--no-engine` | off (default). Skips the engine. |
| `--no-control` | off (default). Skips Control. |
| `--feed URL\|DIR` | `$STRATA_UPDATE_FEED`, then the feed used last, then the project feed at `https://github.com/Niko1221/Strata/releases/latest/download/strata-feed.json`. |
| `--components-dir DIR` | the per-user component folder, or `$STRATA_COMPONENTS_DIR`. |
| `--backend FAMILY` | detect this PC's GPU. |
| `--wait S` | none (default). Waits up to this long for a serving engine to stop instead of refusing. |
| `--force` | off (default). Tries a version again that failed its test start or was rolled back from. |
| `--model-config RUN_CONFIG.json` | an installed model. The run config for the test start. |
| `--output human\|json\|jsonl` | `human` (default). |

A feed is accepted only with a signature from a key in `update_keys.json` or
`STRATA_UPDATE_TRUSTED_KEYS` (from the code, `strata_cli/update_cmd.py`). A download is the one
explicit step that acquires anything. An update marks the saved tuning presets for the short
re-check; it deletes nothing.

What it changes: the component folder, and the active engine version. It keeps the version it
replaces. `--check` changes nothing.

```
strata update --check
strata update --wait 600
```

| Exit code | Meaning |
|---|---|
| 0 | Done, or nothing to do. |
| 5 | The feed is not trusted or not usable, or this build refuses to acquire. |
| 6 | An update step failed. The old engine stays in place unless the message says otherwise. |
| 7 | An engine is serving, or another update holds the lock. |
| 130 | Stopped. A partial download is kept, and the next update resumes it. |

## Exit codes

Only codes that the code returns appear here.

| Code | Meaning | Where it comes from |
|---|---|---|
| 0 | Success. | `_success`, and every command that returns 0 |
| 1 | A `connect --check` check failed, or the server did not answer the check. | `strata_cli/launch/cmd.py:run_connect` |
| 2 | A usage, argument, name or format error, or a plan precondition that fails before work. | `ProfileError` default, `ConnectionError` default, `_serve_error` default, `_error(..., 2)` |
| 3 | The server cannot be reached, or its telemetry is malformed or too large. | `strata_cli/client.py`, `ClientError` default |
| 4 | The server refused the credential (HTTP 401 or 403), or `runtime verify` met an unauthenticated owner endpoint. | `strata_cli/client.py`, `strata_control/runtime.py` |
| 5 | The command refuses or cannot proceed: a missing component, an untrusted feed, no installer, no engine, an unusable file set. | `engine_component.EngineMissing`, `acquisition.AcquisitionRefused`, `feed.FeedError` default, `_error(..., 5)` |
| 6 | The operation failed: a local file operation, a server start, a refused apply, a failed run, or a reviewed item that changed. | `_run` OSError handler, `runtime`, `tune_apply.ApplyError` default, `ProfileError(..., 6)` |
| 7 | An update is refused because an engine is serving or another update holds the lock. | `strata_control/updater.py`, class `Busy` |
| 8 | A deadline expired: the startup wait, the stop drain deadline, or an operation whose completion is unknown. | `strata_control/runtime.py`, `strata_control/journal.py` |
| 130 | Interrupted (Ctrl+C). | `strata_cli/cli.py:_run`, `_serve_jsonl` |
| 404 | `strata tune rollback` found no application record for that ID. | `strata_control/tune_apply.py:rollback` |
| 409 | `strata tune rollback` refused because the run config changed after the apply. | `strata_control/tune_apply.py:rollback` |

The management bridge returns the same codes in its `exit_code` field and exits with that value.
Read [MANAGEMENT_PROTOCOL.md](MANAGEMENT_PROTOCOL.md).
