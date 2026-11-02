# Several requests at once (batch slots)

Without a parallel option, Strata serves **one request at a time**: the others wait in the server's queue. With `"parallel": N`
(the engine's `--batch N`, also spelled `--slots N`) the engine keeps up to N conversations open and decodes them
**together**: every verify window then carries one token of each conversation, so the dense weights, the shared
expert, the head and every routed expert two conversations share are read once per window for all of them.
Combined with a layer split across several GPUs and `--batch-groups`, the cards also work on different
conversations at the same time instead of waiting for each other.

Upstream keeps this opt-in and changes nothing when the options are absent (#465; the engine part is PR #559).
In this V100 fork, setup writes `"parallel": 2` for a new Volta configuration. It preserves an existing parallel
setting. Use `--parallel 1` to disable it. The engine can fall back to one request if two slots do not fit in
GPU memory; check `concurrency.serving` in `/v1/status`. Extra slots can reduce single-request speed.

**Qwen3.6-35B-A3B** has the same option on its own engine path (`"parallel": N`, `--batch N`, `--batch-mtp`, up to 8 slots, the same
engine protocol and the same server): its numbers, what a slot costs and where setup recommends it are in [QWEN36.md](QWEN36.md).

## Turning it on

One GPU: add `"parallel": 2` to the model's config (`strata-<model>.json`) and restart, or run setup with
`--parallel 2`. Setup recommends it only where it does not cost speed (below); any number you ask for is kept as
asked, with a note when it is more than setup would recommend.

```
"parallel": 2
```

With MTP (`--mtp` and `--spec`), `--batch-mtp` (in the config's `args`, or `STRATA_BATCH_MTP=1` in the server's
environment) lets each batch slot verify MTP proposals beside its token: one proposal per window by default, more
with `STRATA_BATCH_VERIFY_WIDTH=1` (below). It is opt-in; without it the batch behaviour
described below is exactly the one without MTP. It needs VRAM per slot for the draft state and buffers, so check the
engine's free-memory log before using it on a smaller card. If it cannot run (one slot, no `--mtp`, a layer split with
`--batch-groups` above 1 or with helper-GPU expert caches or `--remote-expert-opt`, a layer split with two stages on one
GPU) the engine says so and batches as usual. RTX PRO 5000 owners measured +31% to +39% total throughput with 2 to 4
clients on one GPU (a RX R9700 run too).

**With a layer split** (e.g. `"layer_split": "24"` on two GPUs, `"parallel": 2`, `--spec 4 --mtp ...`) it works the
same way, with the slot drafters on the **last stage's GPU** (where the solo drafter and the head are): each slot gets a
drafter that shares the solo drafter's weights and owns only its K/V ring and buffers (the engine logs `--batch-mtp: N
slot drafters on CUDAk (X MiB ... each)` at start; the K/V ring of each also keeps a pinned-RAM copy, as the solo
drafter's does). By default a window carries two rows per active slot (its token and its proposal) through every
stage, so at most four slots are in one window; `STRATA_BATCH_VERIFY_WIDTH=1` widens a group (below) and can fit
fewer slots at a time. More slots (`--batch` up to what fits) rotate through. The windows run through the stages
one after the other (`--batch-groups 1`): on 0.1.41 a layer split with `--batch` 2 or more pipelines the slots in groups
by default, and the pipelined path does not run the slots' MTP drafts, so with `--batch-mtp` and no `--batch-groups` the
engine (0.1.43.1) pipelines and leaves `--batch-mtp` off, and says so (0.1.42/0.1.43 ran one serial group with it: 4 streams on 2x A4000 aggregated 63.9 tok/s against 91.8 pipelined, 85.8 pipelined without `--batch-mtp`; #2038). `--batch-groups 1` or `STRATA_BATCH_MTP_SERIAL=1` keeps `--batch-mtp` with serial windows, which is the better choice for about two streams as measured below; `--batch-groups G` (G above 1) or `--batch-groups auto` given on the command line
keeps the pipeline and turns `--batch-mtp` off. Only a layer split of two stages on two GPUs has been run.

Measured on a layer split of 2x RTX 3080 20 GB (220 W cap) with a Xeon E5-2696 v4, UD-Q4_K_XL, `"layer_split": "23"`,
`"parallel": 2`, `--spec 4`: two concurrent greedy decodes of 300 tokens, the two streams' tok/s added, median of
the measurements (10 rounds of two streams after each restart of the engine, 20 measurements per restart). The
engine was 0.1.41 with the rest of our production stack on it (the shared KV pool, and the adaptive expert tier
running in batch windows in the first two arms; the pipelined path of the third never adapts in windows), so this is
`--batch-mtp` on top of that, not this change alone on main.

| arm | aggregate tok/s | restarts, measurements | per window |
| --- | --- | --- | --- |
| `--batch-mtp`, one group | **89.2** | 3, 60 | 3.3 rows, 34.6 ms, 66-67% of the proposals accepted |
| no `--batch-mtp`, one serial group | 77.3 | 2, 40 | 2.0 rows, 23.7 ms |
| no `--batch-mtp`, `--batch-groups` unset (the 0.1.41 default: two pipelined groups of one slot) | 77.7 | 1, 20 | one row per group step |

That is +15.4% for `--batch-mtp` against serial windows without it (95% bootstrap interval of the ratio of medians
+12.5% to +17.9%) and +14.8% against the pipelined default; the two arms without it tie. One stream alone decodes the
same with and without it (81.7 and 82.0 tok/s), and the two slot drafters cost 35 to 110 expert-cache slots on the
last GPU (50 MiB of private state each).

The pipelined default is better in one place: a new prompt of about 48k tokens arriving while one stream decodes was
read at 2508 tok/s with pipelined groups and 1217 tok/s in serial windows, and the decoding stream finished sooner
(25.0 against 18.0 tok/s over its whole run); `--batch-mtp` does not change that. So it pays where concurrent decodes
dominate; with long reads arriving beside decodes the pipelined groups can be the better choice. Not measured: more
than two stages (the report on #1253 found no gain at four stages), more than two slots, sampled decoding, an
upstream-main binary (the pipelined arm is our build, started with the flags 0.1.41 resolves to).

Our measurement of `--batch-mtp` on a layer split with 2x R9700 and all experts in VRAM: 80.2 -> 62.7 tok/s (-21.9%, 0/5 pairs faster) against the pipelined default, so it pays only when the experts do not all fit in VRAM.

Earlier, on engine 0.1.40.3 with this change and the same two cards, two concurrent streams: UD-Q4_K_XL 39.9 to 43.5
tok/s without and 44.5 to 51.3 with `--batch-mtp` (2 runs per arm), and a small Coder IQ1_M test model with all
experts in VRAM 84.0 without and 101.2 with it, the greedy text of both streams identical.

Text: with the same expert cache on both arms (the slot drafters take VRAM, so the cache differs unless the
reserve is adjusted) and `--pcie-frac 0`, `--adapt-every 0`, the greedy text with `--batch-mtp` was equal to the text
without it for 5 short prompts in all 6 comparisons (one restart per arm). That is text equality, not a bit-exactness
proof, and a 25k-token prompt read while the other stream decodes was not repeatable between two identical streams
of the same arm.

With a layer split, the engine options go into the config's `args`:

```
"args": [ ..., "--batch", "8", "--batch-groups", "4" ],
"layer_split": "12,24,36"
```

| Option | What it does |
| --- | --- |
| `"parallel": N` / `--batch N` / `--slots N` (2..8 normally; with a layer split up to 8 per group, see below) | up to N conversations have batch slots; more requests wait for a free slot. Each slot gets its own state (a session carved like the stage's own: GDN recurrence, QSA K/V and indexer, PLE history) on every GPU of the split. With grouped MTP, more than 8 slots can rotate through eight-row windows if memory permits. |
| `--batch-groups G` | with a layer split (default: auto, below): the N slots in G groups that flow through the GPUs as a pipeline (GPU k runs one group while GPU k+1 runs another). G must divide N. 1 = all slots in one window, GPU after GPU. |
| `--batch-groups auto` | the default on a layer split since 0.1.41 (give no `--batch-groups`): the engine pipelines one group per GPU stage (the most that divide the slots; 8 slots on 4 GPUs = 4 groups of 2) and says so (`INFO batch_groups=G`). `--batch-groups 1` turns it off (all slots in one window, GPU after GPU). Measured, 8 clients, total tok/s against one group: 4 x R9700 166 against 86, 2 GPUs 109 against 78, 3 GPUs 110 against 78. With `--adapt-async 1` and no `--batch-groups` the default is one group (the adaptive tier between batch windows does not run in pipelined groups) and the log says so; with `--batch-mtp` and no `--batch-groups` the pipeline wins and `--batch-mtp` is left off (`--batch-groups 1` or `STRATA_BATCH_MTP_SERIAL=1` keeps it); `--batch-groups G` or `--batch-groups auto` given with `--batch-mtp` pipelines the groups and turns `--batch-mtp` off, and given with `--adapt-async 1` pipelines the groups and turns `--adapt-async` off (the pipelined path does not adapt). |
| Explicit `--layer-split` | Each GPU loads only the dense weights for its own layers. This is automatic. The memory that this releases is available to the expert cache. |

The engine never refuses a count it cannot run: it says so in its log and runs what it can - at most 8 slots without a
layer split (a window holds 8 rows; with a layer split and pipelined groups, 8 per group), as many as fit in VRAM, or none (one request at a time) when not two fit. The server reads
the count the engine reports (`INFO batch_slots=N`), and `GET /v1/status` says it (`concurrency.serving`).

The slot sessions are carved **after** the main session, the native head, the logits buffer
and the MTP draft layer, so the fit check sees the memory really left on each card (the
draft sits on the last stage of a split): a count that does not fit falls back to fewer
slots, or to no batch slots (one request at a time) when not even one extra session fits
every stage. `fit` does not mean "a carve succeeds": each slot must also leave its stage's
booked working set - the `--vram-reserve-mib` reserve, the prompt path's own buffers when
it does not borrow (and, with a profile, an expert cache big enough to lend them, so a
borrowing prompt never has to fall back to its own), the verify windows of a later stage,
the draft head of the last, and a working cache floor - or the first request finds the GPU
full and dies at `verify: instantiate: out of memory`. The fallback is decided at startup,
not with an out-of-memory error mid-request. Verified on this fork's two V100s at the
524,288-token context with the 20/28 layer split and `"parallel": 2`: the engine reported
the fit check, the server stayed ready with the configuration untouched, and every request
took the solo path (`GET /v1/status` said `concurrency.serving` 1). The exact free-VRAM
figures at admission were re-verified after a follow-up fix to the batch-slot allocation
ownership; the final numbers are in the benchmark report.

The fork makes upstream stage-weight trimming automatic for explicit CUDA/HIP splits. Remove
`--trim-stage-weights` and `STRATA_STAGE_TRIM` from those configurations; they are no longer accepted
or needed. The separate experimental Intel (SYCL) engine retains its own opt-in path.

## More than 8 slots on a layer split, and the opt-in admission and decode switches

(From havok2-htwo's patch series, PR #2364.)

**More than 8 slots.** With a layer split a pipelined window holds at most 8 rows, but each group of the pipeline is its
own window, so `--batch 16` on 4 GPUs runs 4 groups of 4 slots (`--batch-groups auto`), `--batch 32` runs 4 groups of 8.
Until 0.1.44.1 the engine cut such a request down to 8 slots and said so; it now runs what is asked for, as many slots
as fit in VRAM and at most 8 per group (`INFO batch_slots=N`). `--batch-mtp` keeps its limit (rows per window are
doubled by the proposals). Nothing changes for 8 slots or fewer. Every slot of a 16-slot pipelined batch decodes
exactly the tokens it decodes alone (`tools/batch_test.py --n 16`, which now takes up to 32 prompts).

**Opt-in switches** (environment of the engine, i.e. `"env"` in the config; none is on by default). Most of them
change how the work is scheduled or copied, not what is computed; `tools/batch_test.py` checks the default batch
path's tokens against solo decoding. `STRATA_BATCH_VERIFY_WIDTH=1` is the exception: it changes the row layout and
is not promised output-identical (see below), so that check is not a proof for it.

| Switch | What it does |
|---|---|
| `STRATA_SLOT_COPY_DIRECT=1` | an admission (and a slot's resume) copies the running state and the QSA K/V device to device, instead of through host images |
| `STRATA_SLOT_COPY_BATCHSYNC=1` | with the above: queue the ~116 copies and synchronize each device once |
| `STRATA_SLOT_CKPT_MOVE=1` | the admission's turn checkpoint moves into the slot (no 112 MB copy); it also keeps the dropped checkpoints' storage, as `STRATA_CKPT_POOL` does; a session SAVE or parked conversation taken right after an admission lacks the moved checkpoint (the slot holds it) |
| `STRATA_CKPT_POOL=1`, `STRATA_CKPT_THREADS=1` | a checkpoint is saved into the vectors of a dropped one; a split's stage parts are saved at once |
| `STRATA_ADMIT_MERGE_FIRST=1` | an admission reads its last prompt tokens and its first token in one window (all experts resident only) |
| `STRATA_PIPE_BYIELD_NODRAIN=1` | a BYIELD that reaches a pipelined batch is dropped without draining the groups |
| `STRATA_BATCH_SELF_COMMIT=1` | a one-row-per-slot window advances the GDN state itself, no commit graph |
| `STRATA_QSA_SLOTS=1` | a window's QSA block scores, top-k and attention run as one launch over its rows |
| `STRATA_QSA_SEGMENTS=1` | a window's contiguous rows of one slot (its `--batch-mtp` proposals) run the indexer append, block scores, top-k and attention as one call over the run; wins over `STRATA_BATCH_ATTN_ROWS` and `STRATA_QSA_SLOTS`; a streamed KV cache keeps the per-row scores/top-k/attention |
| `STRATA_BATCH_VERIFY_WIDTH=1` | with `--batch-mtp` on serial windows: a slot's group is as wide as the row budget allows (see below) |
| `STRATA_HANDOFF_PIPE=P` | a layer split hands a prompt chunk to the next card in P pieces (2..16), gated by events, on a persistent thread |
| `STRATA_ADMIT_SHORT_FIRST=N` (server) | a new prompt of at most N tokens takes the control lines before longer ones (a longer one that waited `STRATA_ADMIT_MAX_WAIT_S`, default 30, counts as short) |
| `STRATA_TRACE=1` | one line per admission with its phases (drain, prompt, first window, slot copy, checkpoint copy) |
| `STRATA_MMQ_DIRECT=1`, `STRATA_MMQ_GROUP_SORT=1` | CUDA, needs the build option `-DSTRATA_MMQ_TAB=ON`: the prompt MMQ reads resident experts in their cache slots; full MMQ groups sorted by row count (docs/DETAILS.md) |

Each slot's session takes VRAM from the expert cache, so on a card or split where the experts do not all fit, 16 slots
cost the single-user speed (see the sections above on what a slot costs).

### What a slot costs, and what setup recommends

Every slot's session takes VRAM that the expert cache would otherwise hold: 0.56 GiB at a 32K context with 8-bit
KV, more with a longer context unless the KV cache streams (`--kv-resident`: then only the attention's 32K window
stays in VRAM, and each slot's whole KV cache takes pinned RAM - 1.6 GB at 128K). On the fork's two V100s,
slot sessions measured 0.24 GiB (first card) and 0.33 GiB (last card) on the 20/28 split at
the 32,768-token context. At the 524,288-token context on a 24/24 split, each slot
requires 3.66 GiB per card. Before the admission fix, two such slots left insufficient
working memory and the first request failed. The corrected engine serves one request
at a time at both the tested 20/28 and 24/24 splits with vision and MTP enabled.
On a card whose
experts mostly run on the CPU, a batch also reads about as many distinct experts as the requests one by one
(different conversations route to different experts), so the gain is in **latency** (nobody waits for a whole
answer), and a request alone
runs slower (the smaller expert cache): 11-24 % on a 12 GB card, see the measurements below.

The elastic K/V (`--kv-grow`) does not run with `--batch`: the engine's condition for it requires no batch slots. A
batch at a full context therefore pays for its sessions out of the expert cache and falls back to fewer slots, or to
one request at a time, when they do not fit (the 3.66 GiB per card per slot at 524,288 tokens above). KV streaming
(`--kv-resident`, e.g. `--kv-resident 32768`) keeps the full `--max-context`: each layer keeps a 32K-cell window in
VRAM, the rest of the slot's K/V sits in pinned host RAM, and the window's cells move over PCIe as it advances. That
makes the sessions small enough to fit where they otherwise would not. The trade is VRAM for host RAM and PCIe
traffic.

So setup recommends `"parallel"` **only where the experts mostly fit in VRAM**: the expert cache (each card's VRAM
less ~5 GB, every card of a layer split counted) must still hold at least half of the model's experts beside the
slots, and the slots may take at most a fifth of it, up to 4 slots. With Q2_0 at 32K that is 3 slots on a 24 GB
card, 4 from 32 GB or on a split such as 2 x 16 GB; IQ3_S needs 32 GB or a split. Everywhere else (any 12 or 16 GB
card alone) it stays at one at a time and setup says: "parallel N reduces waiting for several users but costs
about 10-25% speed per request on this card". `--parallel N` is honoured as asked either way.

## How the server uses the slots

- **One request alone** runs on the usual solo path (verify windows with MTP drafts): the fastest single stream.
- **When a second request arrives**, the first is stopped (`STOP`) and continues in a batch slot with its prompt
  plus what it generated so far - the engine's prompt cache holds exactly that, so nothing is read again - and the
  new request is admitted next to it. By default, a request in a slot decodes **without MTP drafts** (one token per window).
  With `--batch-mtp`, each slot verifies one MTP proposal alongside its current token.
- **A request left alone in a slot** (the others finished, nobody waits) goes back to the solo path: the slot is
  stopped, the engine copies its sessions back and decodes with MTP drafts again (at most twice per request; with
  `--prompt-cache 0` it stays in the slot; `STRATA_PARALLEL_SOLO=0` turns it off). The draft layer's own K/V was
  built for another conversation then, but measured it accepted as many drafts (140 of 172) as a draft layer that
  read the conversation (140 of 173).
- **More requests than slots** wait for a free one (`/metrics` -> `live.slots` shows each slot: idle, reading or
  decoding, its tokens and tok/s; `live.running` the requests in flight).
- **Each admission** reads the request's prompt through the usual prompt path (prompt cache and conversation
  checkpoints included) and produces its first token there; the state is then copied into the slot. Admissions
  are taken one at a time, and **the slots decode between the prompt's chunks**: after each chunk (`--prefill`,
  2048-8192 tokens) they decode for half as long as the chunk took (`STRATA_BATCH_DECODE_SHARE`, default 0.5), so
  a long prompt slows the others down instead of stopping them. The chunks are the ones one uninterrupted read
  takes, so the prompt's arithmetic is unchanged.
- **A long prompt gives way to a short one** (#656's cooperative preemption): when a request with a prompt under
  half as long is waiting, the server sends `BYIELD`; at its next chunk boundary the long read stops, the part read
  so far is copied into a slot, the short request is admitted, and the long one then goes on from its slot with the
  same chunks (at most twice per request).
- **Each slot is a conversation cache.** A finished slot keeps what it holds (the prompt, the answer, and the
  checkpoint at the prompt's last turn boundary); the next turn of that conversation goes to that slot and the
  engine copies its state back (50-60 ms for a short conversation) instead of reading the history again - also for
  a client that drops the reply's thinking from the history (the checkpoint matches up to the new turn). A new
  conversation takes an empty slot, else the one used longest ago.
- Across the RAM cache, the disk tier and the slots, the longest valid prefix of the request's prompt wins: a
  shorter or equal disk record never displaces the longer state an idle slot returns (a return is already
  resident; an equal-length record would only cost a read and a restore). Verified on this fork's two V100s
  with three slots: a 25-token disk record did not displace a 152-token state, nor a 172-token record a
  299-token one.
- Slots are assigned so that consecutive requests land in different pipeline groups (`--batch-groups`).
- A client that disconnects stops its slot (`BSTOP`); the others go on.

## Reading a prompt beside pipelined groups (experimental, opt-in)

With a layer split and `--batch-groups` above 1 a prompt is read in one run: every decoding slot stops until it is
read, however long it is (#1906). One GPU does not have this problem: there the slots decode between the prompt's
chunks for `STRATA_BATCH_DECODE_SHARE` x the chunk's time.

`STRATA_PIPED_CONCURRENT=1` reads a prompt on a thread of its own while the slot groups keep decoding through the
pipeline, instead of stopping them: the prompt's chunks and the slots' windows run on the same GPUs at once, on their
own streams (the windows' at the device's highest priority on CUDA; `STRATA_PIPED_CONCURRENT_PRIO=0` keeps the default),
so the slots never wait for the read and the read is one run with no drain inside it.

- The prompt's K/V goes to the main sessions and the slots' to their own, so they do not touch each other's state. The
  n-gram (PLE) table both of them read is locked. The adaptive VRAM tier stands still during the read: its round lands
  before the read and none starts while it runs.
- The VRAM expert slots the prompt path borrows are lent before the read and refilled after it, with nothing in flight;
  in between the slots' windows compute those experts on the CPU.
- The slots' doorbells are served on a thread of their own during the read, which makes no CUDA call. A CUDA call can
  wait for the driver's lock while its holder waits for the GPU - a free, or the module the first kernel of a new shape
  loads - and a window on that GPU waits for its doorbell (issue #31's pattern: two such hangs were found and read from
  the engine's stall dumps while this was built). The MTP drafter's prompt buffers and the prompt path's pinned upload
  slots are sized before the read, so its thread allocates and frees nothing. Between windows the doorbell thread waits
  for the next launch, which wakes it (a timed sleep is a whole timer tick on Windows, up to 15.6 ms). A window that
  will not finish is still found (its graph ended with a layer still to ring, or no layer rang for 20 s), and the stall
  watchdog stays on for the whole read.
- A waiting shorter request takes over at the read's next chunk (`BYIELD`), also when no slot decodes: the run ends
  there, the pipeline is drained and the part read waits in a slot, as off the pipeline. A read nobody waits for is not cut.
- Off, said once, where the read's thread would allocate or free (`STRATA_PREFILL_HELP=1`, `STRATA_SPLIT_MTP_BATCH=1`),
  where serving a doorbell makes a CUDA call (`--pcie-mode dma`, `STRATA_FS_SLOTS`, the helper caches
  `--expert-cache-device1..3`), and with an expert source that hands out buffers it recycles (the GGUF in place, an
  unbuffered source, `STRATA_IO_PREFETCH`): the experts come from the RAM arena or the mapped file. A prompt with
  pictures is read the usual way.

One log line per read (`strata batch: read N tokens on a thread of its own in X ms (concurrent, to Q of T); the slots
decoded R rows meanwhile (S rows/s; stage k: W windows, H ms host each)`). Built for CUDA and HIP (HIP keeps the default
stream priority, and is not run on a GPU here); SYCL does not have it.

## A prompt read beside pipelined groups (opt-in)

With a layer split and `--batch-groups` above 1 a prompt is read in one run: every decoding slot stops until it is read,
however long it is, `STRATA_BATCH_DECODE_SHARE` does not apply and no `BYIELD` is taken (`batch_read_piece` returns 0
for the pipelined groups; #1906). One GPU does not have this problem: there the slots decode between the prompt's chunks
for `STRATA_BATCH_DECODE_SHARE` x the chunk's time. These variables give the pipelined groups the same choice:

- `STRATA_PIPED_READ_PIECE=N`: a prompt read on the pipelined groups goes in pieces of N tokens, rounded down to whole
  chunks (at least one, at most 32), so a piece reads exactly the chunks a single run would, the two stages stay
  overlapped inside it and only the piece's end drains the pipeline. Between two pieces the slot groups decode through
  the pipeline for `STRATA_BATCH_DECODE_SHARE` x the piece's time, at most `STRATA_PIPED_DECODE_MAX_MS` and at least
  `STRATA_PIPED_DECODE_MIN_MS`, and every group in flight finishes before the next piece, as before any prompt read.
- At a piece's end the pipeline is drained, so a waiting request with a prompt under half as long can take over there
  (`BYIELD`, as between the chunks of a read off the pipeline): the part read waits in a slot and the read goes on from
  it after the short request is in. The pieces are also taken when no slot decodes, for this reason (off the pipeline
  the same read goes in `STRATA_BATCH_YIELD_CHUNKS` pieces then).
- A read that starts within a second (plus the decode budget) of the previous such read's end - prompts arriving back to
  back - first gives the slots one decode budget (the same rule, from the previous read's last full piece); otherwise
  the slots would stop for both reads.

With the variables unset the read is one run as before. One log line per piece (`strata batch: piped read N tokens in X
ms (to Q of T)`, with `, then the slots decoded R rows in Y ms` when slots decode), per give-way (`the prompt read gives
way at Q of T tokens; slot S holds it`) and per lead-in (`a read right after a read: the slots decoded R rows in Y ms
first`).

**What it trades.** Every piece ends by draining the pipeline, so a long read takes longer - also on an idle server,
since the pieces are what lets a short request in - and the decoding streams get tokens where they had none. It is a
latency choice, not a throughput gain: leave it off where throughput or the long prompt's own TTFT matters most, use it
where a stream that stops for several seconds, or a short question stuck behind a long read, is worse than a long prompt
that waits a little longer.

## The VRAM expert tier keeps adapting in batch windows

Batch windows count which experts they route to, and every `--adapt-every` windows (default 4) the engine swaps the
most-routed experts that are not in VRAM into the cache in place of the least-routed resident ones, the same rule the
solo path uses (`--adapt-swaps` and `--adapt-decay` apply too; on a layer split every GPU's cache adapts). It runs when the
tier is not the whole model and both `--adapt-every` and `--adapt-swaps` are above 0. By default the round runs beside the
window's commit and drafts and has landed before the next window starts. It stands still while a long prompt is read
between decoding windows (the prompt may hold a loan of cache slots). The `strata batch:` log line shows the rounds and swaps, each stage's
GPU-reach wait and pool time, and how many routed entries the VRAM tier, the PCIe share and the CPU served per window.
Swaps move experts between the GPU and the CPU, which round differently, so greedy outputs can differ from run to run;
`--adapt-every 1000000` keeps the tier fixed (as in the exactness settings below). The pipelined `--batch-groups` path
does not adapt.

With `--adapt-async 1` (needs the resident RAM mode) the round does not hold a window up: each batch window moves the
round on one step (table changes and uploads on the main thread, the copies on a helper thread and the cards' refill
streams), as on the solo path. A request that arrives finishes the round in flight before it starts, and a prompt read
between decoding windows does not move it. `--adapt-min-gain F` (default 1.5) raises the bar a swap must clear (the
routing count of the expert coming in against the one going out), so fewer experts move per round; both tiers use it.
The log line's `adapt wait` is the time the main thread spent on the tier per window (the blocking tier: waiting for
its round).

Measured with the same setup as the `--batch-mtp` numbers above (2x RTX 3080 20 GB, UD-Q4_K_XL, `"layer_split": "23"`,
`"parallel": 2`, `--batch-mtp`, engine 0.1.41 with our production stack; two concurrent greedy decodes, the streams'
tok/s added, median of 20 measurements per restart):

| tier | aggregate tok/s | restarts | per window | routed entries served from VRAM |
| --- | --- | --- | --- | --- |
| `--adapt-async 1 --adapt-every 2` | **89.2** | 3 | 34.6 ms, adapt wait 0.2 ms | 92.1% |
| blocking, `--adapt-every 2` | 74.7 | 1 | 41.4 ms, adapt wait 8.4 ms | 92.3% |
| none, the cache frozen from a profile the tier had learned on the same prompts | 69.5 | 1 | 43.6 ms | 77.2% |
| none, `--adapt-every 0` | 45.2 | 2 | 66-68 ms | 52.7% |

The switch changes the tier in the solo path as well (there is no switch for batch windows alone), so the table
compares the tier as a whole, not "batch windows only": the solo decode speed moves the same way (81.7, 66.3, 69.0 and
43.1 tok/s). The prompts are the benchmark's own, eight topics that repeat, which the tier learns quickly, so on a
more varied workload the gain may be smaller (not measured). Upstream main runs no tier in batch windows, and this
was not measured against an upstream binary.

## Exactness

A batch row's arithmetic is the single-token window's, so with greedy decoding **every conversation of a batch
produces exactly the tokens it produces alone** - verified token by token for 8 concurrent conversations of 150
tokens, with and without the pipeline (`tools/batch_test.py`), and on one RTX 5070 for 4 conversations, for a long
prompt read while two others decode, for a prompt that gave way and went on, for a next turn continued from its
slot (from all it holds, and from its turn checkpoint without the reply's thinking), and for a request stopped in
its slot and continued on the solo path (`tools/batch_interleave_test.py`). These
settings make the comparison exact:

- `STRATA_IQ_MT_MIN=1` (the multi-token CPU expert kernels for every group, as for the solo path's own
  exactness tests: by default an expert's rows round differently alone than in a group, so the output depends on
  how many rows of a window share an expert - which differs between a batch and a request alone),
- `--pcie-frac 0`: the PCIe share of the missed experts is chosen per window from the window's misses, so the
  same expert can run on the GPU in one window and on the CPU in another, which rounds differently, and
- `--adapt-every 1000000` (the VRAM tier fixed), and `--no-prefill-borrow` while a prompt is read beside decoding
  slots: their windows then see the expert cache without the slots the prompt borrowed (those experts run on the
  CPU).

Measured on this fork's two V100s (Q2_0, int8 KV, 32,768-token context, two slots,
`tools/batch_test.py` with the settings above): each slot's 128 generated tokens were
identical to its solo run, token for token. The two admissions took 1.3 s, and the two
128-token answers (256 tokens together) decoded in 5.8 s at a 44.0 tok/s aggregate
(49.6 tok/s counted from the first token).
With the default settings the outputs stay coherent but drift apart after some tokens, as two solo runs whose
windows differ can. Measured on the RTX 5070 (Q2_0, 4 conversations of 200 tokens, `tools/batch_test.py` without
the settings above): one equal to its solo run, the others apart from token 0, 99 and 174 - the first token already
differed for one, because the adaptive VRAM tier had moved experts between the solo runs and the batch; all four
texts read as well as their solo runs.

Sampled requests (temperature, top_p, top_k, min_p, seed) are drawn row by row with the solo window's
counter-based draw (Philox(seed, position)).

## Limits (for now)

- By default, batch windows carry no MTP drafts: a conversation in a slot decodes one token per window (the solo
  path keeps its drafts, which is why a request alone is not put in a slot, and goes back to it when left alone).
- `--batch-mtp` uses one proposal per slot, or up to `min(--spec, --mtp-max-t, 4)` with
  `STRATA_BATCH_VERIFY_WIDTH=1` (two rows per slot from three active slots up). The width switch is not promised
  output-identical to the default batch path and has no quality benchmark. With a layer split it needs each
  stage on its own GPU, does not combine with `--batch-groups` above 1 (pipelined slot groups run no drafts, #1413)
  and has not been run with helper expert caches.
- Repetition / frequency / presence penalties are not applied in batch windows.
- A prompt shorter than one chunk is read in one piece (the slots wait for it); a read gives way only at a chunk
  boundary, and not for pictures.
- Admissions are one at a time: two new long prompts are read one after the other.
- `--batch-groups` needs every stage on its own GPU. A pipelined slot is a conversation cache again (0.1.41): a request left alone in its slot goes back to the solo path with its drafts, as on one GPU.
- The slot sessions take VRAM (above) and, with KV streaming, pinned RAM.

## Measured

One RTX 5070 (12 GB), Ryzen 5 7600, 64 GB DDR5, Q2_0, 32K context, through the HTTP server: C different requests
sent at once (an 800-word essay each, 256 tokens per answer, greedy, thinking off), median of 3 rounds:

| Concurrent | Setting | Total tok/s | vs one at a time | Per request tok/s | First token: median / last of the round |
| ---: | --- | ---: | ---: | ---: | ---: |
| 1 | one at a time (default; what setup recommends on this card) | 74.3 | - | 83.6 | 0.4 s / 0.4 s |
| 1 | `"parallel": 2` | 67.3 | -9 % | 74.8 | 0.4 s / 0.4 s |
| 1 | `"parallel": 4` | 57.9 | -22 % | 63.8 | 0.4 s / 0.4 s |
| 2 | one at a time | 71.6 | - | 77.3 | 2.1 s / 4.1 s |
| 2 | `"parallel": 2` | 61.0 | -15 % | 32.6 | 0.5 s / 0.9 s |
| 2 | `"parallel": 4` | 54.6 | -24 % | 28.7 | 0.6 s / 0.7 s |
| 4 | one at a time | 70.7 | - | 79.6 | 6.0 s / 11.2 s |
| 4 | `"parallel": 2` | 61.2 | -13 % | 32.2 | 4.5 s / 9.3 s |
| 4 | `"parallel": 4` | 63.1 | -11 % | 16.9 | 1.0 s / 1.8 s |

On this card the slots buy **waiting time, not speed**: the fourth of four requests starts after 1.8 s instead of
11.2 s, but together they decode 11-24 % slower than one after the other, and a request alone loses 11 % (2 slots)
or 24 % (4 slots), because the slots' sessions (0.56 GiB each) come out of the expert cache and most experts run
on the CPU: a batch window over 4 conversations reads 24 CPU experts per layer against ~8 for one, so it costs about
what the 4 tokens cost one after the other (`strata batch:` in the engine log: 54 ms per 4-row window, ~20 ms per
1-row window). This is why setup leaves a 12 GB card at one at a time. Cards that hold most experts in VRAM, and a
layer split, are where the slots also add speed (below).

A historical upstream measurement used a 4-GPU layer split (4 x 16 GB, PCIe Gen3), IQ3_S, eight slots,
four pipeline groups, and stage-weight trimming. Each HTTP answer contained 400 tokens at temperature 0.7 (PR #559):

| Concurrent requests | Per request | Total |
| ---: | ---: | ---: |
| 1 | 123 tok/s (solo path) | 120 tok/s |
| 2 | 57 tok/s | 113 tok/s |
| 4 | 51 tok/s | 205 tok/s |
| 8 | 45 tok/s | 360 tok/s |

With the patches below on engine 0.1.38 and parking on, through the service: 8 requests at temperature 0 -> 369
tok/s, at 0.7 -> 358 tok/s.

Stage-weight trimming alone raised the share of experts held in GPU memory on that machine from
76-85 % to 84-100 % per card. This is an upstream measurement, not a V100 measurement.

### One launch for all slots (opt-in)

In a batch window, QSA block scoring, top-k and decode attention ran as one launch per row, and the GDN recurrence
(in the window and in the 1-row commit) as one launch per slot. Two switches run each of them as one launch for all
rows, every row still reading its own slot's indexer keys, K/V pools and GDN state (passed by value as kernel
params, so the graph captured per row layout stays valid):

| Switch | What it does |
| --- | --- |
| `STRATA_BATCH_ATTN_ROWS=1` | block scores, top-k and decode attention of all the window's rows (up to 8) in one launch each. Not with KV streaming (a slot with `kv_mode` 1 keeps the per-row launches). |
| `STRATA_BATCH_GDN_GROUPS=1` | the GDN recurrence in the window, and the commit of 1-row groups, for all slots in one launch. The conv commits stay per slot. |

Both are off by default; off, the engine runs the same launches as before. On, the arithmetic per row is the same:
4 concurrent prompts at temperature 0 gave the same bytes as without them, with `--batch 4` and `--batch 8`.

### Wider `--batch-mtp` groups (opt-in)

`STRATA_BATCH_VERIFY_WIDTH=1` (the engine's environment) lets a slot's group carry more than one proposal. It needs
the serial batch MTP windows: one group (`--batch-groups 1`, or `STRATA_BATCH_MTP_SERIAL=1`, which keeps
`--batch-mtp` on a layer split) or one GPU. Pipelined groups run no `--batch-mtp`, so the switch has no effect there.

- With at most two active slots, each group holds up to `min(--spec, --mtp-max-t, 4)` rows (`--spec` alone when
  `--mtp-max-t` is unset): the slot's token first, then its proposals. Two slots at the four-row cap fill the
  eight-row window.
- With three or more active slots, each group holds at most two rows: the token and one proposal.
- A group is often narrower than that budget. It holds only the proposals the drafter actually produced, and it
  shrinks so its rows stay inside `--max-context`. A window still holds at most eight rows, so a slot that does not
  fit waits for the next window.
- Off (the default), `--batch-mtp` keeps two rows per slot. Without `--batch-mtp` a slot has one row. The solo path
  and pipelined groups are unchanged.

Its rows use the same per-row verify kernels, but the wider group changes what each window verifies. It is not
promised output-identical to the default batch path or to solo decoding: under the current service routing and
per-window plans a different path can be taken. No quality benchmark has been done. The switch stays opt-in.
Wider groups also make the captured verify graphs more numerous. `--batch-mtp` keeps at most 64 layouts per verifier
and evicts the least recently used beyond that; 0.1.39 kept every layout.

### One call per slot run (opt-in)

`STRATA_QSA_SEGMENTS=1` is the per-slot counterpart of the switches above. When a window carries a run of
contiguous rows of one slot (the wider `--batch-mtp` groups), the slot's indexer append, block scores, top-k and
decode attention run once over the run instead of once per row. It is off by default; outside batch windows, and
for one-row runs, nothing changes. It reuses the existing multi-row calls and the window's per-row buffers, so it
allocates nothing new. The score and attention arithmetic is FP32, and the result is bitwise the per-row loop's
(`batch_rows_parity` checks runs of 1, 2, 4 and 8 rows, int8 and fp16 pools, and a captured graph replayed A/B/A).
A streamed KV cache (`--kv-resident`) keeps the per-row resolve-then-attend order, so block scores, top-k and
attention run per row on such a window (the indexer append still runs over the run, since the indexer key pool is
not streamed). When both could run, it wins over `STRATA_BATCH_ATTN_ROWS` and `STRATA_QSA_SLOTS`.
[QSAROWS.md](QSAROWS.md) has the kernel details. Neither switch changes a default. The
[2026-11-02 V100 concurrency report](../benchmarks/v100-revolta-concurrency-2026-11-02.md) records the parity runs and
the concurrent HTTP runs at 32K and 262K.

## Together with conversation parking

`--conversation-cache-mib N --conversation-cache-slots K` (DETAILS.md) works with the layer split too: a request
whose conversation was parked is restored on every stage before its admission, so an agent and its sub-agents, or
several chats that alternate, come back without reading their history again. Measured on the same 4-GPU split,
two long conversations alternating through the HTTP server: the first turns took 3.9 s and 5.7 s to the first token
(their prompts read), the follow-ups 0.53 s and 0.46 s.

The V100 fork also retains the optional disk tier (`--conversation-cache-disk` and
`--conversation-cache-disk-gib`). Solo and batch admissions can restore a matching disk record.
The disk store saves the outgoing main session during admission; replacing a held batch slot does
not itself save that slot to disk. Do not assume that every evicted slot has an L3 record.
The system-prompt root of an admission is captured into the RAM and disk tiers independently of
the slot state: a new chat reuses its shared system prompt even when the slot that held it was
replaced. The root is evicted only after the conversation records (see
`--conversation-cache-keep-root` and `--conversation-cache-disk-keep-root` in DETAILS.md).

## Testing

Four scripts drive a built engine or a running server; each exits non-zero on a failure. `serve/test_parallel.py`
tests the server's side with a scripted engine (no GPU).

| Script | What it checks |
| --- | --- |
| `tools/batch_test.py` | the same prompts alone (`GEN`) and together in the batch slots (`BGEN`): every slot's greedy tokens equal its solo tokens; prints the aggregate rate. `--batch-groups` in `--extra` tests the pipeline, `--keys "temperature=0.7"` the sampled rows. |
| `tools/batch_interleave_test.py` | a long prompt read while two slots decode, a prompt that gives way (`BYIELD`) and goes on, and a next turn continued from its slot: each equal to its solo tokens. On a two-stage layer split with pipelined groups add `--batch 4 --batch-groups 2` to `--extra` (its 3 slots do not divide into two groups) and set `STRATA_PIPED_READ_PIECE`; run it without a conversation cache, or step 4 restores the prompt instead of reading it. |
| `tools/parking_test.py` | a follow-up to a conversation decodes the same tokens whether its state stayed live or came back from the parking cache (with a layer split: every stage's image). |
| `tools/early_close_test.py` | a client that stops reading a streamed answer early (alone, and with a second request running) does not leave its tokens to the next request (server). |

For exact comparisons pass `--pcie-frac 0 --adapt-every 1000000` (and the scripts set `STRATA_IQ_MT_MIN=1`):

```
python3 tools/batch_test.py --exe engine/strata --config strata-<model>.json --batch 8 --n 8 \
    --extra "--layer-split 12,24,36 --batch-groups 4 --pcie-frac 0 --adapt-every 1000000"
python3 tools/batch_interleave_test.py --exe engine/strata --config strata-<model>.json \
    --extra "--pcie-frac 0 --adapt-every 1000000 --no-prefill-borrow"
python3 tools/parking_test.py --exe engine/strata --config strata-<model>.json \
    --extra "--layer-split 12,24,36 --conversation-cache-mib 8192 --conversation-cache-slots 4 --pcie-frac 0"
STRATA_KEY=<key> python3 tools/early_close_test.py http://127.0.0.1:8080
```

For single-GPU `--batch-mtp`, use a model with `rt/draft_vocab.bin` to exercise admission with a
shared draft-vocabulary head. This compares both slots' output against solo decoding:

```
python3 tools/batch_test.py --exe engine/strata --config strata-<model>.json --batch 2 --n 2 --max-new 64 \
    --extra "--batch-mtp --pcie-frac 0 --adapt-every 1000000"
```

## Engine protocol (`--serve`)

On top of `GEN` / `GENI`:

| Line | Direction | Meaning |
| --- | --- | --- |
| `BGEN <slot> <max_new> [keys] <ids>` | in | read the prompt (as `GEN 1`), then continue in `<slot>` |
| `BGENI <slot> <max_new> [keys] <file> <ids>` | in | the same with images |
| `BADM <slot> <1/0>` | out | after the admission's `DONE`: 1 = it continues in the slot, 0 = it ended |
| `BT <slot> <id>` | out | a token of that slot |
| `BDONE <slot> <generated> <stop/length/cancel> <ms>` | out | the slot is free again (it keeps its conversation) |
| `BSTOP <slot>` | in | end that slot at its next window |
| `BYIELD <slot>` | in | the prompt being read gives way at its next chunk boundary; its part read waits in `<slot>` (the admission's own, or a free slot for a solo request) |
| `YIELDED <slot> <tokens>` | out | before the `DONE cancel` of a read that gave way: the request is sent again later and goes on from there |
| `INFO ... batch_slots=N` | out | the slots the engine runs (only with `--batch`) |

`tools/batch_test.py` drives the engine directly: the same prompts alone, then together, compared token by token,
and the aggregate rate.
