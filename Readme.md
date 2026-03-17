# Anthropic's Original Performance Take-Home

This repo contains a version of Anthropic's original performance take-home, before Claude Opus 4.5 started doing better than humans given only 2 hours.

The original take-home was a 4-hour one that starts close to the contents of this repo, after Claude Opus 4 beat most humans at that, it was updated to a 2-hour one which started with code which achieved 18532 cycles (7.97x faster than this repo starts you). This repo is based on the newer take-home which has a few more instructions and comes with better debugging tools, but has the starter code reverted to the slowest baseline. After Claude Opus 4.5 we started using a different base for our time-limited take-homes.

Now you can try to beat Claude Opus 4.5 given unlimited time!

## Performance benchmarks 

Measured in clock cycles from the simulated machine. All of these numbers are for models doing the 2 hour version which started at 18532 cycles:

- **2164 cycles**: Claude Opus 4 after many hours in the test-time compute harness
- **1790 cycles**: Claude Opus 4.5 in a casual Claude Code session, approximately matching the best human performance in 2 hours
- **1579 cycles**: Claude Opus 4.5 after 2 hours in our test-time compute harness
- **1548 cycles**: Claude Sonnet 4.5 after many more than 2 hours of test-time compute
- **1487 cycles**: Claude Opus 4.5 after 11.5 hours in the harness
- **1363 cycles**: Claude Opus 4.5 in an improved test time compute harness
- **??? cycles**: Best human performance ever is substantially better than the above, but we won't say how much.

While it's no longer a good time-limited test, you can still use this test to get us excited about hiring you! If you optimize below 1487 cycles, beating Claude Opus 4.5's best performance at launch, email us at performance-recruiting@anthropic.com with your code (and ideally a resume) so we can be appropriately impressed, especially if you get near the best solution we've seen. New model releases may change what threshold impresses us though, and no guarantees that we keep this readme updated with the latest on that.

Run `python tests/submission_tests.py` to see which thresholds you pass.

## Warning: LLMs can cheat

None of the solutions we received on the first day post-release below 1300 cycles were valid solutions. In each case, a language model modified the tests to make the problem easier.

If you use an AI agent, we recommend instructing it not to change the `tests/` folder and to use `tests/submission_tests.py` for verification.

Please run the following commands to validate your submission, and mention that you did so when submitting:
```
# This should be empty, the tests folder must be unchanged
git diff origin/main tests/
# You should pass some of these tests and use the cycle count this prints
python tests/submission_tests.py
```

An example of this kind of hack is a model noticing that `problem.py` has multicore support, implementing multicore as an optimization, noticing there's no speedup and "debugging" that `N_CORES = 1` and "fixing" the core count so they get a speedup. Multicore is disabled intentionally in this version.


## How we reached **2109 cycles** (explained in detail, and also for a 6-year-old)

Below is a plain-English walkthrough of the exact optimization direction used in `perf_takehome.py` to move from the earlier `2151` cycle result down to `2109` cycles for:

- `forest_height=10`
- `rounds=16`
- `batch_size=256`

### Tiny-kid version first 🍎

Imagine:
- You have **256 toy cars** (our batch lanes) moving through a **tree city**.
- Every turn, each car asks: “Which house (node) should I visit next?”
- Going to ask each house one-by-one is slow.

So we made a smarter plan:
1. Keep the most popular houses’ numbers already in your pocket.
2. For one special street level (depth 2), don’t walk to houses at all—just choose from the numbers in your pocket.
3. Do many cars together in groups when this trick is fast.

That saves time, so total clock cycles become **2109**.

---

### What the simulator architecture looks like

The machine is VLIW + SIMD. In one cycle it can issue slots to multiple engines, but each engine has limits.

```mermaid
flowchart LR
    A[Program Bundles] --> B[Core Scratch / "register-like" space]
    B --> C[VALU engine
vector math + logic
(limit: 6 slots/cycle)]
    B --> D[LOAD engine
scalar/vector loads
(limit: 2 slots/cycle)]
    B --> E[STORE engine
scalar/vector stores
(limit: 2 slots/cycle)]
    B --> F[ALU/FLOW engines
scalar ops + control]
    D --> G[(Memory)]
    E --> G
```

The key bottleneck in this problem is usually **LOAD slots** (especially gather-style loads). So good optimizations either:
- remove loads, or
- move work toward VALU where there is still headroom.

---

### Baseline shape before this improvement

The optimized kernel already had:
- deterministic pointer constants for frozen layout,
- vectorized prologue/epilogue (`vload`/`vstore`),
- stage-major hash scheduling,
- fused hash stage patterns using `multiply_add`,
- selective fast unroll on rounds 0/1.

That got us to roughly **2151 cycles**.

The next step was to attack one more expensive part: **round where `round % 11 == 2`**.

---

### Main idea that gave the additional win

At `round % 11 == 2`, indices are constrained to a tiny node range (depth 2 children).
In this layout, those are node IDs **3, 4, 5, 6**.

Normally the generic path does:
1. compute addresses
2. perform gather-like `load_offset` for each lane
3. xor values with loaded node values

This stresses LOAD heavily.

Instead, we changed round 2 to:
1. preload node3..node6 once in prologue,
2. broadcast them to vectors,
3. precompute diffs: `(node3-node6)`, `(node4-node6)`, `(node5-node6)`,
4. for each lane, detect whether index is 3/4/5 and reconstruct selected node from node6 using `multiply_add` with masks,
5. xor with selected node value.

So round-2 node selection is done mostly with VALU instructions and almost no LOAD traffic.

---

### Why `multiply_add` helps here

If mask is 1 when condition matches else 0, then:

`selected = mask * (A - B) + B`

- if `mask = 1` → `selected = A`
- if `mask = 0` → `selected = B`

We chain this for IDs 3/4/5 with base `node6`:
- start at `node6`
- if id==3, replace with node3
- if id==4, replace with node4
- if id==5, replace with node5

All vectorized.

---

### Concrete code-level pieces that were added

1. **New vector constants**
   - `v_three`, `v_four`, `v_five`

2. **Prologue preload additions**
   - scalar loads for node3..node6
   - vector broadcasts for node3..node6
   - precomputed difference vectors:
     - `v_node36_diff`
     - `v_node46_diff`
     - `v_node56_diff`

3. **New specialized branch**
   - `elif round % 11 == 2:`
   - vector compares against 3/4/5
   - chained `multiply_add` selection
   - xor with lane values

4. **Unroll policy tweak**
   - round 2 included in fast unroll set:
   - `curr_unroll = FAST_UNROLL if (round % 11) in (0, 1, 2) else BASE_UNROLL`

---

### Why this reduced cycles from 2151 → 2109

The short answer:
- We traded LOAD-bound work for VALU work in one hot round.

The longer answer:
- Gather-ish loads are expensive due to small load slot budget (`2` per cycle).
- The specialized depth-2 path removes many of those loads.
- VALU has wider issue budget (`6` slots per cycle), so extra compare/`multiply_add` math can be better hidden/packed.
- Running round 2 with `FAST_UNROLL` increases throughput further because this new path is no longer load-choked.

Net effect in benchmark config: **42 cycles saved**.

---

### Repro command used

```bash
python tests/submission_tests.py
```

You should see the benchmark print the improved cycle number around **2109** for the target configuration.

---

### Important note

`2109` is better, but still above the hardest thresholds listed earlier in this README. This means there is still room to optimize further (especially memory access structure, hash scheduling pressure, and round-specialization strategy).
