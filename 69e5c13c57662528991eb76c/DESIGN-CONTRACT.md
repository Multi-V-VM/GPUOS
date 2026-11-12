# GPUOS lease contract: design companion

This document expands the interface and reasoning in the twelve-page IPDPS
manuscript. It is a specification for integration, not a report that these
interfaces are all implemented. The current evidence remains component-level.
No GPU experiments were added in this revision.

## Scope and central claim

The proposed initial service manages one physical device, one urgent class,
and trusted background clients. A backend may keep a cooperative worker
resident, execute finite compiler slices, or use a separately protected CUDA
context. The scheduling decision must distinguish these backends because they
return different resources at suspension.

The contribution is the coupling between an execution checkpoint, a resource
return contract, and the lifetime of storage and code. A generic `paused=true`
is insufficient: a worker can stop issuing arithmetic while retaining the CTA
allocation that urgent work needs. Likewise, a completed producer does not
make its output reclaimable while a consumer still uses it.

This contract does not provide memory isolation between arbitrary kernels in
a shared context. It does not export a driver context image, promise migration
to another GPU, or establish a deterministic scheduling bound on RTX 5090.

## Objects and tokens

| Object | Required fields | Ownership rule |
|---|---|---|
| Process | Identity, protection domain, allocations, executable registry | A runtime identity is not a hardware protection domain |
| Task | Code epoch, input/output views, access modes, dependency events, checkpoint schema | Owns references until retirement |
| Slice grant | Lease token, grant ID, program range, reduction extent, predecessor cursor | Authorizes a finite extent of work |
| Lease | Resource vector, suspension mode, expiry policy, allowed completion events | Revocation closes future grants before draining existing ones |
| Checkpoint | Task/code/schema version, completed cursor, state references | Published only after the state it names is valid |
| Acknowledgement | Lease token, covered grant frontier, completion event, resource-return mask | Covers every conflicting grant issued before revocation |

A lease token is `(process, task, executable_epoch, lease_generation)`. A grant
ID is monotonic within a generation. Reuse of a task slot increments its
identity generation; reuse of an operator table slot changes its executable
epoch. A completion from an older generation is rejected even if the physical
slot or pointer has the same numerical value.

The contract has two progress positions. The **submission cursor** records
work already enqueued and may advance before the device executes it. The
**committed cursor** records work covered by a completed stream/event or a
valid device publication. Only the latter can support a revocation
acknowledgement. The current finite-slice compiler primarily exposes a host
submission cursor; the broker waits for stream completion before releasing
its in-flight mark.

## Operations and linearization

| Operation | Preconditions | Result / linearization condition |
|---|---|---|
| `prepare` | Supported semantics, valid views, available storage and code | Validated descriptor; no permission to execute |
| `grant` | Dependencies complete, lease eligible, resource admission passes | Grant becomes visible under the same lock/ordering protocol used to close admissions |
| `revoke` | Matching live lease token | Atomically closes grants for the generation; returns a waitable operation |
| `ack` | Prior grants drained or checkpointed under backend contract | Publishes committed cursor and resource-return mask |
| `retire` | No in-flight access, dependent consumer, or code caller remains | Releases references and permits physical reuse |

`revoke` and `ack` are separate. Receiving a revoke request does not mean a
worker has finished its current access. Lease expiry has the same limitation:
it denies future authorization; it cannot retroactively cancel an issued
memory transaction. Revocation is complete only when the required mask has
returned, or when an explicit lower-layer teardown establishes an equivalent
condition.

The implemented IPC broker illustrates one admission linearization point.
Under a shared lock, a worker checks urgency and marks a slice active. Under
that same lock, the urgent client closes background admission and waits for
active slices to clear. Splitting the check and active mark into separate
unsynchronized actions would permit a worker to launch after a false idle
observation. This protocol assumes workers report completion honestly.

## Backend capability interpretation

| Backend | What can acknowledge | What the controller cannot infer |
|---|---|---|
| Cooperative resident worker | Arithmetic issue paused at a checkpoint | Registers, shared memory, and CTA slots have returned |
| Finite sliced kernel | Launch completion; CTA resources returned | Task buffers are reclaimable, or a fixed microsecond latency |
| Separate CUDA context | Completion information exposed by the driver | Application-controlled context-save format or worst-case scheduling bound |
| Proposed bounded hardware pause | The resources and deadlines named by its documented contract | Guarantees outside that contract, especially memory traffic and code migration |

A returned-CTA mask is a logical resource condition, not necessarily exclusive
ownership of a named physical SM. Hardware isolation and physical partition
controls require separate backend capabilities. A controller must preserve
capacity for its own control work: a polling kernel that cannot become
resident cannot serve as a bounded acknowledgement path.

## Safety argument and its limits

The manuscript's argument assumes correct memory ordering, trusted clients,
checked bounds, disjoint output ownership for concurrently granted slices,
and a truthful backend completion contract. It assumes no unhandled device
fault during the transition being justified.

1. **Commit before publish.** An event or cursor never refers to incomplete
   state. System-visible signals require the corresponding system scope;
   device-only synchronization does not automatically publish to the CPU.
2. **No new authorization after closure.** Closing generation `g` prevents
   subsequent grants for `g`. The controller records and drains the entire
   previously issued frontier before accepting a resource-return claim.
3. **Last consumer before reuse.** A task's output, checkpoint, and executable
   remain valid while any authorized reader or caller can still access them.

Prepared tasks begin with owned buffers and unpublished outputs. Each grant
preserves ownership. A completed commit advances progress; dependent consumers
become eligible only after publication. Closing admission prevents additional
old-generation work, while completion drains the recorded frontier. Retirement
then removes the last references before reuse. Induction over these operations
establishes the stated safety properties for the abstract protocol.

This is not a formal verification of the CUDA or Triton implementation. In
particular, a prototype that updates progress after asynchronous submission
cannot claim the abstract commit invariant until device completion is also
checked. Bitwise correctness tests and sanitizer runs cover tested executions,
not every interleaving or adversarial client.

## Crash-safe state is an additional service

An in-place FP32 accumulator is sufficient for orderly pause at completed
kernel boundaries. It is not automatically a crash-safe checkpoint. If a
kernel partially overwrites that accumulator and fails before publishing the
new cursor, the old cursor points to a mixed state that cannot safely resume.

A crash-recoverable design needs, for example, two versioned state buffers:
write the inactive version, wait for all writes, publish the version/cursor
pair, and retain the previous version until readers of that version retire.
Alternatively, replay can start from an earlier intact checkpoint if all
inputs and side effects permit it. Re-executing a pure unpublished tile and
re-executing an externally visible update have different semantics.

A process dying does not prove its GPU accesses have stopped. The supervisor
must obtain completion or context-teardown evidence before reassigning the
allocation. Device reset can invalidate all device-resident checkpoints; a
reset-surviving policy additionally needs host or other durable copies and a
restoration protocol. Neither service is established by the current IPC
prototype. The paper's word “committed” denotes publication/lifetime validity,
not persistence across power loss or device reset.

## Checkpoint feasibility

In the serialized finite-slice approximation, let:

- `tau`: useful execution time in one slice;
- `h`: amortized overhead per slice, including launch and state commit;
- `p`, `d`, `a`: publication, residual checkpoint, and acknowledgement costs;
- `L`: pause target; `epsilon`: allowed overhead fraction.

The admissible interval is

`h (1 - epsilon) / epsilon <= tau <= L - p - d - a`.

The left constraint comes from `h / (tau + h) <= epsilon`. The right assumes
at most one in-flight slice per worker and bounds the remaining useful work
plus its completion/control path. Multiple workers require a bound on the
slowest conflicting slice; multiple queued slices must be prevented or
included. Quantities need bounds under the admitted interference when used
for hard admission; averages provide only a planning estimate.

Checkpoint traffic can make `h` and `d` depend on tile geometry. Launch overlap,
cache locality, tensor precision, and concurrent scheduling can also change
them. The formula is therefore a feasibility test, not a measured GPU model.
When the interval is empty, slicing alone cannot achieve both objectives.
Possible responses are a different backend, reduced sharing, a relaxed
objective, or rejection. K tiles and CTA counts are work bounds, not time units.

## Admission and statistical scope

A candidate key includes the urgent phase and geometry, model/code version,
backend, allocation, and background declaration. Unknown combinations are
not estimated by adding isolated resource prices. Cache eviction can change
DRAM traffic; power can change clocks; partitioning can slow the foreground
before any background operand is fetched.

The existing lookup selects from measured cooperative configurations using
calibration samples and freezes its decision before held-out validation.
That provides evidence for the tested configuration. It does not imply that
future request arrivals, context lengths, routing distributions, or clocks
have the same latency distribution.

Adding component p99 values does not yield the p99 of their sum. If each
component has a validated bound `P[X_j > b_j] <= delta_j`, the union bound gives
`P[sum X_j > sum b_j] <= sum delta_j`, without independence. The total error
budget must therefore be allocated across components, or the joint completion
interval must be measured. Correlation remains relevant to performance even
though the union bound does not require independence.

Deadline protection needs both safe resource return and a time bound. Its
premises include urgent execution capacity, control-path progress, bounded
residual work, and bounded interference. The current empirical step-budget
results are not deterministic deadlines. With 20 diagnostic trials per mode,
a reported empirical p99 is close to the maximum sample and does not identify
a population tail.

## Overload, fairness, and extension points

A best-effort lease can starve when urgent demand fills the timeline. Aging
can choose which eligible background task gets a grant; it cannot create
slack. A promised minimum background rate needs an explicit reservation and
foreground admission control. Memory should not be retained indefinitely by
a lease that cannot make progress.

The initial one-urgent-class setting avoids claiming a complete scheduler for
competing hard deadlines. Extending it requires a policy for arrival admission,
urgent queueing, cancellation, and overload. EDF ordering alone would not
establish schedulability on a GPU with non-preemptible intervals and coupled
memory resources. Multi-GPU collectives additionally need coordinated grants
or safe collective boundaries; local lease safety is insufficient to ensure
job-wide progress.

## Design validation cases for future integration

These are obligations to test when implementation resumes, not completed tests.

| Case | Required observation |
|---|---|
| Urgency races with a worker grant | Exactly one lock-ordered outcome; no post-closure grant |
| Old acknowledgement arrives after slot reuse | Rejected by token/generation check |
| Host cursor advances before completion | No committed progress or return mask exposed early |
| Worker pauses while holding CTA allocation | Urgent admission accounts for the retained footprint |
| Task is cancelled during a copy | Destination allocation remains owned until copy termination |
| Failure during an accumulator update | No recovery from a partially overwritten old checkpoint |
| Code slot is replaced during execution | Old code remains loaded until prior callers retire |
| Batch/context geometry changes | Cost key invalidated or requalified before admission |
| Foreground demand remains saturated | No minimum-background-rate promise without reservation |
| Collective peer is paused | No local admission guarantee presented as distributed progress |

## Relationship to prior work

The GPU-OS resource and slack agenda is attributed to
[Nourai and Bastani](https://arxiv.org/abs/2611.08720).
[REEF](https://www.usenix.org/conference/osdi22/presentation/han) establishes
reset-based preemption for suitable inference work; saving committed partial
state has different replay assumptions.
[XSched](https://www.usenix.org/conference/osdi25/presentation/shen-weihang)
provides a portable preemptible-queue abstraction; this design specifies the
state/resource meaning that admission requires around a backend.
[Salus](https://proceedings.mlsys.org/paper_files/paper/2020/hash/d9cd83bc91b8c36a0c7c0fcca59228f2-Abstract.html)
and [Gandiva](https://www.usenix.org/conference/osdi18/presentation/xiao)
establish earlier sharing, memory-management, predictability, and placement
ideas. None of these systems is a local experimental baseline in this artifact;
no comparative speedup over them is claimed.
