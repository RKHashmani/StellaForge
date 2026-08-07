# ConStellaration data generation

[`run.sh`](./run.sh) generates
pipeline inputs from the
[`proxima-fusion/constellaration`](https://huggingface.co/datasets/proxima-fusion/constellaration)
dataset, runs them through driftless-star on HTCondor, records progress, and
archives completed batches to reduce the number of loose files on staging.

This directory is its own Pixi workspace. Its tasks delegate to the locked
`pipeline` environment in the repository root, so the experiment reuses the
same Snakemake and HTCondor dependencies without adding experiment-specific
tasks or dependencies to the main `pixi.toml`.

## Quick start

From the repository root, run:

```
bash run.sh --all
```

The individual experiment tasks are also available directly:

```bash
# Generate inputs only
pixi run --manifest-path experiments/constellaration/pixi.toml generate --limit 10

# Launch generated inputs only
pixi run --manifest-path experiments/constellaration/pixi.toml launch --cores 4

# Run the experiment tests
pixi run --manifest-path experiments/constellaration/pixi.toml test
```

If your shell is already in `experiments/constellaration/`, the shorter forms
`pixi run generate`, `pixi run launch`, `pixi run batch`, and `pixi run test`
are equivalent.

Note that continuously running the generation on htcondor might require screen/tmux.

`--all` keeps creating batches until the complete selected Hugging Face dataset
split has been run and archived. Without `--all`, one invocation processes one
batch and exits; invoking it again advances to the next batch using
`manifest.next_offset`.

The script locates the repository root, changes into it, and runs this Pixi
task with the following defaults. Before launching, it checks that this user
has no active local Snakemake/Ouroboros controller and runs Snakemake's
`--unlock` operation to clear a stale repository lock left by an interrupted
controller. The unlock uses the no-op `experiments/constellaration/unlock.smk`; it does not
run or modify `inputs/quick_run`. The script refuses to unlock when a controller
is still active.

```
pixi run --manifest-path experiments/constellaration/pixi.toml batch \
  --output-root /staging/groups/driftless_star/constellaration_runs \
  --profile executors/htcondor/profiles/htcondor-gpu \
  --container-runtime apptainer \
  --gpu-ids all \
  --loop-iters 10 \
  --max-parallel 10 \
  --keep-going \
  --cores 8
```

Extra arguments supplied to the shell script come last and override these
defaults.

## What one invocation does

1. It locks the run root so that two batch drivers cannot update the same
   manifest simultaneously.
2. It reads `manifest.json`. If an unfinished batch exists, it resumes that
   batch. Otherwise, it fetches the next batch of dataset rows and creates one input
   directory per `plasma_config_id`.
3. It keeps up to 10 configs active simultaneously. Each individual
   driftless-star controller dispatches its stage jobs through the GPU HTCondor
   profile. When one config finishes, the next pending config starts. Every
   config uses its own live `jobs/constellaration/<id>/` directory under the
   repository in `/home`, so concurrent controllers do not share an HTCondor
   unified event log and satisfy CHTC's requirement that scheduler logs not be
   written directly to `/staging`. When a controller exits, its directory is
   moved to `outputs/<id>/htcondor/attempt_N/` on staging. The submit-side
   launcher also applies a compatibility guard for the executor plugin 0.3.0
   `_event_logs` initialization bug. It sets the actual HTCondor batch name to
   `<config-id>_<rule>`, for example
   `DGDvAUqji95R8kRxZmucCg6_stage1_vmec`, while HTCondor continues assigning a
   unique numeric Cluster ID to every submitted job.
4. It updates `manifest.json` before and after every launch. A run may be
   `pending`, `running`, `succeeded`, `failed`, `interrupted`, or `archived`.
5. When every config in the batch has succeeded, it creates `run1.tar`,
   `run2.tar`, and so on. The tar is read back and checked for every expected
   config before any loose files are removed. Note this is required because
   CHTC has a file number limit.
6. After verification, the batch's loose `inputs/<id>/` and `outputs/<id>/`
   directories are deleted. The tar and manifest remain on staging.
7. With `--all`, it repeats from step 2 until the dataset row count is reached.
   The final tar may contain fewer configs than `--batch-size`.

The parallel controllers have disjoint absolute input/output trees. The batch
runner therefore disables Snakemake's repository-wide lock for these
controllers; they cannot target one another's run artifacts. With the wrapper
defaults, as many as 10 controllers can each ask the profile for up to 8 active
HTCondor jobs.

## Iterations and convergence

The wrapper currently passes `--loop-iters 10`. This is a safety limit, not a
request to always execute ten iterations. The closed-loop driver stops early
when Stage 5 reports convergence or a halt condition. If a configuration has
not converged after ten iterations, that invocation ends at the limit.

Each newly generated VMEC input uses `NITER_ARRAY = 10000` by default. This is
the maximum number of VMEC equilibrium iterations within one Stage 1 run and
is separate from the closed-loop `--loop-iters` limit.

Use a larger safety limit when needed:

```bash
./experiments/constellaration/run.sh --loop-iters 100
```

Use `--loop-iters 0` for only one forward pass through Stages 1–5, without the
closed-loop feedback process:

```bash
./experiments/constellaration/run.sh --loop-iters 0
```

## Files on staging

While a batch is active, the layout is:

```text
/staging/groups/driftless_star/constellaration_runs/
├── manifest.json
├── .batch.lock
├── inputs/
│   └── <plasma_config_id>/
│       ├── config.yaml
│       ├── dataset_row.json
│       └── stage input files
└── outputs/
    └── <plasma_config_id>/
        ├── htcondor/
        │   └── attempt_N/
        │       └── submit, event-log, stdout, and stderr files
        └── loop/
            └── pipeline iteration outputs
```

After successful batches are archived:

```text
/staging/groups/driftless_star/constellaration_runs/
├── manifest.json
├── .batch.lock
├── run1.tar
├── run2.tar
├── inputs/
└── outputs/
```

These are uncompressed tar archives. They reduce the staging file/inode count
by combining many files into one archive, but they do not significantly reduce
the number of bytes used.

## Manifest and status

`manifest.json` is the durable index of generated configuration IDs. It records:

- dataset row index and `plasma_config_id`;
- batch number and archive name;
- whether the config has been run;
- current status, attempt count, and last exit code;
- the original config path and its path inside the archive.
- the dataset's total row count and whether continuous mode reached the end.

Examples:

```bash
RUN_ROOT=/staging/groups/driftless_star/constellaration_runs

# Show every batch and its status
jq '.batches[] | {name, count, status, archive}' "${RUN_ROOT}/manifest.json"

# Inspect one configuration
jq '.runs["DGDvAUqji95R8kRxZmucCg6"]' "${RUN_ROOT}/manifest.json"
```

## Failure and restart behavior

If a config fails, the batch is not archived and its loose files remain
available for diagnosis. Run the same command again after fixing the problem:

```bash
./experiments/constellaration/run.sh --all
```

The driver resumes the unfinished batch, skips configs already marked
`succeeded`, and retries configs marked `failed`, `interrupted`, or `pending`.
It starts a new dataset batch only after the current batch has been archived.

The wrapper supplies `--keep-going`, so a failure does not prevent the other
configs in the config batch from being attempted. When invoking the Pixi
task directly, enable that behavior with:

```bash
./experiments/constellaration/run.sh --keep-going
```

## Useful overrides

```bash
# Process every dataset row as consecutive tar batches
./experiments/constellaration/run.sh --all

# Test with two generated configs
./experiments/constellaration/run.sh --batch-size 2

# Submit all batch of config controllers at once instead of keeping 10 active
./experiments/constellaration/run.sh --max-parallel 100

# Run explicit IDs instead of the next dataset rows
./experiments/constellaration/run.sh \
  --id DGDvAUqji95R8kRxZmucCg6 \
  --id DN4iUQNyzJ25VxSzdewLE9r

# Allow two concurrent jobs per allocated GPU
./experiments/constellaration/run.sh --jobs-per-gpu 2
```

The default controller concurrency is 10 and the allowed range is 1–100. The
maximum batch size is also 100. Options describing a new batch, such as
`--batch-size`, `--offset`, or `--id`, are ignored while an existing unfinished
batch is being resumed.

Continuous mode cannot be combined with `--id`, `--ids-file`, `--offset`, or
`--dry-run`. Its scope is the complete selected dataset config and split.

## Inspect or restore an archive

List an archive without extracting it:

```bash
tar -tf /staging/groups/driftless_star/constellaration_runs/run1.tar
```

Restore its input and output trees to their original locations:

```bash
RUN_ROOT=/staging/groups/driftless_star/constellaration_runs
tar -xf "${RUN_ROOT}/run1.tar" -C "${RUN_ROOT}"
```

Restoring files does not change their `archived` status in `manifest.json` and
does not automatically rerun them.
