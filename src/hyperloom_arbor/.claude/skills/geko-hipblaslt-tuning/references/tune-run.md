# geko tune — smoke, full tune, save, deploy (Steps 4–6)

## Step 4: Smoke test

```bash
docker exec $TARGET bash -c "
rm -rf /tmp/geko_smoke_test
ROCR_VISIBLE_DEVICES=0 geko --tune \
  --hipblaslt /tmp/rocm-libraries/projects/hipblaslt \
  --inline 128 2624 1 6144 B B S N N \
  --arch gfx950 \
  --devices 0 \
  --n_slots 1 \
  --workdir /tmp/geko_smoke_test \
  2>&1 | grep -E 'INFO|WARNING|ERROR|completed|DONE|ratio|Traceback'
"
```

Expected final lines:
```
GEKO:INFO [pipeline:run_optimize] Optimization workflow completed successfully!
GEKO:INFO [cli:dispatch] Tuning outputs under '/tmp/geko_smoke_test'
```

---

## Step 5: Full tune run

### Prepare workload YAML

```python
import csv, yaml

shapes = []
seen = set()
with open("your_shapes.csv") as f:
    for row in csv.DictReader(f):
        key = (row["M"], row["N"], row["K"])
        if key in seen: continue
        seen.add(key)
        shapes.append({
            "function": "matmul",
            "M": int(row["M"]), "N": int(row["N"]), "K": int(row["K"]),
            "transA": "N", "transB": "N", "batch_count": 1,
            "a_type": "bf16_r", "b_type": "bf16_r",
            "c_type": "bf16_r", "d_type": "bf16_r",
            "scale_type": "f32_r", "compute_type": "c_f32_r",
            "call_count": 1, "initialization": "trig_float",
            "rotating": 512, "use_gpu_timer": True,
            "print_kernel_info": True, "iters": 100, "cold_iters": 20,
        })

with open("geko_workload.yaml", "w") as f:
    yaml.dump(shapes, f, default_flow_style=True, sort_keys=False)
```

### Launch (background)

```bash
docker cp geko_workload.yaml $TARGET:/tmp/geko_workload.yaml

docker exec -d $TARGET bash -c "
ROCR_VISIBLE_DEVICES=0 geko --tune \
  --hipblaslt /tmp/rocm-libraries/projects/hipblaslt \
  --workload-log /tmp/geko_workload.yaml \
  --arch gfx950 \
  --devices 0 \
  --n_slots 1 \
  --workdir /tmp/geko_tune_output \
  > /tmp/geko_tune.log 2>&1
echo DONE >> /tmp/geko_tune.log
"

# Monitor
docker exec $TARGET tail -f /tmp/geko_tune.log
```

> **`--n_slots 1`**: One benchmark at a time. The default of 4 is safe with `--ipc=host`
> but use 1 when GPU is shared or you want to avoid any interference.

---

## Step 6: Save results to host

```bash
OUT=<your-output-dir>/geko_tune_$(date +%Y-%m-%d)
mkdir -p $OUT

docker cp $TARGET:/tmp/geko_tune_output/full_libs  $OUT/full_libs
docker cp $TARGET:/tmp/geko_tune_output/final_libs $OUT/final_libs
docker cp $TARGET:/tmp/geko_tune_output/results    $OUT/results
docker cp $TARGET:/tmp/geko_tune.log               $OUT/geko_tune.log
```

---

## Using tuned kernels

```bash
export HIPBLASLT_TENSILE_LIBPATH=/path/to/full_libs

python3 -m vllm.entrypoints.openai.api_server \
  --model /path/to/model ...
```

`full_libs` guarantees no regressions: for all tuned shapes it picks the best of
(tuned, reference); untouched shapes fall back to built-in hipblaslt heuristic.

---

