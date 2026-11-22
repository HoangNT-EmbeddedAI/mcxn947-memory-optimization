# Memory-Efficient INT8 Object Detection on an NPU-Accelerated MCU

**Inference Memory Optimization · Computational Graph Scheduling · Embedded AI Performance Engineering**

**Graduation Thesis — Nguyễn Trọng Hoàng**  
Computer Engineering · VNU University of Engineering and Technology (VNU-UET) · 2026

> **Key result:** Reduced the **compiled firmware SRAM footprint by 34.7%** (**299,040 → 195,288 bytes**) for INT8 person-detection inference on the **NXP FRDM-MCXN947**, using the **eIQ Neutron NPU**. The optimized pipeline achieved **15.65 FPS**, with a documented memory–latency trade-off.

This project investigates a question central to embedded AI inference engineering: **How can a neural-network workload fit within a strict on-chip memory budget while remaining executable on a hardware accelerator and preserving useful detection quality?**

## Documentation

- [**Defense presentation (19 pages, Vietnamese)**](https://raw.githubusercontent.com/HoangNT-EmbeddedAI/mcxn947-memory-optimization/main/docs/defense_slides.pdf) — concise overview of the problem, implementation, benchmarks, and findings.
- [**Full graduation thesis (96 pages, Vietnamese)**](https://raw.githubusercontent.com/HoangNT-EmbeddedAI/mcxn947-memory-optimization/main/docs/graduation_thesis.pdf) — methodology, graph and memory analysis, experiments, and limitations.

These links download the original PDF files directly. If GitHub displays “Unable to render code block” in its PDF preview, use the links above and open the downloaded file in your browser or PDF reader.

This README is an English technical overview for engineers and reviewers. Reported measurements and conclusions are sourced from the thesis; they have **not** been independently re-benchmarked for this README.

## 1. Project at a Glance

| Item | Details |
| --- | --- |
| **Target board / MCU** | NXP FRDM-MCXN947 / MCXN947 (Arm Cortex-M33) |
| **AI accelerator** | NXP eIQ Neutron NPU |
| **Inference stack** | TensorFlow Lite Micro / NXP eIQ; Neutron Converter (`mcxn94x` target) |
| **Model** | INT8-quantized FastestDet, fine-tuned for **person detection** |
| **Input / output tensors** | `128 × 160 × 3` INT8 → `16 × 20 × 8` INT8 |
| **Core techniques** | Patch-based inference; graph partitioning; memory-aware execution scheduling; tensor-lifetime analysis; static buffer reuse |
| **Primary objective** | Reduce firmware SRAM requirements, not maximize FPS |
| **Evaluation** | Firmware memory statistics; Neutron Converter reports; latency/FPS benchmarking; detection comparison on 1,000 VOC person images |

## 2. Engineering Problem: Memory Is More Than Model Size

Quantizing a CNN to INT8 reduces the storage cost per tensor element, but it does **not** eliminate inference-time memory pressure. On an MCU, the runtime must also accommodate **intermediate activations, long-lived feature maps, TensorArena storage, accelerator scratch space, and buffers shared across execution stages**.

The baseline FastestDet graph was convertible to the target Neutron stack, yet the converter reported **289,280 bytes of `Total data`**, including **134,400 bytes of scratch**. The compiled baseline firmware occupied **299,040 bytes of SRAM**. These values describe **different accounting layers** and must not be treated as interchangeable measures of peak runtime memory.

The optimization therefore targets two distinct bottlenecks:

1. **High-resolution prefix activations:** early CNN layers process relatively large spatial feature maps.
2. **Branching suffix dependencies:** later feature fusion and detection layers require overlapping tensor lifetimes, runtime scratch space, and intermediate buffers.

The design goal was to **lower simultaneous memory requirements without redesigning the learned detector architecture or changing its trained weights**.

## 3. Inference Architecture

```text
                         Input: 128 × 160 × 3 (INT8)
                                      │
                                      ▼
                     4 × 4 overlapping input patches
                         (receptive-field halo)
                                      │
                                      ▼
                    FastestDet prefix on Neutron NPU
                      (16 sequential patch passes)
                                      │
                                      ▼
                   Crop valid outputs + stitch features
                        32 × 40 × 36 (INT8)
                                      │
                                      ▼
                   Partitioned FastestDet suffix graph
                 ┌───────────────────────────────────┐
                 │ Dependency-aware execution order  │
                 │ Tensor-lifetime-aware buffer reuse │
                 │ NPU-executable submodel invocation │
                 └───────────────────────────────────┘
                                      │
                                      ▼
                       Output: 16 × 20 × 8 (INT8)
                                      │
                                      ▼
                 CPU: box decoding → thresholding → NMS
                                      │
                                      ▼
                              Person detections
```

**Hardware/software boundary:** the CNN inference stages use the Neutron inference stack; detection post-processing is described on the CPU. This is a **board-level inference benchmark**, not a production automotive camera-to-application pipeline.

## 4. Optimization Strategy

### 4.1 Profile the graph and identify the actual bottleneck

The analysis combined tensor shapes and dependencies with **Neutron Converter memory reports** and **compiled firmware SRAM/Flash statistics**. This separated theoretical activation estimates from converter-reported runtime data and final firmware footprint.

For the baseline graph, the conversion logs reported **92/92 built-in operators converted** and **45 Neutron operators**. Conversion compatibility did not, by itself, guarantee an acceptable SRAM footprint or inference latency.

### 4.2 Reduce prefix activation memory through spatial tiling

Rather than running the early CNN layers on the complete image, the selected configuration splits execution **after `stage2_1`** and processes a **4 × 4 grid of overlapping patches**.

| Selected prefix configuration | Shape / value |
| --- | ---: |
| Input patch | `52 × 60 × 3` INT8 |
| Prefix output per patch | `13 × 15 × 36` INT8 |
| Valid output retained per patch | `8 × 10 × 36` INT8 |
| Reconstructed feature map | `32 × 40 × 36` INT8 |
| Estimated peak activation memory | **150 → 67.5 KiB** (approximately **55% lower**) |
| Estimated computational overhead | **+11.1% MACs** |

**Why overlap matters:** the receptive-field *halo* supplies neighboring pixels needed for correct convolutions at patch boundaries. Invalid edge regions are cropped before valid features are stitched into the full intermediate feature map.

The reported prefix output and suffix input share the same INT8 quantization parameters (`scale ≈ 0.0583019`, `zero_point = -122`), avoiding an additional **requantization at that interface**. Patch extraction and feature stitching still incur data-movement overhead.

**Converter-level effect:** `Total data` for the converted prefix fell from **235,520 to 35,880 bytes** (**84.8% lower**). This is a **prefix-only converter metric**, **not** the end-to-end firmware SRAM reduction.

### 4.3 Partition and schedule the suffix with tensor lifetimes in mind

Prefix tiling alone was insufficient: the intact suffix still required **280,680 bytes of converter-reported `Total data`**. The suffix was therefore partitioned into **Neutron-executable submodels** and analyzed as a directed acyclic graph (DAG).

The scheduling approach considers:

- **Topological dependencies:** execute a submodel only after its inputs are available.
- **Live tensors:** retain an intermediate tensor until its **last consumer** has finished.
- **Submodel memory cost:** account for runtime-specific memory reported by the conversion tool, not only tensor output sizes.
- **Safe buffer reuse:** reuse a physical region only when its previous contents are no longer required and size/alignment/read–write constraints are satisfied.

A subset-state dynamic-programming formulation was used in the scheduling analysis to minimize the maximum estimated memory requirement across valid execution orders. The reported **analytical step-peak** decreased from approximately **202.5 KiB (conservative)** to **180.0 KiB (optimized-copy)**, or **11.1%**. This estimate is **distinct from** the compiled firmware SRAM result below.

## 5. Hardware Benchmark and Trade-offs

The thesis compares the full-graph baseline and optimized implementations on the **FRDM-MCXN947**.

| Metric | Baseline | Optimized | Change |
| --- | ---: | ---: | ---: |
| **Compiled firmware SRAM footprint** | **299,040 B** | **195,288 B** | **−103,752 B (−34.7%)** |
| SRAM occupancy (as reported) | 76.00% | 49.66% | −26.34 percentage points |
| Program Flash footprint | 335,976 B | 361,084 B | **+7.47%** |
| Inference latency | 36.052 ms | 63.892 ms | **+77.22%** |
| Inference throughput | 27.73 FPS | 15.65 FPS | **−43.56%** |

**Interpretation:** the optimization achieved its **memory-capacity objective**, but incurred extra patch processing, multiple submodel invocations, crop/stitch operations, and execution-orchestration overhead. It should **not** be described as an inference speedup.

For the optimized pipeline, the thesis reports **47.858 ms of NPU invoke time** and **16.034 ms of CPU/TFLM orchestration overhead**, summing to **63.892 ms**. These are **timing components**, not hardware-counter measurements of accelerator or CPU utilization.

**Measurement caveat:** the 34.7% result compares **compiled firmware SRAM statistics with major buffers statically allocated**. It is not a measured worst-case stack/heap high-water mark or a claim about power consumption.

## 6. Detection Quality Validation

The optimized MCU pipeline was compared with a **TensorFlow Lite reference** using **1,000 VOC person images**. The thesis evaluates raw-output behavior and **precision, recall, and F1-score trends** over confidence thresholds, including CPU-side decoding and non-maximum suppression.

The reported curves were **comparable within the evaluated conditions**. This supports preservation of useful detection behavior, but does **not** establish bit-exact equivalence, zero accuracy loss, or a quantified AP/mAP improvement.

## 7. Relevance to Embedded AI Inference Engineering

This project demonstrates engineering work that maps to core responsibilities in embedded AI inference roles:

| Engineering competency | Evidence in this thesis |
| --- | --- |
| **Inference deployment on an accelerator** | INT8 FastestDet execution using the MCXN947 eIQ Neutron stack |
| **Computational graph analysis** | Prefix/suffix splitting, data dependencies, executable submodels |
| **Memory footprint optimization** | Activation reduction, live-tensor analysis, SRAM-constrained buffer reuse |
| **Runtime/operator awareness** | Neutron Converter compatibility, scratch and `Total data` reports |
| **Profiling and bottleneck diagnosis** | Converter statistics, firmware SRAM/Flash footprint, latency/FPS breakdown |
| **End-to-end engineering trade-offs** | Quantified SRAM savings versus latency/throughput and Flash cost |
| **Correctness validation** | Reference-model comparison on 1,000 person-detection images |

**Scope boundary:** this thesis uses **TFLM/eIQ and Neutron NPU on an MCU**. The graph, tensor-lifetime, memory-accounting, and profiling principles are **transferable to larger SoC inference systems**, but this work does **not** claim hands-on deployment with Qualcomm QNN, TensorRT, TIDL, ONNX Runtime, GPU/DSP pipelines, DMA/zero-copy, or embedded Linux.

## 8. Limitations and Next Engineering Steps

- **Memory–latency trade-off:** reducing SRAM increased the number of execution stages and lowered throughput from 27.73 to 15.65 FPS.
- **Model/platform scope:** experiments focus on a fixed-input, person-only FastestDet model on a single MCU/NPU platform.
- **Integration scope:** a live camera sensor, production ADAS application, and full sensor-to-application latency were outside the reported implementation.
- **Measurement scope:** DDR/cache bandwidth, accelerator utilization counters, power/energy, and worst-case dynamic stack/heap usage were not reported.
- **Engineering opportunities:** reduce patch/copy overhead, benchmark alternative splits under a hard FPS/SRAM budget, automate submodel planning, and validate memory-transfer/synchronization behavior on a target SoC runtime.

These are **future directions**, not features claimed as already implemented.

## 9. Academic Information

| Field | Details |
| --- | --- |
| **Author** | Nguyễn Trọng Hoàng |
| **Program** | Computer Engineering |
| **Institution** | VNU University of Engineering and Technology, Vietnam National University, Hanoi |
| **Supervisor** | Dr. Hoàng Gia Hưng |
| **Year** | 2026 |
| **Recognition** | Highest-rated graduation thesis **within the author's defense committee** (as reported by the author) |

### Further Reading

- [Defense presentation — 19 pages](https://raw.githubusercontent.com/HoangNT-EmbeddedAI/mcxn947-memory-optimization/main/docs/defense_slides.pdf)
- [Full graduation thesis — 96 pages](https://raw.githubusercontent.com/HoangNT-EmbeddedAI/mcxn947-memory-optimization/main/docs/graduation_thesis.pdf)

*Documents are in Vietnamese. Direct-download links are provided to avoid GitHub PDF preview errors.*
