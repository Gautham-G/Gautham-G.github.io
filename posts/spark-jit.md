# Plan First, Execute Later

*The same pattern keeps showing up — in compilers, query engines, and now AI agents.*

---

## Why I'm Writing This

I spend my days in two worlds. As a data engineer, I write Spark jobs that chew through terabytes of data across clusters. As an ML engineer, I write JAX code that trains models on my local machine. For a long time, these felt like completely separate disciplines — different tools, different communities, different ways of thinking.

Then one day I noticed something. Both systems do the same weird thing: you tell them what to do, and they *don't do it*. They nod, take notes, and wait. Only when you explicitly ask for a result do they spring into action — but by then, they've quietly rewritten your instructions into something far more efficient.

Spark calls this **lazy evaluation**. JAX calls it **JIT compilation**. The terminology is different, the domains are different, but the underlying principle is identical: **if you see the full picture before you start, you can plan a smarter path through it.**

What's more, this same principle has recently resurfaced in a completely different field — AI agents — where adding a "planning step" before execution dramatically improves performance. It turns out "think before you act" isn't just good life advice. It's a design pattern.

Let's unpack how each system does this, what their optimizers look like under the hood, and why this idea keeps showing up everywhere.

---

## Part 1: JAX JIT — "Let Me See the Whole Function First"

### What Happens When You Write `@jit`

When you decorate a function with `@jax.jit`, JAX doesn't compile it immediately. It waits until the first time you call it. Then something interesting happens:

```python
import jax
import jax.numpy as jnp

@jax.jit
def my_function(x):
    y = jnp.sin(x)
    z = y * 2 + 1
    return jnp.sum(z)

# Nothing compiled yet...
result = my_function(jnp.ones(1000))  # NOW it compiles and runs
```

On that first call, JAX performs a process called **tracing**. Instead of passing your actual data through the function, it passes special **tracer objects** — placeholder values that record every operation performed on them. The output of tracing is a **jaxpr** (JAX Program Representation): a simple, functional intermediate representation of your computation.

Think of it like a recipe. Instead of cooking the meal immediately, JAX first writes down every step: "take the sine, multiply by 2, add 1, sum everything." Once the full recipe is written, it can be optimized.

### From Recipe to Machine Code: The XLA Pipeline

The jaxpr gets handed to **XLA** (Accelerated Linear Algebra), Google's optimizing compiler. XLA transforms it through several stages:

**1. HLO (High Level Optimizer) IR** — The jaxpr is lowered into XLA's own intermediate language. This is where the real optimization happens.

**2. Optimization Passes** — XLA applies hundreds of transformation passes. The most important ones:

- **Operator Fusion**: This is the crown jewel. XLA looks at sequences of operations (say, `sin → multiply → add → sum`) and merges them into a *single kernel*. Why does this matter? Because every separate operation would normally mean: read data from memory → compute → write result back to memory → read it again for the next operation. Fused operations keep intermediate values in CPU registers or GPU shared memory, never touching main memory for the in-between steps. This alone can yield 2-10x speedups.

- **Memory Layout Optimization**: XLA figures out the best way to arrange your tensors in memory for the target hardware (row-major vs column-major, tiling for GPU warps, etc.).

- **Constant Folding**: If part of your computation only depends on constants, XLA computes those at compile time.

- **Dead Code Elimination**: Any computation whose result is never used gets removed entirely.

**3. Code Generation** — The optimized HLO graph is compiled to actual machine code for your specific hardware (x86 CPU, NVIDIA GPU, Apple Silicon, TPU).

### Visualizing the Optimization

Here's what fusion looks like conceptually:

```
Before fusion (naive execution):
  [Read x from memory]
  → sin(x) → [Write to memory]
  → [Read from memory] → multiply by 2 → [Write to memory]
  → [Read from memory] → add 1 → [Write to memory]
  → [Read from memory] → sum → [Write result]

  Memory round-trips: 4 reads + 4 writes = 8 transfers

After fusion (XLA optimized):
  [Read x from memory]
  → sin → multiply → add → sum (all in registers)
  → [Write result]

  Memory round-trips: 1 read + 1 write = 2 transfers
```

For large arrays, those eliminated memory transfers are *enormous*. Modern hardware can compute arithmetic orders of magnitude faster than it can shuffle data between compute units and memory. This is why fusion is so powerful — it turns a memory-bandwidth-bound computation into a compute-bound one.

---

## Part 2: PySpark — "Let Me See the Whole Query First"

### What Happens When You Chain Transformations

PySpark does something remarkably similar, but in the world of distributed data processing:

```python
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, year, avg

spark = SparkSession.builder.getOrCreate()

# None of this executes yet!
result = (
    spark.read.parquet("s3://massive-dataset/sales/")
    .filter(col("country") == "US")
    .filter(year(col("date")) == 2025)
    .groupBy("product_category")
    .agg(avg("revenue").alias("avg_revenue"))
    .orderBy(col("avg_revenue").desc())
)

# Still nothing. Spark just built a plan.
result.show()  # NOW it executes
```

Every transformation (`.filter()`, `.groupBy()`, `.agg()`) is **lazy** — Spark records what you want but doesn't do anything. It builds a **logical plan**: an abstract tree of operations. Only when you call an *action* (`.show()`, `.collect()`, `.write()`) does Spark execute.

### The Catalyst Optimizer: Spark's Brain

Before execution, Spark's **Catalyst optimizer** transforms your logical plan through four phases:

**Phase 1 — Analysis**: Resolves column names and types, validates the plan.

**Phase 2 — Logical Optimization**: This is where the magic happens. Key rules include:

- **Predicate Pushdown**: "You're filtering for `country == 'US'` *after* reading 50TB of data? Let me push that filter down to the data source so we only read the US partition in the first place." This can turn a 50TB read into a 2TB read.

- **Column Pruning**: "Your query only uses `country`, `date`, `product_category`, and `revenue`, but the table has 200 columns? Let me only read those 4." With columnar formats like Parquet, this means physically skipping most of the data on disk.

- **Constant Folding**: "You're computing `year(date) == 2025` on every row? Let me pre-compute the constant part."

- **Join Reordering**: If you have multiple joins, Catalyst figures out which order produces the smallest intermediate datasets.

**Phase 3 — Physical Planning**: Catalyst generates multiple candidate execution plans (e.g., broadcast hash join vs. sort-merge join) and uses cost-based optimization to pick the cheapest one.

**Phase 4 — Code Generation (Whole-Stage Codegen)**: Similar to XLA's operator fusion! Spark compiles chains of physical operators into a single Java function using the Janino compiler. Instead of each operator calling the next through virtual method dispatch, the whole pipeline becomes one tight loop over the data. This is Spark's version of "kernel fusion."

### Visualizing the Optimization

```
Your logical plan:
  Read ALL columns from ALL partitions
  → Filter country == 'US'
  → Filter year(date) == 2025
  → GroupBy product_category
  → Aggregate avg(revenue)
  → Sort

Optimized physical plan:
  Read ONLY (country, date, product_category, revenue)
  FROM ONLY the country='US' partition
  WHERE year(date) == 2025          ← predicate pushed to scan
  → HashAggregate(product_category, avg(revenue))
  → Sort

  All operators compiled into a single codegen function.
```

The optimized version might read 1% of the data the naive version would touch.

---

## Part 3: The Parallel — Same Pattern, Different Worlds

Here's the structural similarity laid out:

| Concept | JAX JIT | PySpark |
|---|---|---|
| **Deferred execution** | Tracing (records ops, doesn't execute) | Lazy transformations (builds plan, doesn't execute) |
| **Intermediate representation** | jaxpr → XLA HLO | Logical plan → Physical plan |
| **Key optimization** | Operator fusion (reduce memory transfers) | Predicate pushdown (reduce data read) |
| **Code generation** | XLA compiles to hardware-specific machine code | Whole-Stage Codegen compiles to JVM bytecode |
| **When it runs** | First function call with concrete data | Action called (`.show()`, `.collect()`) |
| **What it optimizes** | *How* to compute (arithmetic efficiency) | *What* data to move (I/O efficiency) |

The last row is the critical difference. JAX and Spark are solving fundamentally different bottleneck problems:

- **JAX/XLA** lives in a world where *compute and memory bandwidth* are the bottleneck. Your data is already local (on your GPU, on your Mac's unified memory). The question is: how do we minimize the number of times data bounces between compute units and memory?

- **PySpark/Catalyst** lives in a world where *data volume and network I/O* are the bottleneck. Your data is spread across a cluster of machines (or sitting in a data lake). The question is: how do we avoid reading and shuffling data we don't need?

Both answer the same meta-question: **"Given the full picture of what the user wants, what work can we skip?"** XLA skips memory transfers. Catalyst skips disk reads and network shuffles.

---

## Part 4: What About Matrix Multiplication?

The original question was: could we compare matrix multiplication on PySpark (querying from big data) vs JAX JIT on a Mac? Let's think about why this comparison is more nuanced than it first appears.

### JAX: Built for This

Matrix multiplication is JAX's bread and butter. On Apple Silicon, here's what happens:

```python
import jax
import jax.numpy as jnp

@jax.jit
def matmul(A, B):
    return A @ B

A = jnp.ones((2048, 2048))
B = jnp.ones((2048, 2048))
result = matmul(A, B)  # Compiles and runs on hardware-optimized BLAS
```

XLA compiles this down to optimized BLAS (Basic Linear Algebra Subprograms) calls. On Apple Silicon, this means leveraging the **AMX (Apple Matrix Coprocessor)** — dedicated silicon designed specifically for matrix operations. The entire 2048×2048 multiply stays in unified memory, with the AMX crunching through it at near-peak throughput.

### PySpark: Not Designed for This

PySpark *can* do distributed matrix multiplication via `BlockMatrix`, but it's fighting against its own design:

```python
from pyspark.mllib.linalg.distributed import BlockMatrix

# Block matrices partitioned across a cluster
# Multiplying them requires massive data shuffles between nodes
result = block_matrix_A.multiply(block_matrix_B)
```

Here's the problem: dense matrix multiplication is inherently an *all-to-all* communication pattern. Every element of the result depends on an entire row of A and an entire column of B. In a distributed setting, this means every node needs data from every other node. The network becomes the bottleneck, and Catalyst's usual tricks (predicate pushdown, column pruning) don't help — there's nothing to prune.

PySpark's own documentation acknowledges this: for local linear algebra, it recommends converting to NumPy arrays. PySpark's strength is processing *massive tabular datasets* — filtering, joining, aggregating — not dense numerical computation.

### The Right Takeaway

This isn't a story of "JAX is faster than Spark." It's a story of **choosing the right tool for the problem**:

- Need to multiply large matrices for an ML model on your machine? **JAX** (or NumPy, PyTorch, etc.)
- Need to aggregate and transform petabytes of log data across a cluster? **PySpark**
- Need to do both? Use PySpark to extract and prepare the data, then hand it to JAX for the numerical computation. This is exactly what production ML pipelines do.

The lazy evaluation pattern in both tools serves the same purpose — giving the optimizer a complete picture before execution — but the optimizations themselves target completely different bottlenecks.

---

## Part 5: The Same Idea in AI Agents

Here's where things get interesting. The exact same "plan first" principle has emerged independently in AI agent design — and the performance gains are strikingly similar.

### The Naive Agent: ReAct (Act Immediately)

Early LLM agents used a pattern called **ReAct** — Reason and Act. The agent sees a task, thinks for one step, takes one action, observes the result, thinks again, takes another action, and so on. It's essentially greedy execution: do the next reasonable thing, then figure out what comes after.

```
User: "Find the quarterly revenue for Acme Corp and compare it to industry averages."

ReAct agent:
  Think: "I should search for Acme Corp revenue."
  Act:   [search "Acme Corp revenue"]
  Observe: Got some results...
  Think: "Now I need industry averages."
  Act:   [search "industry average revenue"]
  Observe: Got some results, but wrong industry...
  Think: "Let me refine..."
  Act:   [search again...]
  ...7 more steps, some redundant, some dead ends
```

This is the agent equivalent of PySpark executing each transformation eagerly, or JAX running each operation without fusion. Every step is locally reasonable but globally inefficient. The agent might search for the same thing twice, go down dead ends it could have avoided, or miss that two sub-tasks could be parallelized.

### The Planning Agent: Think, Then Act

More recent agent architectures add an explicit **planning phase**. Before taking any action, the agent builds a structured plan — a dependency graph of sub-tasks, essentially — and then executes against it.

```
Planning agent:
  Plan:
    1. Identify Acme Corp's industry (SIC/NAICS code)
    2. Search for Acme Corp quarterly revenue (last 4 quarters)
    3. Search for industry average revenue using the code from step 1
    4. Compare and summarize (depends on 2 and 3)

  Execute:
    Steps 2 and 3 can run in parallel (both depend on 1, not each other)
    Step 4 waits for both, then synthesizes
```

The structural parallel to our earlier systems is hard to miss:

| | JAX JIT | PySpark | Planning Agent |
|---|---|---|---|
| **Deferred execution** | Tracing | Lazy transforms | Plan generation |
| **Intermediate representation** | jaxpr / HLO | Logical plan | Task dependency graph |
| **Key optimization** | Operator fusion | Predicate pushdown | Redundancy elimination, parallelization |
| **What it skips** | Memory round-trips | Unnecessary data reads | Dead-end actions, duplicate work |

The planning agent's "optimizer" isn't a compiler or a query planner — it's the LLM itself, reasoning about which steps are necessary, which depend on each other, and which can be skipped or combined. But the effect is the same: **seeing the full task before executing lets you find a better path through it.**

### Why It Works: The Same Bottleneck Logic

Each system's planning phase targets its domain's specific bottleneck:

- **JAX** plans to minimize *memory bandwidth* (the bottleneck in numerical computation)
- **Spark** plans to minimize *data I/O* (the bottleneck in distributed processing)
- **Agents** plan to minimize *LLM calls and tool invocations* (the bottleneck in agent execution — each action costs time, tokens, and sometimes money)

An agent that makes 4 well-chosen tool calls instead of 12 stumbling ones is faster, cheaper, and more accurate. Just like a fused kernel that makes 2 memory transfers instead of 8, or a Spark query that reads 2TB instead of 50TB.

---

## The Universal Pattern

The next time you encounter a system that "doesn't execute immediately," ask yourself: what representation does it build, what bottleneck does it optimize, and what work does it skip?

You'll find this pattern in TensorFlow's graph mode, PyTorch's `torch.compile`, Dask's task graphs, SQL query planners, Rust's lazy iterator chains, and now in AI agent architectures. The insight is always the same — **knowing what you want to do before you do it lets you do it smarter.**

Or more simply: plan first, execute later.
