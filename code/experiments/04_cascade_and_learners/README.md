# `04_cascade_and_learners/`

Two questions the main table cannot answer on its own: would a cascade beat one call, and is
the random forest doing the work or would any learner do.

| Table | What it answers |
|---|---|
| `01_tables/PART4_A.csv`, `PART4_B.csv` | the router against cascade arms and against alternative regression heads, on both protocols |
| `01_tables/PART4_PAIRED_*.csv` | the paired cluster-bootstrap interval for each comparison |

The router calls exactly one verifier per instance. The cascade arms here are the honest
comparison for that choice: they are allowed more than one call, and the latency column is
what that costs.

Code: [`../08_routing_code/part4_cascade_v1.py`](../08_routing_code/part4_cascade_v1.py).
Run with `./reproduce.sh cascade`, which waits on Part 3's completion marker.
