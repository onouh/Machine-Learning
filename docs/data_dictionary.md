# Data dictionary

Use the exact H2 source labels from the project requirements when documenting dataset fields. If implementation code normalizes column names, maintain a one-to-one mapping back to source labels.

The source CSV currently uses compact labels such as `ArrivalDateYear`, `ArrivalDateWeekNumber`, and `ArrivalDateDayOfMonth`; the provided handout formats some temporal labels with spaces. Verify the actual CSV header before processing. See the assignment-provided variable definitions and cite the original article:

- António, Almeida, and Nunes, “Hotel booking demand datasets,” *Data in Brief* (2019): <https://pmc.ncbi.nlm.nih.gov/articles/PMC6297060/>.
- Dataset location and setup: [data/README.md](../data/README.md).

A complete per-column profile, missingness summary, and feature-utility decision table are future work in [TASK.md](../TASK.md).
