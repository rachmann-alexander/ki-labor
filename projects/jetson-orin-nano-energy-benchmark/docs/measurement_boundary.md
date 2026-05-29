# Measurement Boundary

The primary energy metric is run-integrated energy per image.

Included:
- Warmup iterations
- Validation inference
- Data loading and preprocessing inside the benchmark command
- Framework execution within the logged benchmark command

Excluded:
- Power-profile setup
- Container startup
- Post-run file copying

The normalization denominator is 50,000 validation images.
