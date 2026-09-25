# MLA trial 5: agent iterations

## agent log.md

# Kimi-K3 MLA decode optimization log

## iter_05
### hypothesis.md

# Hypothesis

Move the single-rank shared-branch event wait from immediately after routed
expert execution to the final three-way add. The routed latent norm and
up-projection do not read `shared_output`, so this preserves the existing
operation order and dependency requirements while allowing more overlap with
the shared GEMM. The change is restricted to the single-rank path used by the
measurement driver.
