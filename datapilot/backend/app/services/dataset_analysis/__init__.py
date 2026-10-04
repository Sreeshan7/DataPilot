"""Phase 4: analysing a dataset the user already has (User B).

Two evidence paths, kept strictly apart (see ``app.schemas.dataset_analysis``):

* the user's own file is parsed and profiled - ``DATAPILOT_PARSED``;
* a dataset known only by name or URL is researched in trusted registries - metadata
  only, never row-level claims.
"""
