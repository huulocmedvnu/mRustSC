"""Accessors, mirroring `scanpy.get`."""

from scrust.get._frames import aggregate, obs_df, rank_genes_groups_df, var_df

__all__ = ["aggregate", "obs_df", "rank_genes_groups_df", "var_df"]
