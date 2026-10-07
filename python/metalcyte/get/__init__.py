"""Helpers that return results as tables, like `scanpy.get`."""

from metalcyte.get._frames import aggregate, obs_df, rank_genes_groups_df, var_df

__all__ = ["aggregate", "obs_df", "rank_genes_groups_df", "var_df"]
