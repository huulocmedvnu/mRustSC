"""Tools, mirroring `scanpy.tl`.

One module per area of responsibility; this file only re-exports.
"""

from silicell.tl._cluster import leiden, louvain
from silicell.tl._de import filter_rank_genes_groups, rank_genes_groups
from silicell.tl._embedding import tsne, umap
from silicell.tl._layout import dendrogram, draw_graph, embedding_density
from silicell.tl._score import marker_gene_overlap, score_genes, score_genes_cell_cycle
from silicell.tl._trajectory import diffmap, dpt, paga

__all__ = [
    "dendrogram",
    "diffmap",
    "dpt",
    "draw_graph",
    "embedding_density",
    "filter_rank_genes_groups",
    "leiden",
    "louvain",
    "marker_gene_overlap",
    "paga",
    "rank_genes_groups",
    "score_genes",
    "score_genes_cell_cycle",
    "tsne",
    "umap",
]
