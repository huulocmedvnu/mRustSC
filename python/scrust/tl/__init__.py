"""Tools, mirroring `scanpy.tl`.

One module per area of responsibility; this file only re-exports.
"""

from scrust.tl._cluster import leiden, louvain
from scrust.tl._de import filter_rank_genes_groups, rank_genes_groups
from scrust.tl._embedding import tsne, umap
from scrust.tl._layout import dendrogram, draw_graph, embedding_density
from scrust.tl._score import marker_gene_overlap, score_genes, score_genes_cell_cycle
from scrust.tl._trajectory import diffmap, dpt, paga

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
