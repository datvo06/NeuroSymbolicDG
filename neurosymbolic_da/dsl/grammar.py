"""PCFG grammar over layout programs.

Builds a universal grammar (all productions up to depth D_max) and
provides a forward() method that calls DSL ops. The handler installed
at call time determines the semantics (eval, inside, symbolic).
"""

import torch
import torch.nn as nn
from torch import Tensor

from neurosymbolic_da.dsl.ops import choice, conj, group_rel, has, rel, score
from neurosymbolic_da.dsl.primitives import Env
from neurosymbolic_da.dsl.relations import (
    RELATION_NAMES,
    RELATION_NAMES_INVARIANT,
    LearnedRelationParams,
    OrthogonalLearnedRelationParams,
    RelationParams,
    ResidualRelationParams,
    canonicalize_coords,
    normalize_coords,
    transform_bbox,
)


def sparsemax(z: Tensor, dim: int = -1) -> Tensor:
    """Sparsemax activation (Martins & Astudillo, 2016).

    Like softmax but produces exactly-zero outputs for low-scoring entries.
    This gives hard sparsity — each class uses only a few productions.
    """
    z_sorted, _ = z.sort(dim=dim, descending=True)
    z_cumsum = z_sorted.cumsum(dim=dim)
    k = torch.arange(1, z.size(dim) + 1, device=z.device, dtype=z.dtype)

    # Reshape k for broadcasting
    shape = [1] * z.ndim
    shape[dim] = -1
    k = k.view(shape)

    support = (1 + k * z_sorted > z_cumsum).to(z.dtype)
    k_z = support.sum(dim=dim, keepdim=True)
    tau = (z_cumsum.gather(dim, (k_z - 1).long().clamp(min=0)) - 1) / k_z

    return torch.clamp(z - tau, min=0)


class LayoutGrammar(nn.Module):
    """A PCFG over the layout DSL.

    The grammar enumerates all productions up to a bounded depth and stores
    log-weights as parameters. forward() calls DSL ops — the active handler
    determines semantics.

    When input_conditional=True, production weights are generated dynamically
    from bottleneck features via a small network, enabling the grammar to
    select different structural explanations for different inputs.

    Args:
        n_primitives: number of primitive types (k)
        n_classes: number of output classes
        max_depth: maximum derivation depth (D_max)
        use_sparsemax: use sparsemax instead of softmax for weight normalization
        input_conditional: if True, generate per-image weights from features
        feature_dim: dimension of input features for conditional weights (k*3)
    """

    def __init__(self, n_primitives: int, n_classes: int, max_depth: int = 2,
                 use_sparsemax: bool = False, input_conditional: bool = False,
                 feature_dim: int = 24, invariant_coords: bool = False,
                 domain_conditional: bool = False, n_domains: int = 0,
                 domain_embed_dim: int = 32):
        super().__init__()
        self.n_primitives = n_primitives
        self.n_classes = n_classes
        self.max_depth = max_depth
        self.use_sparsemax = use_sparsemax
        self.input_conditional = input_conditional
        self.invariant_coords = invariant_coords
        self.domain_conditional = domain_conditional
        self.n_domains = n_domains

        # Build production list
        productions = self._enumerate_productions()
        self.n_productions = len(productions)
        self.productions = productions

        # Pre-build index tensors for vectorized relation computation
        self._build_pair_indices()

        # Log-weights: one set per class [n_classes, n_productions]
        self.log_weights = nn.Parameter(torch.zeros(n_classes, self.n_productions))

        if input_conditional:
            # Weight modulation network: features → per-class weight adjustments
            # Output shape: [n_classes * n_productions], reshaped to [n_classes, n_productions]
            # Uses a residual design: base log_weights + modulation(features)
            hidden_dim = min(256, self.n_productions)
            self.weight_net = nn.Sequential(
                nn.Linear(feature_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, n_classes * self.n_productions),
            )
            # Initialize weight_net output near zero so initial behavior = static weights
            nn.init.zeros_(self.weight_net[-1].weight)
            nn.init.zeros_(self.weight_net[-1].bias)

        if domain_conditional and n_domains > 0:
            self.domain_embeddings = nn.Embedding(n_domains, domain_embed_dim)
            self.domain_proj = nn.Linear(domain_embed_dim, self.n_productions, bias=False)
            # Zero-init: starts as shared grammar, offsets diverge during training
            nn.init.zeros_(self.domain_proj.weight)

    def _enumerate_productions(self) -> list[dict]:
        """Enumerate all productions in the universal grammar.

        Level 0: has(j) — primitive existence
        Level 1: rel(name, i, j) — primitive-to-primitive spatial relations
        Level 2: sublayout(members) — group of 2+ primitives (conj of has terms)
        Level 3: group_rel(name, sub_a, sub_b) — spatial relation between
                 disjoint sublayouts

        Levels 2-3 are only included when max_depth >= 2.
        Production count: O(k + k²|R| + C(k,2) + |R|·C(k,2)·C(k-2,2))
        """
        prods = []
        k = self.n_primitives
        rel_names = RELATION_NAMES_INVARIANT if self.invariant_coords else RELATION_NAMES

        # Level 0: has(j) for each primitive type j
        for j in range(k):
            prods.append({"type": "has", "prim": j})

        # Level 1: rel(name, i, j) for each relation and ordered pair i != j
        for name in rel_names:
            for i in range(k):
                for j in range(k):
                    if i != j:
                        prods.append({"type": "rel", "name": name, "a": i, "b": j})

        self._n_base_productions = len(prods)

        if self.max_depth >= 2:
            # Level 2: sublayouts — groups of 2 primitives
            # Each sublayout is has(i) ∧ has(j) for unordered pair {i, j}
            sublayout_list = []
            for i in range(k):
                for j in range(i + 1, k):
                    sublayout_list.append(frozenset({i, j}))
                    prods.append({"type": "sublayout", "members": frozenset({i, j})})

            self._n_sublayout_start = self._n_base_productions
            self._n_sublayout_productions = len(sublayout_list)
            self._sublayout_list = sublayout_list

            # Level 3: group_rel — spatial relation between disjoint sublayouts
            for name in rel_names:
                for ia, sa in enumerate(sublayout_list):
                    for ib, sb in enumerate(sublayout_list):
                        if sa.isdisjoint(sb):
                            prods.append({
                                "type": "group_rel",
                                "name": name,
                                "sub_a": ia,
                                "sub_b": ib,
                            })

        return prods

    def _build_pair_indices(self):
        """Pre-build index tensors for vectorized relation computation."""
        k = self.n_primitives
        pairs_a, pairs_b = [], []
        for i in range(k):
            for j in range(k):
                if i != j:
                    pairs_a.append(i)
                    pairs_b.append(j)
        self.register_buffer("_pair_a", torch.tensor(pairs_a, dtype=torch.long))
        self.register_buffer("_pair_b", torch.tensor(pairs_b, dtype=torch.long))
        self._n_pairs = len(pairs_a)

        if self.max_depth >= 2:
            # Sublayout member indices: [n_sublayouts, 2]
            sub_i, sub_j = [], []
            for members in self._sublayout_list:
                i, j = sorted(members)
                sub_i.append(i)
                sub_j.append(j)
            self.register_buffer(
                "_sub_i", torch.tensor(sub_i, dtype=torch.long)
            )
            self.register_buffer(
                "_sub_j", torch.tensor(sub_j, dtype=torch.long)
            )

            # Group_rel indices: sub_a, sub_b, relation_idx for each group_rel prod
            rel_names = RELATION_NAMES_INVARIANT if self.invariant_coords else RELATION_NAMES
            gr_sub_a, gr_sub_b, gr_rel_idx = [], [], []
            group_start = self._n_sublayout_start + self._n_sublayout_productions
            for idx in range(group_start, len(self.productions)):
                prod = self.productions[idx]
                gr_sub_a.append(prod["sub_a"])
                gr_sub_b.append(prod["sub_b"])
                gr_rel_idx.append(rel_names.index(prod["name"]))
            self.register_buffer(
                "_gr_sub_a", torch.tensor(gr_sub_a, dtype=torch.long)
            )
            self.register_buffer(
                "_gr_sub_b", torch.tensor(gr_sub_b, dtype=torch.long)
            )
            self.register_buffer(
                "_gr_rel_idx", torch.tensor(gr_rel_idx, dtype=torch.long)
            )
            self._n_group_rel = len(gr_sub_a)

    def _get_weights(self, class_idx: int) -> Tensor:
        """Get normalized weights for a class.

        With use_sparsemax=True, uses sparsemax which produces exact zeros,
        forcing each class to commit to a sparse set of productions.
        """
        if self.use_sparsemax:
            return sparsemax(self.log_weights[class_idx], dim=0)
        return torch.softmax(self.log_weights[class_idx], dim=0)

    def get_conditional_log_weights(self, features: Tensor) -> Tensor:
        """Compute input-conditional log-weights.

        Combines static per-class weights with feature-dependent modulation:
            log_weights_dynamic = log_weights_static + weight_net(features)

        Args:
            features: [B, feature_dim] bottleneck features

        Returns:
            [B, n_classes, n_productions] per-image, per-class log-weights
        """
        B = features.shape[0]
        # Static base weights: [n_classes, n_productions] → [1, n_classes, n_productions]
        base = self.log_weights.unsqueeze(0).expand(B, -1, -1)
        # Dynamic modulation: [B, n_classes * n_productions] → [B, n_classes, n_productions]
        modulation = self.weight_net(features).view(B, self.n_classes, self.n_productions)
        return base + modulation

    def forward(self, class_idx: int):
        """Run the grammar for a given class, producing a DSL expression.

        The result type depends on the active handler:
        - eval handler: Tensor (scalar score)
        - inside handler: InsideTable
        - symbolic handler: DerivNode

        With max_depth=1: flat grammar (Level 0-1 only).
        With max_depth>=2: hierarchical grammar (Level 0-3).
        """
        weights = self._get_weights(class_idx)

        # Build Level 0 + Level 1 base constraint terms
        base_terms = []
        for idx in range(self._n_base_productions):
            prod = self.productions[idx]
            w = weights[idx]
            if prod["type"] == "has":
                term = score(w, has(prod["prim"]))
            else:  # rel
                term = score(w, rel(prod["name"], prod["a"], prod["b"]))
            base_terms.append(term)

        if self.max_depth < 2:
            # Flat grammar: marginalize over base terms only
            return choice(*base_terms)

        # Level 2: sublayout terms — conjunction of has(i) ∧ has(j)
        # We need per-primitive has terms (first k base_terms)
        has_terms = base_terms[:self.n_primitives]

        sublayout_terms = []
        for idx_offset, members in enumerate(self._sublayout_list):
            prod_idx = self._n_sublayout_start + idx_offset
            w = weights[prod_idx]
            i, j = sorted(members)
            sub = conj(has_terms[i], has_terms[j])
            sublayout_terms.append(score(w, sub))

        # Level 3: group_rel terms — spatial relations between sublayouts
        group_rel_terms = []
        group_start = self._n_sublayout_start + self._n_sublayout_productions
        for idx in range(group_start, len(self.productions)):
            prod = self.productions[idx]
            w = weights[idx]
            g = group_rel(
                prod["name"],
                sublayout_terms[prod["sub_a"]],
                sublayout_terms[prod["sub_b"]],
            )
            group_rel_terms.append(score(w, g))

        # Marginalize over all levels
        all_terms = base_terms + sublayout_terms + group_rel_terms
        return choice(*all_terms)

    def _extract_primitives(self, env: Env, params: RelationParams):
        """Extract primitive fields and compute base (Level 0+1) scores.

        When invariant_coords=True, applies normalize_coords (scale invariance)
        and canonicalize_coords (rotation invariance) before computing relations,
        and adds the dist_ratio relation.

        Returns (all_scores, all_cx, all_cy, all_conf, all_x1, all_y1, all_x2, all_y2)
        where all_scores is [B, n_base_productions].
        """
        k = self.n_primitives

        # Extract all primitive fields into [B, k] tensors
        all_cx = torch.stack([env[j].cx for j in range(k)], dim=1)
        all_cy = torch.stack([env[j].cy for j in range(k)], dim=1)
        all_conf = torch.stack([env[j].conf for j in range(k)], dim=1)

        # Extract bbox coords
        all_x1 = torch.stack([env[j].x1 for j in range(k)], dim=1)
        all_y1 = torch.stack([env[j].y1 for j in range(k)], dim=1)
        all_x2 = torch.stack([env[j].x2 for j in range(k)], dim=1)
        all_y2 = torch.stack([env[j].y2 for j in range(k)], dim=1)

        # Apply coordinate transforms for invariance
        if self.invariant_coords:
            orig_cx, orig_cy = all_cx, all_cy
            all_cx, all_cy, spread = normalize_coords(all_cx, all_cy)
            all_cx, all_cy = canonicalize_coords(all_cx, all_cy)
            all_x1, all_y1, all_x2, all_y2 = transform_bbox(
                all_x1, all_y1, all_x2, all_y2,
                orig_cx, orig_cy, all_cx, all_cy, spread,
            )

        # Index into ordered pairs: [B, n_pairs]
        cx_a = all_cx[:, self._pair_a]
        cx_b = all_cx[:, self._pair_b]
        cy_a = all_cy[:, self._pair_a]
        cy_b = all_cy[:, self._pair_b]

        # Compute pairwise features (shared by learned and residual variants)
        pw_x1_a = all_x1[:, self._pair_a]
        pw_y1_a = all_y1[:, self._pair_a]
        pw_x2_a = all_x2[:, self._pair_a]
        pw_y2_a = all_y2[:, self._pair_a]
        pw_x1_b = all_x1[:, self._pair_b]
        pw_y1_b = all_y1[:, self._pair_b]
        pw_x2_b = all_x2[:, self._pair_b]
        pw_y2_b = all_y2[:, self._pair_b]

        if isinstance(params, LearnedRelationParams):
            # Fully learned: MLP replaces all hand-coded relations
            pw_features = params.compute_pairwise_features(
                cx_a, cy_a, cx_b, cy_b,
                pw_x1_a, pw_y1_a, pw_x2_a, pw_y2_a,
                pw_x1_b, pw_y1_b, pw_x2_b, pw_y2_b,
            )  # [B, n_pairs, 6]
            learned_scores = params.compute_relations(pw_features)  # [B, n_pairs, n_rels]
            rel_scores = [learned_scores[:, :, r] for r in range(learned_scores.shape[-1])]
        else:
            # Hand-coded relations
            above = torch.sigmoid(params.lambda_above * (cy_b - cy_a - params.margin_above))
            left_of = torch.sigmoid(params.lambda_left * (cx_b - cx_a - params.margin_left))
            aligned_h = torch.exp(-(cy_a - cy_b) ** 2 / (2 * params.tau_h ** 2))
            aligned_v = torch.exp(-(cx_a - cx_b) ** 2 / (2 * params.tau_v ** 2))
            dist_sq = (cx_a - cx_b) ** 2 + (cy_a - cy_b) ** 2
            near = torch.exp(-dist_sq / (2 * params.rho ** 2))

            # Contains
            margins = torch.stack([
                pw_x1_b - pw_x1_a,
                pw_y1_b - pw_y1_a,
                pw_x2_a - pw_x2_b,
                pw_y2_a - pw_y2_b,
            ], dim=-1)  # [B, n_pairs, 4]
            contains = torch.sigmoid(params.lambda_contains * margins.min(dim=-1).values)

            rel_scores = [above, left_of, aligned_h, aligned_v, near, contains]

            # dist_ratio: rotation-invariant distance (only with invariant_coords)
            if self.invariant_coords:
                dist_ratio = torch.exp(-dist_sq / (2 * params.sigma_dist ** 2))
                rel_scores.append(dist_ratio)

            # Apply residual corrections if using ResidualRelationParams
            if isinstance(params, ResidualRelationParams):
                pw_features = params.compute_pairwise_features(
                    cx_a, cy_a, cx_b, cy_b,
                    pw_x1_a, pw_y1_a, pw_x2_a, pw_y2_a,
                    pw_x1_b, pw_y1_b, pw_x2_b, pw_y2_b,
                )  # [B, n_pairs, 6]
                corrections = params.compute_residual(pw_features)  # [B, n_pairs, n_rels]
                n_base_rels = 6
                for r in range(min(n_base_rels, corrections.shape[-1])):
                    rel_scores[r] = (rel_scores[r] + corrections[:, :, r]).clamp(0, 1)

        # Concatenate: has(k) + n_rels × n_pairs = n_base_productions
        base_scores = torch.cat([all_conf] + rel_scores, dim=1)  # [B, n_base_productions]

        return base_scores, all_cx, all_cy, all_conf, all_x1, all_y1, all_x2, all_y2

    def get_production_scores(self, env: Env, params: RelationParams) -> Tensor:
        """Return raw production scores [B, n_productions] for distribution alignment.

        These are the primitive confidence and relation scores before weighting
        by learned grammar weights. Useful for MMD alignment across domains.
        """
        base_scores, *_ = self._extract_primitives(env, params)
        return base_scores

    def get_effective_log_weights(self, domain_ids: Tensor | None = None) -> Tensor:
        """Get log-weights with optional domain-conditional offsets.

        Args:
            domain_ids: [B] int tensor of domain indices, or None for base grammar

        Returns:
            If domain_ids: [B, n_classes, n_productions]
            If None: [n_classes, n_productions]
        """
        if domain_ids is not None and self.domain_conditional:
            # [B, embed_dim] -> [B, n_productions]
            offsets = self.domain_proj(self.domain_embeddings(domain_ids))
            # base [n_classes, n_prod] + offsets [B, 1, n_prod] -> [B, n_classes, n_prod]
            return self.log_weights.unsqueeze(0) + offsets.unsqueeze(1)
        return self.log_weights

    def forward_vectorized(self, env: Env, params: RelationParams,
                           features: Tensor | None = None,
                           domain_ids: Tensor | None = None) -> Tensor:
        """Fully vectorized eval — all tensor ops, no Python loops.

        Supports both flat (max_depth=1) and hierarchical (max_depth>=2) grammars.

        Args:
            env: batched Env (Primitive fields have shape [B])
            params: relation parameters
            features: [B, feature_dim] bottleneck features (required if input_conditional)

        Returns:
            class_scores: [B, n_classes] raw scores (not log-softmax)
        """
        base_scores, all_cx, all_cy, all_conf, all_x1, all_y1, all_x2, all_y2 = (
            self._extract_primitives(env, params)
        )

        if self.max_depth < 2:
            if self.input_conditional and features is not None:
                # Per-image weights: [B, n_classes, n_productions]
                cond_log_weights = self.get_conditional_log_weights(features)
                if self.use_sparsemax:
                    weights = sparsemax(cond_log_weights, dim=2)
                else:
                    weights = torch.softmax(cond_log_weights, dim=2)
                # [B, n_prod] @ [B, n_prod, n_classes] → [B, n_classes]
                # Use einsum: scores[b,p] * weights[b,c,p] → out[b,c]
                return torch.einsum('bp,bcp->bc', base_scores, weights)
            eff_lw = self.get_effective_log_weights(domain_ids)
            if eff_lw.ndim == 3:
                # Domain-conditional: [B, n_classes, n_productions]
                if self.use_sparsemax:
                    weights = sparsemax(eff_lw, dim=2)
                else:
                    weights = torch.softmax(eff_lw, dim=2)
                return torch.einsum('bp,bcp->bc', base_scores, weights)
            weights = sparsemax(eff_lw, dim=1) if self.use_sparsemax else torch.softmax(eff_lw, dim=1)
            return base_scores @ weights.T

        # Level 2: sublayout scores = conf_i * conf_j for each pair {i, j}
        # [B, n_sublayouts]
        sub_conf = all_conf[:, self._sub_i] * all_conf[:, self._sub_j]

        if self._n_group_rel == 0:
            all_scores = torch.cat([base_scores, sub_conf], dim=1)
            if self.input_conditional and features is not None:
                cond_log_weights = self.get_conditional_log_weights(features)
                if self.use_sparsemax:
                    weights = sparsemax(cond_log_weights, dim=2)
                else:
                    weights = torch.softmax(cond_log_weights, dim=2)
                return torch.einsum('bp,bcp->bc', all_scores, weights)
            eff_lw = self.get_effective_log_weights(domain_ids)
            if eff_lw.ndim == 3:
                if self.use_sparsemax:
                    weights = sparsemax(eff_lw, dim=2)
                else:
                    weights = torch.softmax(eff_lw, dim=2)
                return torch.einsum('bp,bcp->bc', all_scores, weights)
            weights = sparsemax(eff_lw, dim=1) if self.use_sparsemax else torch.softmax(eff_lw, dim=1)
            return all_scores @ weights.T

        # Level 3: group_rel scores = sub_a_score * sub_b_score * rel(agg_a, agg_b)
        # Compute aggregate centroids for each sublayout: confidence-weighted mean
        # [B, n_sublayouts]
        conf_i = all_conf[:, self._sub_i]  # [B, n_sub]
        conf_j = all_conf[:, self._sub_j]  # [B, n_sub]
        total_conf = conf_i + conf_j  # [B, n_sub]
        w_i = conf_i / (total_conf + 1e-8)
        w_j = conf_j / (total_conf + 1e-8)

        # Aggregate centroids [B, n_sub]
        agg_cx = w_i * all_cx[:, self._sub_i] + w_j * all_cx[:, self._sub_j]
        agg_cy = w_i * all_cy[:, self._sub_i] + w_j * all_cy[:, self._sub_j]

        # Aggregate bboxes (enclosing box) [B, n_sub]
        agg_x1 = torch.min(all_x1[:, self._sub_i], all_x1[:, self._sub_j])
        agg_y1 = torch.min(all_y1[:, self._sub_i], all_y1[:, self._sub_j])
        agg_x2 = torch.max(all_x2[:, self._sub_i], all_x2[:, self._sub_j])
        agg_y2 = torch.max(all_y2[:, self._sub_i], all_y2[:, self._sub_j])

        # Index into group_rel pairs: [B, n_group_rel]
        gr_cx_a = agg_cx[:, self._gr_sub_a]
        gr_cx_b = agg_cx[:, self._gr_sub_b]
        gr_cy_a = agg_cy[:, self._gr_sub_a]
        gr_cy_b = agg_cy[:, self._gr_sub_b]
        gr_x1_a = agg_x1[:, self._gr_sub_a]
        gr_x1_b = agg_x1[:, self._gr_sub_b]
        gr_y1_a = agg_y1[:, self._gr_sub_a]
        gr_y1_b = agg_y1[:, self._gr_sub_b]
        gr_x2_a = agg_x2[:, self._gr_sub_a]
        gr_x2_b = agg_x2[:, self._gr_sub_b]
        gr_y2_a = agg_y2[:, self._gr_sub_a]
        gr_y2_b = agg_y2[:, self._gr_sub_b]

        # Compute all relations for all group_rel pairs: [B, n_group_rel]
        gr_above = torch.sigmoid(
            params.lambda_above * (gr_cy_b - gr_cy_a - params.margin_above)
        )
        gr_left = torch.sigmoid(
            params.lambda_left * (gr_cx_b - gr_cx_a - params.margin_left)
        )
        gr_align_h = torch.exp(-(gr_cy_a - gr_cy_b) ** 2 / (2 * params.tau_h ** 2))
        gr_align_v = torch.exp(-(gr_cx_a - gr_cx_b) ** 2 / (2 * params.tau_v ** 2))
        gr_dist_sq = (gr_cx_a - gr_cx_b) ** 2 + (gr_cy_a - gr_cy_b) ** 2
        gr_near = torch.exp(-gr_dist_sq / (2 * params.rho ** 2))
        gr_margins = torch.stack([
            gr_x1_b - gr_x1_a, gr_y1_b - gr_y1_a,
            gr_x2_a - gr_x2_b, gr_y2_a - gr_y2_b,
        ], dim=-1)
        gr_contains = torch.sigmoid(
            params.lambda_contains * gr_margins.min(dim=-1).values
        )

        gr_rel_list = [gr_above, gr_left, gr_align_h, gr_align_v, gr_near, gr_contains]
        if self.invariant_coords:
            gr_dist_ratio = torch.exp(-gr_dist_sq / (2 * params.sigma_dist ** 2))
            gr_rel_list.append(gr_dist_ratio)

        # Stack all relations: [B, n_group_rel, n_rels]
        all_gr_rels = torch.stack(gr_rel_list, dim=-1)

        # Select the correct relation for each group_rel production: [B, n_group_rel]
        gr_rel_scores = all_gr_rels[
            :, torch.arange(self._n_group_rel, device=all_gr_rels.device), self._gr_rel_idx
        ]

        # group_rel score = sub_a_conf_product * sub_b_conf_product * relation_score
        gr_sub_a_scores = sub_conf[:, self._gr_sub_a]  # [B, n_group_rel]
        gr_sub_b_scores = sub_conf[:, self._gr_sub_b]  # [B, n_group_rel]
        gr_scores = gr_sub_a_scores * gr_sub_b_scores * gr_rel_scores

        # Concatenate all levels: [B, n_productions]
        all_scores = torch.cat([base_scores, sub_conf, gr_scores], dim=1)

        if self.input_conditional and features is not None:
            cond_log_weights = self.get_conditional_log_weights(features)
            if self.use_sparsemax:
                weights = sparsemax(cond_log_weights, dim=2)
            else:
                weights = torch.softmax(cond_log_weights, dim=2)
            return torch.einsum('bp,bcp->bc', all_scores, weights)

        eff_lw = self.get_effective_log_weights(domain_ids)
        if eff_lw.ndim == 3:
            if self.use_sparsemax:
                weights = sparsemax(eff_lw, dim=2)
            else:
                weights = torch.softmax(eff_lw, dim=2)
            return torch.einsum('bp,bcp->bc', all_scores, weights)
        weights = sparsemax(eff_lw, dim=1) if self.use_sparsemax else torch.softmax(eff_lw, dim=1)
        return all_scores @ weights.T
