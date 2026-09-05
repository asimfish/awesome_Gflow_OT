<a id="insights"></a>
### Insights and Open Problems

The cross-paper synthesis lives in [reports/INSIGHTS.md](reports/INSIGHTS.md) and the competitor matrix in [reports/COMPETITOR_MATRIX.md](reports/COMPETITOR_MATRIX.md). Headline conclusions:

1. **Internal-flow non-uniqueness is the entry point.** On a general graph, matching the terminal distribution fixes only the terminal edge flows; the backward-policy freedom (T02 Prop. 18) and, with cycles, the cone of circulations (T19 Prop. 5) leave a family of valid internal flows. Minimising total internal flow selects one, and after fixing the source marginal that one realises an OT plan (O08 Thm. 3.2).
2. **The equivalence itself is classical; the contribution is the combination.** Shortest-path-cost Kantorovich OT on a graph equals min-cost flow (Beckmann). Emitting a local policy is not new either (a min-cost edge flow normalises into one; GSBoG also learns policies). What is new is the package: non-acyclic GFlowNet minimum-flow objective + two fixed marginals + neural policy + training and execution on implicit combinatorial graphs where the cost matrix is never materialised. The 20! permutation experiment is a feasibility demo without an exact OT reference.
3. **The least-crowded follow-up is an optimality certificate.** Balance residuals certify total-variation error of the terminal distribution (N03 Thm. 3.5/3.6) but a zero residual never certifies minimal total flow (a flow can be perfectly balanced and route everything the long way). A certificate therefore needs primal-feasibility repair + a dual-feasible potential + the optimality gap (O08 Thm. 3.3 / App. A.4 weak duality); on implicit graphs a computable *global* certificate is still open. Within the literature we checked, nobody has done this for GFlowNets; classical LP posterior-error analysis must be surveyed before claiming novelty. See `reports/INSIGHTS.md` §6b.
4. **Two follow-ups are crowded.** Conditional GFlowNets that amortise OT over a family of graphs collide with ULOT (NeurIPS 2025) and UNOT (ICML 2025); entropic GFN-OT collides with the discrete / graph Schrodinger-bridge line (DDSBM, ICLR 2025; GSBoG, ICML 2026 main track, same quadrant as O08 but with no error bound).
5. **Venue reality check.** Both GFN x OT papers (O07, O08) are ICML 2026 SPIGM *workshop* papers; all three competitors are main-track. The structural moat left to GFlowNets is the implicit combinatorial graph where no cost matrix can be materialised.

<a id="deliverables"></a>
### Reports and Slides

| Deliverable | Path |
|---|---|
| Consolidated report, Chinese (104 pages) | [PDF](reports/pdf/awesome_gflow_ot_report_zh.pdf) · [Markdown](reports/AWESOME_GFLOW_OT_REPORT_zh.md) |
| Consolidated report, English (137 pages) | [PDF](reports/pdf/awesome_gflow_ot_report_en.pdf) · [Markdown](reports/AWESOME_GFLOW_OT_REPORT_en.md) |
| Summary slides (single-file HTML, 24 slides; keyboard / wheel / touch) | [slides/awesome_gflow_ot_slides.html](slides/awesome_gflow_ot_slides.html) |
| Summary slides (Beamer PDF, 29 pages incl. backup) | [slides/awesome_gflow_ot_slides.pdf](slides/awesome_gflow_ot_slides.pdf) |
| Cross-paper synthesis | [reports/INSIGHTS.md](reports/INSIGHTS.md) |
| Competitor matrix | [reports/COMPETITOR_MATRIX.md](reports/COMPETITOR_MATRIX.md) |
| 2026 trend scans | `reports/TRENDS_GFN_2026.md`, `reports/TRENDS_OT_2026.md` |

<a id="contributing"></a>
### Contributing and Citation

Add a paper by dropping a JSON card into `data/meta/` (see any existing card for the schema), optionally a report into `reports/`, then run `python3 src/generator.py`. Please keep the venue discipline (main / journal / workshop / preprint) and cite theorem or table numbers when summarising.

```bibtex
@misc{awesome_gflow_ot_2026,
  title  = {Awesome GFlowNet x Optimal Transport: a curated, evidence-first reading list},
  author = {asimfish},
  year   = {2026},
  url    = {https://github.com/asimfish/awesome_Gflow_OT}
}
```

Translation engine: [SuperTranslate](https://github.com/asimfish/super_translate). Slides tooling: [beamer-skill](https://github.com/Noi1r/beamer-skill). Writing discipline: [anti-defensive-writing](https://github.com/Kiterlin/anti-defensive-writing), [shuorenhua](https://github.com/MrGeDiao/shuorenhua). List conventions: [awesome-ml4co](https://github.com/Thinklab-SJTU/awesome-ml4co).

Licensed under [MIT](LICENSE). Paper PDFs remain under their original licenses (arXiv).
