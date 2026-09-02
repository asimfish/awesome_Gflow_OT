<a id="insights"></a>
### Insights and Open Problems

The cross-paper synthesis lives in [reports/INSIGHTS.md](reports/INSIGHTS.md) and the competitor matrix in [reports/COMPETITOR_MATRIX.md](reports/COMPETITOR_MATRIX.md). Headline conclusions:

1. **Internal-flow non-uniqueness is the entry point.** Matching the terminal distribution fixes only the boundary; the cycle space of the state graph leaves a large family of valid internal flows. Minimum total flow selects one, and that one is an OT plan.
2. **The equivalence itself is classical; the contribution is the translation.** Shortest-path-cost Kantorovich OT on a graph equals min-cost flow. What GFlowNets add is an executable *local routing policy* on combinatorial graphs where the cost matrix is never materialised.
3. **The least-crowded follow-up is an error bound.** Balance residuals already certify total-variation error of the terminal distribution; extending them to OT cost gap and marginal violation (with dual potentials as certificates) has no direct competitor yet.
4. **Two follow-ups are crowded.** Conditional GFlowNets that amortise OT over a family of graphs collide with ULOT (NeurIPS 2025) and UNOT (ICML 2025); entropic GFN-OT collides with the discrete / graph Schrodinger-bridge line (DDSBM, ICLR 2025; GSBoG, ICML 2026 main track, same quadrant as O08 but with no error bound).
5. **Venue reality check.** Both GFN x OT papers (O07, O08) are ICML 2026 SPIGM *workshop* papers; all three competitors are main-track. The structural moat left to GFlowNets is the implicit combinatorial graph where no cost matrix can be materialised.

<a id="deliverables"></a>
### Reports and Slides

| Deliverable | Path |
|---|---|
| Consolidated report, Chinese (91 pages) | [PDF](reports/pdf/awesome_gflow_ot_report_zh.pdf) · [Markdown](reports/AWESOME_GFLOW_OT_REPORT_zh.md) |
| Consolidated report, English (120 pages) | [PDF](reports/pdf/awesome_gflow_ot_report_en.pdf) · [Markdown](reports/AWESOME_GFLOW_OT_REPORT_en.md) |
| Summary slides (single-file HTML, 23 slides; keyboard / wheel / touch) | [slides/awesome_gflow_ot_slides.html](slides/awesome_gflow_ot_slides.html) |
| Summary slides (Beamer PDF, 28 pages incl. backup) | [slides/awesome_gflow_ot_slides.pdf](slides/awesome_gflow_ot_slides.pdf) |
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
