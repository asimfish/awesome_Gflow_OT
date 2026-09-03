# Awesome GFlowNet x Optimal Transport

[![Awesome](https://awesome.re/badge.svg)](https://awesome.re)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Papers](https://img.shields.io/badge/papers-18%20core%20%2B%202026%20additions-orange.svg)](#content)
[![Reports](https://img.shields.io/badge/deep--dive%20reports-18-green.svg)](#deep-dive-reports)
[![Translated](https://img.shields.io/badge/zh%20PDFs-18%2F18-blueviolet.svg)](#deep-dive-reports)
[![Report](https://img.shields.io/badge/report-zh%2091p%20%7C%20en%20120p-informational.svg)](#deliverables)

[English](README.md) | [中文](README_zh.md)

A curated, evidence-first reading list on the intersection of **Generative Flow Networks (GFlowNets)** and **Optimal Transport (OT)**: the theory that a GFlowNet trained to match a reward-induced target distribution, when asked to minimise its total internal edge flow on a (possibly cyclic) state graph, recovers a **Kantorovich optimal transport plan** under the shortest-path cost of that graph.

Every paper here comes with:

- a **deep-dive report** in Chinese (`reports/`), written from the PDF with theorem/table-level citations (8-section template; editorial notes flag every judgement call);
- the **original PDF** (`papers/`) and a **layout-preserving Chinese translation** produced by [SuperTranslate](https://github.com/asimfish/super_translate) with object-level QA (`papers_zh/`);
- a machine-readable metadata card (`data/meta/`), from which this README is generated (`src/generator.py`).

Venue discipline: main conference / journal / workshop / preprint are always labelled separately. A workshop paper is never written as a main-conference paper.

*Maintained by [asimfish](https://github.com/asimfish). Structure follows [awesome-ml4co](https://github.com/Thinklab-SJTU/awesome-ml4co).*
