# NEURA

> NEURA: An agentic system for autonomous neuroimaging workflows

This repository contains the codebase for **NEURA**, an LLM-powered agentic system for autonomous neuroimaging workflow planning, execution, validation, and reporting. Given a free-text research question and a local neuroimaging workspace, NEURA grounds the task in domain knowledge, builds an analysis plan, selects and orchestrates heterogeneous tools, validates intermediate results, and produces traceable reports with linked artefacts and execution logs.

The current codebase accompanies our manuscript submission. It includes the agent framework, desktop interface, atlas resources, example analysis artefacts, and the local-tool integration layer used to connect external neuroimaging software.

## Highlights

- Natural-language neuroimaging question to executable workflow
- Dual knowledge grounding with a disease-region knowledge graph and a tool knowledge graph
- Dual-layer retrieval over local papers and online literature
- Multi-tool orchestration across structural MRI, diffusion MRI, and fMRI workflows
- Expert-constrained Mixture-of-Experts Review (MoER) for plan, statistics, and report validation
- Full run tracking with intermediate files, logs, generated code, and final reports
- Both CLI and desktop application entry points

## System Overview

NEURA follows an environment-aware workflow:

1. **Parse the task and inspect the workspace** to identify modalities, file layout, and metadata.
2. **Retrieve evidence** from local papers (`paper/`) and PubMed, then ground the question with disease- and tool-level knowledge.
3. **Generate an evidence-based plan** under neuroimaging and statistical constraints.
4. **Select and execute toolchains** through an extensible interface for local neuroimaging software and Python-based analytics.
5. **Validate and iterate** with MoER reviewers, reflection, and self-correction.
6. **Produce a report** with linked outputs, figures, tables, and execution provenance.

Representative results reported in the manuscript:

- `89.5%` planning accuracy on **NeuroEval**, an expert-curated benchmark of 110 neuroimaging tasks across 11 neurological and psychiatric domains
- Average gains over direct LLM queries of `30.5%` in planning accuracy, `25.6%` in tool selection, and `36.7%` in tool ordering
- SCA3 case studies in which NEURA identified cerebellar atrophy and abnormal diffusivity patterns consistent with known pathology

## Repository Layout

```text
NEURA/
|- main.py                 # CLI entry point
|- src/                    # Agent core, tools, knowledge modules, configs
|- desk_app/               # Desktop interface (PySide6)
|- data/                   # Local neuroimaging workspace and subject metadata
|- paper/                  # Local PDF papers for the RAG knowledge base
|- ICBM152_adult/          # Atlas and template resources
|- analysis_results/       # Example analysis artefacts included in this snapshot
|- output/                 # Existing example outputs/cache from prior runs
```

Important submodules:

- `src/agent/`: LangGraph-based workflow, state management, MoER, report auditing, task execution, run tracking
- `src/tools/`: adapters for SPM, DPABI, DSI Studio, FreeSurfer, FSL, DICOM conversion, statistics, and visualization
- `src/knowledge/`: local PDF indexing, PubMed retrieval, vector store, and dynamic knowledge graph utilities
- `desk_app/`: desktop UI for running tasks and browsing intermediate results

## Requirements

### Python

- Python `3.9+` recommended
- Install core dependencies:

```bash
pip install -r requirements.txt
```

- Install desktop UI dependencies if needed:

```bash
pip install -r desk_app/requirements.txt
```

### External neuroimaging software

For full end-to-end execution, NEURA can integrate with locally installed tools:

- MATLAB
- SPM
- DPABI
- DSI Studio
- FreeSurfer
- FSL
- dcm2niix

The current implementation is primarily configured for a **Windows + WSL2** environment:

- MATLAB / SPM / DPABI / DSI Studio are invoked from Windows
- FreeSurfer / FSL are expected through WSL2

If these tools are unavailable, parts of the pipeline can still be used for planning, knowledge retrieval, data inspection, and some script-generation workflows, but full neuroimaging execution will not be reproducible.

## Configuration

At minimum, set your SiliconFlow API key before using the LLM-backed planning and retrieval components:

```powershell
$env:SILICONFLOW_API_KEY = "your_api_key"
```

Optional API configuration:

```powershell
$env:SILICONFLOW_API_BASE = "https://api.siliconflow.cn/v1"
```

If you want to enable local neuroimaging tools, configure the paths expected by `src/config_local_tools.py`:

```powershell
$env:MATLAB_ROOT = "C:\\Program Files\\MATLAB\\R2019b"
$env:SPM_PATH = "C:\\spm12"
$env:DPABI_PATH = "C:\\DPABI"
$env:DSI_STUDIO_PATH = "C:\\dsi_studio"
$env:DCM2NIIX_PATH = "C:\\path\\to\\dcm2niix.exe"
$env:WSL_DISTRO = "Ubuntu-22.04"
$env:FSL_WSL_DISTRO = "Ubuntu"
$env:FREESURFER_HOME = "/usr/local/freesurfer/7.4.1"
$env:FSLDIR = "/usr/local/fsl"
```

The default paths in the local-tool config are machine-specific development defaults and should be overridden in a clean deployment.

## Data Layout

NEURA expects a local workspace organized by group and modality. A typical layout is:

```text
data/
|- HC/
|  |- anat/<subject_id>/*
|  |- dwi/<subject_id>/*
|  |- func/<subject_id>/*
|- SCA3/
|  |- anat/<subject_id>/*
|  |- dwi/<subject_id>/*
|  |- func/<subject_id>/*
|- data.xlsx              # optional subject-level metadata / covariates
```

Supported modality keys in the current codebase:

- `anat`
- `dwi`
- `func`

Additional notes:

- Place local methodological or review papers as PDF files in `paper/` to enable local RAG indexing.
- The repository snapshot currently includes directory structure and example outputs, but **does not include full raw clinical neuroimaging datasets**.

## Supported Tool Backends

The registry currently exposes the following analysis backends:

- `dicom_to_nifti`
- `spm_analysis`
- `dpabi_analysis`
- `dsi_studio_analysis`
- `dipy_analysis`
- `freesurfer_analysis`
- `fsl_analysis`
- `python_stats`
- `data_visualization`

Actual availability depends on your local installation and environment configuration.

## Quick Start

### CLI

Run a single research question:

```bash
python main.py "Compare cerebellar structure between SCA3 patients and healthy controls" --auto
```

Interactive mode:

```bash
python main.py --interactive
```

Show the LangGraph workflow:

```bash
python main.py --graph
```

List recent runs:

```bash
python main.py --list
```

### Desktop App

```bash
python desk_app/main.py
```

The desktop application provides a GUI for submitting questions, monitoring task status, inspecting logs, and browsing generated results.

## Outputs and Provenance

New agent runs are written under:

```text
outputs/runs/<run_id>/
```

Typical run artefacts include:

- parsed task and planning outputs
- task lists and tool selection records
- tool-specific output directories
- generated scripts
- validation results
- final reports

This repository snapshot also includes:

- `analysis_results/`: example figures, CSVs, JSON summaries, and report text
- `output/`: legacy/example output folders from prior development runs

## Reproducibility Notes

- Exact end-to-end reproduction depends on local installations of the external neuroimaging software listed above.
- PubMed retrieval and SiliconFlow-backed LLM calls require network access and valid API configuration.
- Local paper indexing depends on the PDFs placed in `paper/`, which is empty in this snapshot.
- Atlas resources included in `ICBM152_adult/` are used for region-level analysis and report validation.

## Citation

If you use this repository in academic work, please cite the accompanying manuscript:

**NEURA: An agentic system for autonomous neuroimaging workflows**

BibTeX and publication details will be added after the review/publication process.

## License

This repository is released under the `MIT` License. See [LICENSE](/I:/NEURA/LICENSE).

Third-party resources bundled in the repository may remain under their own licenses and terms. In particular, atlas/template resources, executables, and any non-original datasets or derived artefacts should be checked individually before redistribution.
