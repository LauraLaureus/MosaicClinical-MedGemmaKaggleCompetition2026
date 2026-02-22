# MedGemmaKaggleCompetition2026
Submission to MedGemma Kaggle Competition

# Mosaic Clinical

Chronologically synchronizing the patient's past to safeguard their clinical future.

# 🎯 Project Overview

Mosaic Clinical turns scattered clinical narratives into a single source of medical truth.
Team:
    Laura del Pino Díaz - NLP Engineer
    Inés del Pino Díaz - Neurologist & Stakeholder

# 🩺 The Problem

Doctors have 10-15 minutes per patient, but spend 5+ minutes pre-consult mining fragmented histories:
    Specialist reports ✓
    Lab results ✓
    Discharge summaries ✓

Result: Manual synthesis → burnout + lost eye contact.

Mosaic Clinical: 30-second synthesis → doctor's focus returns to patient.

# Impact:

- 70% faster pre-consults (100 → 30 min/day)
- Burnout reduction (eliminates #1 repetitive task)

# 🚀 Solution

MedGemma-1.5-4b-it (HAI-DEF) transforms raw docs → granular templates.

# Core Innovations
- Attention-Safe Chunking (1-2 fields/chunk)
- Rigid Mask Filtering (rejects hallucinations)
- Deterministic (seed=314 + temp=0.0)

Mosaic clinical works following the next schema:

1. Loads chronologically the patients docs: ( currently .txt/.png) 
2. Perfomrs Chunking of the summary template.
3. Extracts information using MedGemma-1.5-4b-it (served by LM Studio)
4. Filters the output to provide a clean updated summary template.

# 🛠️ Technical Implementation
Current implementation is a Proof-of-Concept

Model: unsloth/medgemma-1.5-4b-it-GGUF (Q8_K_XL)

Server: LM Studio 

GPU: 8GB VRAM (RTX 2070)

Files: core.py (main library) + main.py (Gradio UI)

## Intended Production Pipeline

Hospital EMR runs Nightly Airflow ETL with the Mosaic Clinical Library using MedGemma in a GPU Cluster or server.
Every run is provided with the "summary template", the patient's records and will deliver an email per patient directly to the clinician Inbox

Scale: 5,000 patients/night, on-premises privacy.


# 🚀 Quick Start

```
pip install -r requirements.txt
python main.py  # Gradio demo
```

Files:
- core.py - Extraction pipeline
- main.py - Interactive Gradio demo
- requirements.txt - Dependencies

# 🎥 Video Demo

[See the video introduction on Youtube](https://youtu.be/zyEoa3LcFl4)

