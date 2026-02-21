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
    70% faster pre-consults (100 → 30 min/day)
    Burnout reduction (eliminates #1 repetitive task)

# 🚀 Solution

MedGemma-1.5-4b-it (HAI-DEF) transforms raw docs → granular templates.

# Core Innovations
    Attention-Safe Chunking (1-2 fields/chunk)
    Rigid Mask Filtering (rejects hallucinations)
    Deterministic (seed=314 + temp=0.0)

Raw Docs (.txt/.jpg) 
  ↓ Chunking (core.py)
MedGemma-1.5-4b-it (LM Studio)
  ↓ filter_output()
Clean Template

# 🛠️ Technical Implementation
Current implementation is a Proof-of-Concept

Model: unsloth/medgemma-1.5-4b-it-GGUF (Q8_K_XL)
Server: LM Studio (localhost:1234)
GPU: 8GB VRAM
Files: core.py + main.py (Gradio)

## Intended Production Pipeline

Hospital EMR 
  ↓ Nightly Airflow ETL
Dockerized Mosaic Library
  ↓ MedGemma GPU Cluster
Templates (TXT/JSON)
  ↓ Encrypted Email/SAML
Clinician Inbox

Scale: 5,000 patients/night, on-premises privacy.


# 🚀 Quick Start

bash
pip install -r requirements.txt
python main.py  # Gradio demo

Files:
    core.py - Extraction pipeline
    main.py - Interactive Gradio demo
    requirements.txt - Dependencies

🎥 Video Demo


Mosaic Clinical: From chaos → clinical truth. 
