# FINAL PRESENTATION CONTENT GUIDE
## TCGA Pan-Cancer Gene Expression Classification Using Deep Learning
### BIO Final Project — Long Island University — Spring 2026
### 24 Slides | Complete Content + Talking Points

---



---

---

## SECTION 1: OPENING (Slides 1–2)

---

### Slide 1: Title Slide

**Title:** Gene Expression Cancer Classification Using Deep Convolutional Neural Networks

**Subtitle:** A Deep Learning Approach to Pan-Cancer Type Prediction from TCGA RNA-Seq Data

**Details:**
- Dataset: TCGA HiSeqV2 | 10,459 samples | 32 Cancer Types
- BIO Final Project — Spring 2026 | Long Island University
- Your Name(s) | Course Name

**Talking Points:**
- Good [morning/afternoon]. I'm [name] and today we're presenting our final project on classifying cancer types from gene expression data using deep learning.
- What we built is an end-to-end deep learning pipeline that takes a tumor's molecular profile — over 20,000 gene activity measurements — and predicts which of 32 different cancer types the patient has.
- We'll walk through why this problem matters, what we built, how we trained it, and what we achieved.

---

### Slide 2: Presentation Roadmap

**Content — 8 sections we'll cover:**
1. Motivation — why cancer classification from genes matters
2. The Dataset — TCGA, RNA-Seq, class imbalance
3. Prior Work — Mostavi and Ramirez, what they achieved and where they fell short
4. Our Key Design Decision — the READ/COAD merge
5. Model Architecture — 1D-CNN with Squeeze-Excitation blocks
6. Training Strategy — MixUp, SMOTE, ensemble, TTA, calibration
7. Results — 94.33% accuracy, Macro F1, ROC-AUC, per-class breakdown
8. Limitations and Future Work

**Talking Points:**
- Here is what we'll cover across the next 20-odd minutes.
- We start with the problem and dataset, move into related work and our key design decisions, then into the architecture and training, and finish with results and honest limitations.
- Feel free to hold questions until the end — we'll have time for that.

---

---

## SECTION 2: BACKGROUND & DATA (Slides 3–6)

---

### Slide 3: Motivation — The Problem We Are Solving

**Headline:** Correct cancer type identification guides treatment. A wrong classification can mean wrong therapy.

**Content:**
- Cancer is not one disease — it is over 100 distinct diseases sharing one trait: uncontrolled cell growth
- Each cancer type has different molecular drivers, different treatment options, and different prognoses
- Traditional diagnosis: tissue biopsy reviewed by a pathologist under a microscope
- Limitations of traditional diagnosis:
  - Some tumors look similar visually but behave differently at the molecular level
  - Biopsy is invasive and not always possible
  - Inter-observer variability between pathologists
  - Precision oncology requires molecular-level classification, not just visual
- The opportunity: RNA sequencing measures 20,000+ gene activity levels simultaneously — a comprehensive molecular fingerprint
- Deep learning can read that fingerprint and reliably identify cancer type

**Visual:** Simple split — left shows "microscope/biopsy" with limitations, right shows "RNA-Seq + Model" with benefits

**Talking Points:**
- Cancer is not one disease. It is over 100 different diseases that all share the basic trait of uncontrolled cell growth. But beyond that, a breast tumor and a leukemia are fundamentally different — different cells, different genes, different treatments.
- The standard way to diagnose cancer today is through a biopsy — a pathologist examines tissue under a microscope. This works, but it has limits. Some cancers look similar visually even though they need completely different treatments. Getting it wrong has serious consequences.
- Gene expression profiling gives us something more fundamental — it tells us which genes are active in the tumor. Different cancer types have consistently different activity patterns, and that is what our model learns to identify.

---

### Slide 4: What Is Gene Expression?

**Headline:** RNA-Seq gives us 20,530 gene activity values per patient — the molecular fingerprint of their tumor.

**Content:**
- Every cell in the body has the same DNA. What makes a liver cell different from a tumor cell is which genes are actively being used — this is gene expression
- RNA sequencing (RNA-Seq) measures gene expression at scale:
  - Take a tumor biopsy sample
  - Sequence all RNA molecules present in the cell
  - Count how many transcripts each gene produced
  - Result: a vector of 20,530 numbers — one per gene — representing the tumor's molecular activity level
- Why this works for classification:
  - A prostate tumor massively over-expresses KLK3 (the PSA gene)
  - A thyroid tumor has strong thyroid-specific gene patterns
  - A leukemia looks completely different from any solid tumor
  - These signatures are consistent enough that a deep learning model can learn them
- Our model input: one 20,530-dimensional gene expression vector per patient sample

**Visual:** Simple flow — DNA → RNA transcription → RNA-Seq counting → 20,530-value vector → model input

**Talking Points:**
- RNA-Seq is a technology that measures the activity level of every gene in a cell at once.
- When we run it on a tumor sample, we get 20,530 numbers — one per gene — describing exactly how active each gene was in that tumor. That is our input.
- Different cancer types have very different activity patterns. A prostate cancer cell over-expresses the PSA gene hundreds of times more than a normal cell. A thyroid cancer cell has completely different thyroid-specific patterns. These differences are large and consistent — and a deep learning model can learn to use them as reliable identifiers.

---

### Slide 5: The TCGA Dataset

**Headline:** 10,459 patient samples, 20,530 genes, 32 cancer types — the gold standard pan-cancer dataset.

**Content:**
- Source: The Cancer Genome Atlas (TCGA) — NIH-funded, publicly available via UCSC Xena Hub
- HiSeqV2 RNA-Seq dataset — log2(RPKM+1) normalized gene expression
- Key statistics:
  - 10,459 patient tumor samples
  - 20,530 gene expression features per sample
  - 32 cancer types (after our biologically justified merge — explained next)
- Spans all major organ systems: brain, lung, breast, colon, blood, kidney, skin, liver, pancreas, prostate, thyroid, and more
- Same dataset used by Mostavi et al. (2020) and Ramirez et al. (2020) — our results are directly comparable
- Stored in HDF5 format for efficient large-matrix access during training
- Split: 70% training / 15% validation / 15% test — stratified by cancer type

**Visual:** Summary table — sample count, feature count, class count, source, split percentages

**Talking Points:**
- Our dataset is from The Cancer Genome Atlas — TCGA — one of the most comprehensive cancer genomics programs ever done. Over 11,000 patient samples across 33 cancer types, all profiled with RNA-Seq.
- We used the HiSeqV2 dataset specifically: 10,459 samples with 20,530 gene expression values each, covering what became 32 cancer types after a design decision we'll explain in a moment.
- One key reason we chose this dataset besides being the gold standard is that both prior papers we compare against used the same data. So our performance numbers are directly comparable with no dataset difference as a confound.

---

### Slide 6: The Class Imbalance Challenge

**Headline:** The dataset has a 27:1 imbalance — largest class has 1,218 samples, smallest has 45. We have to fix this.

**Content:**
- Class distribution is severely unequal across the 32 cancer types
- Largest class: BRCA (Breast Invasive Carcinoma) — 1,218 samples
- Smallest class: CHOL (Cholangiocarcinoma) — 45 samples
- Imbalance ratio: 27:1
- Why this is a machine learning problem:
  - Model trained on imbalanced data focuses learning on common classes
  - Rare cancer types receive very few gradient updates during training
  - Result: high accuracy on common cancers, near-failure on rare ones
  - This defeats the purpose — we need the model to work for every cancer type
- Our solution: SMOTE (Synthetic Minority Oversampling Technique)
  - Any training class with fewer than 400 samples gets synthetic samples generated
  - New samples created by interpolating between existing real samples (k=3 nearest neighbors)
  - Brings all classes up to a minimum of 400 training examples
  - Applied to training data only — validation and test never touched

**Visual:** Horizontal bar chart showing sample counts per cancer type (class_distribution.png from your plots/ folder)

**Talking Points:**
- One of the first things you notice in this dataset is how imbalanced it is. BRCA — breast cancer — has over 1,200 samples. But cholangiocarcinoma, a rare bile duct cancer, has only 45.
- That is a 27-to-1 ratio. If you just train on raw data without fixing this, the model will learn to classify BRCA very well because 1,200 samples give it plenty of signal, and it will essentially give up on CHOL because 45 samples barely move the gradient.
- We fixed this with SMOTE. Rather than just copying existing samples — which would cause the model to memorize them — SMOTE generates new synthetic samples by interpolating between real ones. Every class gets at least 400 training examples.

---

---

## SECTION 3: PRIOR WORK & OUR KEY DECISION (Slides 7–9)

---

### Slide 7: Related Work — What Came Before Us

**Headline:** Two strong prior works on the same dataset — both with the same failure case.

**Content:**

**Mostavi et al. (2020) — BMC Medical Genomics:**
- Method: Single 1D-CNN trained on TCGA gene expression (33 classes)
- Key result: CNNs outperform SVM, random forests, and other classical ML methods
- Validated known cancer biomarkers in learned representations (GATA3 for breast, KLK3 for prostate)
- Critical gap: READ (rectum adenocarcinoma) F1 = 0.40 — worst class by a large margin, model misclassified almost half of READ samples
- No ensemble, no test-time augmentation, no calibration

**Ramirez et al. (2020) — Frontiers in Physics:**
- Method: Graph Convolutional Neural Network (GCNN) using protein-protein interaction networks + gene expression
- Key idea: Encode known biological gene-gene interactions as a graph structure
- Despite richer biological input, showed the same READ/COAD confusion pattern
- Critical gap: Even with PPI structural priors, the READ/COAD boundary could not be learned

**The shared insight:**
- When two completely different methods both fail on the same class pair, the problem is in the labels, not the models
- READ and COAD are transcriptomically indistinguishable — this is a label problem, not a modeling problem

**Talking Points:**
- Before explaining our model, it's worth understanding what already existed.
- Mostavi et al. in 2020 published a CNN approach on the same TCGA dataset. Strong work — they showed CNNs outperform classical ML and their model even recovered known cancer biomarkers. But they had one bad failure: rectum adenocarcinoma, READ, had F1 of only 0.40. Nearly half their READ samples were being classified wrong.
- Ramirez et al. tried a graph neural network that adds biological protein interaction data. Even more sophisticated, but the same READ/COAD confusion showed up.
- When two completely different methods fail on the same pair, that tells you the problem is in the labels, not the architecture. That observation is what drove our most important design decision.

---

### Slide 8: The READ/COAD Problem — The Root Cause

**Headline:** TCGA labeled colon and rectum cancer separately. The molecular data says they are the same disease.

**Content:**
- COAD = Colon Adenocarcinoma | READ = Rectum Adenocarcinoma
- Both arise from colorectal epithelial tissue — adjacent sections of the same organ
- At the gene expression level, READ and COAD are indistinguishable:
  - Hoadley et al. (2018) — the definitive TCGA Pan-Cancer molecular atlas — shows READ and COAD form a single transcriptomic cluster across every analysis
  - No gene expression boundary exists between them
- Clinical reality: oncologists do not distinguish the two for treatment
  - Same TNM staging system
  - Same chemotherapy: FOLFOX, FOLFIRI
  - Same targeted agents: bevacizumab (anti-VEGF), cetuximab (anti-EGFR)
- Why TCGA labeled them separately: anatomical/clinical records from hospital charts, not molecular profiles
- Result in prior models: model tries to learn a boundary that does not exist in the data → F1 = 0.40 for READ

**Visual:** Simple diagram — COAD and READ samples overlapping in gene expression space (no separable cluster boundary)

**Talking Points:**
- Here is the root cause of the READ problem.
- COAD is colon adenocarcinoma and READ is rectum adenocarcinoma. Both are colorectal cancers — same tissue type, same molecular drivers, same treatment. The Hoadley 2018 pan-cancer study, which is the most comprehensive molecular analysis of TCGA ever done, confirmed that READ and COAD form a single cluster in gene expression space. There is no boundary between them.
- TCGA labeled them separately because hospital records do — it is an anatomical distinction. But at the molecular level it is one disease.
- So when a model tries to learn to separate READ from COAD based on gene expression, it is trying to find a boundary that does not exist. It guesses. That gives you F1 = 0.40.

---

### Slide 9: Our Key Decision — Merging READ into COAD

**Headline:** We merged READ into COAD to create one colorectal class — biologically justified, and it works.

**Content:**

**The decision:** Relabel all READ samples as COAD → 33 classes become 32 (CRC = Colorectal Adenocarcinoma)

**Before vs. After:**
- Before (Mostavi): 33 classes, READ F1 = 0.40 — worst class in the entire benchmark
- After (ours): 32 classes, CRC F1 = 0.9924 — near-perfect classification

**Three-point justification — this is not a shortcut:**
1. Molecular: Hoadley et al. (2018) confirmed READ and COAD are one transcriptomic entity
2. Clinical: Same staging, same drugs, same treatment path — oncologists treat them as one disease
3. Empirical: Both Mostavi and Ramirez showed the same confusion — no learnable boundary exists in the data

**The principle:** A class label should correspond to a biologically meaningful, learnable distinction. READ and COAD do not have one at the transcriptomic level. We are correcting an annotation artifact, not inflating performance.

**Talking Points:**
- Our decision: merge all READ samples into the COAD class. One colorectal cancer class instead of two.
- The result is immediate and dramatic. CRC goes from F1 = 0.40 to 0.9924. The single worst class in prior work becomes one of our best.
- A professor might ask — is this just making the problem easier? And the answer is no. We are making the labels more accurate. Three independent lines of evidence support this merge: the molecular clustering analysis, the clinical treatment standards, and the empirical failure of two independent prior methods. All three point to the same conclusion: READ and COAD are one disease in the data.

---

---

## SECTION 4: MODEL & TRAINING (Slides 10–15)

---

### Slide 10: Data Preprocessing Pipeline

**Headline:** Before the model sees anything, data goes through a 5-step pipeline — each step has a specific reason.

**Content:**

**Step 1 — READ to COAD Merge**
- Remap all READ labels to COAD during metadata loading
- 33 classes become 32 before any other processing

**Step 2 — Log2 Normalization**
- Transform: log2(expression + 1) for every gene in every sample
- Why: RNA-Seq values are right-skewed across several orders of magnitude. Raw values of 50,000 and 2 become 15.6 and 1.6 — a manageable, near-normal distribution that models train on effectively

**Step 3 — Standard Scaling**
- Subtract mean, divide by standard deviation per gene (fit on training set only)
- Why: Ensures all 20,530 genes start with equal weight. Without this, high-magnitude genes dominate gradient updates. Scaler applied to val/test using training statistics only — no data leakage

**Step 4 — Stratified 70/15/15 Split**
- Preserves cancer type proportions in each split
- Why: Without stratification, rare cancers may have zero or very few samples in train. Test set untouched until final evaluation

**Step 5 — SMOTE on Training Set**
- Upsample any training class below 400 samples to exactly 400 (k=3 nearest neighbors)
- Why: Forces the model to take rare cancers seriously without just copying samples

**Talking Points:**
- Before any model training, the raw data goes through five preprocessing steps.
- The merge we already covered. Log2 normalization brings the skewed RNA-Seq values into a range that neural networks can effectively learn from. Standard scaling equalizes the contribution of all genes.
- The stratified split is important — we need every cancer type proportionally represented in train, validation, and test. And SMOTE on the training set only ensures class imbalance does not kill performance on rare cancers.

---

### Slide 11: Model Architecture — 1D-CNN with SE Attention

**Headline:** Four convolutional blocks with Squeeze-Excitation attention — reads gene expression as a 1D sequence and learns which gene groups matter.

**Content:**

**Architecture flow:**
- Input: 20,530 gene expression values (1 channel, 1D sequence)
- Conv Block 1: 64 filters, kernel=7, BatchNorm, ReLU, MaxPool(4) → SE block
- Conv Block 2: 128 filters, kernel=5, BatchNorm, ReLU, MaxPool(4) → SE block
- Conv Block 3: 256 filters, kernel=3, BatchNorm, ReLU, MaxPool(2) → SE block
- Conv Block 4: 512 filters, kernel=3, BatchNorm, ReLU → SE block
- Global Pooling: AdaptiveAvgPool + AdaptiveMaxPool concatenated → 1,024-dim vector
- FC Layer 1: Linear(1024→512) + ReLU + Dropout(50%)
- FC Layer 2: Linear(512→256) + ReLU + Dropout(30%)
- Output: Linear(256→32) — one score per cancer type

**Why 1D-CNN:**
- Gene expression is a 1D feature vector — natural architecture choice
- Local convolutional filters capture co-expressed neighboring genes (syntenic co-regulation)
- Weight sharing across positions = parameter efficiency on 20,530-dim input
- Prior validated: Mostavi et al. showed CNNs > SVM/RF on this exact dataset

**Visual:** Left-to-right architecture diagram showing input → 4 conv+SE blocks → global pool → FC head → 32 outputs

**Talking Points:**
- The model is a 1D Convolutional Neural Network. We treat the 20,530-gene expression vector as a 1D sequence and run it through four convolutional blocks of increasing depth — 64, 128, 256, then 512 filters.
- Kernel sizes decrease from 7 to 3 as we go deeper. Early layers use larger kernels to capture broader local patterns; deeper layers use smaller kernels to refine them.
- After the four conv blocks, dual global pooling — both average and max — gives us a 1024-dimensional representation. Three fully connected layers reduce this to 32 cancer type scores.
- Why 1D-CNN specifically: genes near each other on chromosomes tend to co-regulate, so local filters have biological meaning. And CNNs share weights, which is critical for parameter efficiency on 20K-dimensional inputs.

---

### Slide 12: Squeeze-Excitation Blocks — Channel Attention

**Headline:** SE blocks let the model dynamically focus on the most relevant gene expression channels per prediction.

**Content:**

**What SE blocks do — 3 steps:**
1. Squeeze: Global average pool each feature channel down to a single number — creates a descriptor vector summarizing how active each channel was
2. Excite: Pass descriptor through two small FC layers + Sigmoid → learned weight (0 to 1) per channel
3. Scale: Multiply each channel's feature map by its learned weight — amplify useful channels, suppress noisy ones

**Effect:** The model learns channel-wise attention. Different inputs get different channel weightings based on what is relevant for that specific sample.

**Why we added them:**
- Different cancer types are defined by different gene groups
- SE blocks allow the model to dynamically shift attention per-sample rather than using static fixed weights
- For a thyroid cancer sample: weight channels related to thyroid genes higher. For a leukemia sample: weight immune pathway channels higher
- SE blocks add very few parameters (two small FC layers) but consistently improve CNN performance (Hu et al. 2018)
- Also provides interpretability: the learned weights reveal which gene channels the model relies on

**Visual:** Simple 3-box diagram: input feature maps → Squeeze (GlobalAvgPool) → Excite (FC+Sigmoid) → Scale (multiply) → reweighted feature maps

**Talking Points:**
- After each convolutional block, we added a Squeeze-Excitation attention block. This is one of the more impactful additions to our architecture.
- The idea: after convolution produces many feature channels, the SE block learns which channels are actually useful for each prediction. It compresses each channel to a summary value, learns a 0-to-1 weight for each channel, and then multiplies the feature maps by those weights. Useful channels get amplified, noisy ones get suppressed.
- For us, this is biologically meaningful — the model is learning to pay attention to the gene groups most relevant for each cancer type. It learns this from data, not from us hard-coding which genes matter.
- SE blocks are cheap — they add very few parameters — but the performance improvement is consistent.

---

### Slide 13: Training Strategy

**Headline:** Four training techniques working together — each addresses a specific risk of overfitting or poor generalization.

**Content:**

**MixUp Augmentation (alpha = 0.2)**
- Blends two training samples: x_mixed = λ·x_A + (1-λ)·x_B, y_mixed = λ·y_A + (1-λ)·y_B
- Lambda drawn from Beta(0.2, 0.2) — mostly close to 0 or 1, occasionally intermediate
- Creates soft label targets → model learns smooth probability outputs
- Prevents memorization of specific training samples → strong regularization
- Naturally improves probability calibration (model never sees hard 0/1 targets)

**Gaussian Noise (sigma = 0.05)**
- Random noise added to every training input
- Simulates technical variability inherent in RNA-Seq measurements
- Makes model robust to measurement noise at inference time

**Label Smoothing (epsilon = 0.1)**
- Target probability: 0.9 for correct class, 0.1 spread across remaining 31 classes
- Prevents model from pushing output logits to extreme values
- Works with MixUp soft targets for combined calibration benefit

**Optimizer: AdamW + Cosine Annealing Warm Restarts**
- AdamW: correct decoupled weight decay (vs. standard Adam)
- LR: starts 0.001, cosine decay to 0.00001, periodic warm restarts (T0=40, Tmult=2)
- Warm restarts help model escape local minima
- Early stopping: patience = 40 epochs — stops when val loss does not improve

**Batch size: 256 | Max epochs: 200**

**Talking Points:**
- We used four training techniques in combination.
- MixUp creates synthetic samples by blending pairs of real samples and their labels. It is a strong regularizer — the model never memorizes specific examples because every training step it sees slightly different blended inputs. It also produces soft targets, which directly improves calibration.
- Gaussian noise simulates the measurement variability that is always present in real RNA-Seq data. Training with noise makes the model robust when it encounters noisy real samples.
- Label smoothing prevents the model from becoming overconfident by using 0.9 instead of 1.0 as the target for the correct class.
- For optimization, AdamW with cosine annealing and warm restarts — the cyclical learning rate helps the model find better solutions, and early stopping prevents overfitting.

---

### Slide 14: Five-Model Ensemble

**Headline:** Five identical models, five different random seeds — averaging their predictions cancels out individual errors.

**Content:**

**Why train five models instead of one:**
- Any single training run is subject to random variance — weight initialization, batch ordering, augmentation all affect the final solution
- Different seeds produce models that have converged to different local minima
- These models make different errors on different samples
- When you average predictions, errors cancel: if 4/5 models say BRCA and 1 says LUAD, the average strongly points to BRCA

**Seeds used:** 42, 123, 456, 789, 999

**Individual model performance (Seed 42 as example):**
- Test accuracy: 92.42%
- Macro F1: 0.9096

**Final ensemble performance:**
- Test accuracy: 94.33% (+1.9%)
- Macro F1: 0.9250 (+0.015)

**The gain is purely from diversity — same architecture, same data, just different random seeds.**

**Optimized weights (found via Nelder-Mead on validation set):**
- Seed 42: 0.2347 | Seed 123: 0.0901 | Seed 456: 0.2400 | Seed 789: 0.3321 | Seed 999: 0.1031
- Not equal weight averaging — heavier weight on stronger models

**Talking Points:**
- Instead of training one model and hoping it is the best, we trained five with different random seeds.
- The seeds affect everything random: weight initialization, the order batches are seen, which samples get blended in MixUp. Five seeds give five genuinely different models that each make different errors.
- When you average five diverse predictions, errors cancel out. The numbers show this clearly: a single model hits 92.4% accuracy. The ensemble hits 94.33% — two percentage points better from diversity alone, no architectural changes.
- We also found optimal per-model weights using Nelder-Mead optimization on the validation set. Seed 789 received the highest weight (0.33) because it found the best local minimum.

---

### Slide 15: Post-Training Pipeline — TTA and Calibration

**Headline:** After training, a five-stage pipeline refines and calibrates predictions — 50 predictions per test sample total.

**Content:**

**Stage 1 — Test-Time Augmentation (TTA)**
- At inference, make 10 predictions per model per sample, each with different random noise (sigma = 0.02)
- Average the 10 probability vectors
- 5 models × 10 passes = 50 total predictions per test sample
- Why: Reduces prediction variance. Especially effective for borderline samples near classification boundaries

**Stage 2 — Weighted Ensemble Averaging**
- Combine 5 models using Nelder-Mead optimized weights (not equal weights)
- 8 random restarts for robustness — take best solution found

**Stage 3 — Meta-Learner Stacking**
- Logistic Regression trained on validation set predictions from all 5 models
- Learns which model is most reliable per cancer type
- 160-dimensional input (5 models × 32 output probabilities each)

**Stage 4 — Temperature Scaling (Calibration)**
- Divide logits by learned temperature T before softmax
- T = 1.006 optimized on validation set
- T slightly above 1.0 → model was very slightly overconfident, now corrected
- Calibrated: 90% predicted confidence should mean ~90% empirical accuracy

**Stage 5 — Final Blend**
- 60% weight on TTA predictions + 40% weight on temperature-calibrated predictions
- TTA weighted higher for diversity advantage

**Talking Points:**
- After training, we run a five-stage inference pipeline.
- Test-time augmentation: for each test sample we make 10 predictions with different noise and average them. Five models times 10 passes equals 50 predictions per sample. This stabilizes outputs significantly, especially for uncertain cases.
- Temperature scaling is a calibration step — it adjusts the model's confidence scores so that when it says 90% probability, it is actually right 90% of the time. Our T of 1.006, very close to 1, tells us our model was already nearly well-calibrated coming out of training — a good sign that MixUp and label smoothing did their job.

---

---

## SECTION 5: RESULTS (Slides 16–20)

---

### Slide 16: Results — Overall Performance

**Headline:** 94.33% accuracy, 0.9250 Macro F1, 0.9957 Mean ROC-AUC across 32 cancer types.

**Content:**

| Metric | Value | What It Means |
|--------|-------|---------------|
| Test Accuracy | 94.33% | Correct on 94.33% of held-out test samples |
| Macro F1-Score | 0.9250 | Average F1 equally across all 32 classes — no class is ignored |
| Mean ROC-AUC | 0.9957 | Near-perfect discrimination — true class probability ranked above false class 99.57% of the time |
| Training Time | ~67 min | Total GPU training time across all 5 model seeds |

**Additional context:**
- Random chance baseline on 32 classes: 3.1%
- Mostavi et al. single-model CNN: ~90–93%
- Our ensemble: 94.33%
- 7 cancer types classified with F1 = 1.0000 — zero errors
- Only 1 class below F1 = 0.70: ACC (adrenocortical carcinoma, F1 = 0.5455) — rarest class, ~45 total samples

**Why Macro F1 matters more than raw accuracy:**
- Accuracy can be inflated by performing well only on common cancers
- Macro F1 treats every class equally — common and rare alike
- Our 0.9250 Macro F1 means strong average performance across all 32 types

**Talking Points:**
- Our final model achieves 94.33% accuracy, 0.9250 Macro F1, and 0.9957 Mean ROC-AUC on the held-out test set.
- To put 94.33% in context: random chance on 32 classes is about 3%. Mostavi's single CNN was around 90 to 93%. We are at 94.33% with a biologically cleaner problem setup.
- The Macro F1 of 0.9250 is probably the most meaningful number — it averages F1 equally across all 32 cancer types, so it cannot be inflated by doing well only on common cancers.
- The ROC-AUC of 0.9957 tells us the model's confidence score for the true class was ranked above a random wrong class 99.57% of the time. Near-perfect discrimination.

---

### Slide 17: Per-Class F1 Score Breakdown

**Headline:** 20 of 32 classes above F1 = 0.95. One difficult outlier at F1 = 0.55 due to extreme data scarcity.

**Content:**

**[Insert f1_scores_per_class.png — sorted descending bar chart]**

**Performance breakdown:**

**Perfect (F1 = 1.0000) — 7 classes:**
THCA, PRAD, PCPG, LAML, LUAD, STAD, CHOL
- Reason: highly tissue-specific gene expression that is uniquely distinguishable from all other types

**High performance (F1 = 0.95–0.99) — 10 classes:**
CRC (0.9924), PAAD (0.9818), BRCA (0.9807), TGCT (0.9796), SKCM (0.9787), UCS (0.9747), HNSC (0.9455), UCEC (0.9449), LGG (0.9412), KIRC (0.9392)

**Moderate (F1 = 0.70–0.95) — 14 classes:**
Includes LUSC (0.9333), KIRP (0.9032), LIHC (0.8966), SARC (0.8889), BLCA (0.8788), ESCA (0.7692), UVM (0.7200)
- Reason: histologically similar to adjacent cancer types — overlapping expression profiles

**Difficult (F1 < 0.70) — 1 class:**
ACC (0.5455) — adrenocortical carcinoma
- Reason: only 45 total TCGA samples — too few real examples even with SMOTE; genuine expression overlap with kidney and endocrine cancers

**Talking Points:**
- This chart shows per-class F1 scores for all 32 cancer types, sorted from best to worst.
- Seven cancer types had zero misclassifications. These are cancers with very distinctive molecular signatures — thyroid, prostate, pheochromocytoma, leukemia, lung adenocarcinoma, stomach, and cholangiocarcinoma. The model learned these with no ambiguity.
- The moderate-performance cancers are the ones that genuinely overlap with neighboring types — lung squamous cell and head and neck squamous cell are both squamous carcinomas with similar expression patterns, for example.
- The one red bar is ACC. This is the rarest cancer in the dataset with only 45 samples total. SMOTE can only do so much with that little real signal. It is also genuinely similar to kidney and endocrine tumors in expression space.

---

### Slide 18: Confusion Matrix Analysis

**Headline:** Errors are not random — they cluster in biologically similar cancer pairs. The CRC block is completely clean.

**Content:**

**[Insert confusion_matrix_final_merged.png]**

**Key observations:**
- Strong diagonal throughout — most cancer types classified correctly with high consistency
- Errors are biologically interpretable, not random:
  - LUSC and HNSC confusion: both squamous cell carcinomas from adjacent tissues (lung vs. head/neck)
  - Kidney subtypes (KIRC, KIRP, KICH): all renal cancers, different cell populations
  - UVM confusions: uveal melanoma overlaps with skin melanoma (SKCM) in some pathways
- CRC block is completely clean — the READ/COAD merge worked perfectly, no within-colorectal confusion
- ACC errors: scatter toward adrenal-adjacent tissues, consistent with small sample size and genuine expression overlap
- The remaining errors reflect genuine biological ambiguity, not model failures — perfect classification would require additional data modalities (mutations, methylation)

**Talking Points:**
- The confusion matrix gives us a picture of where the model's errors land.
- The diagonal should be dark — those are correct predictions — and most of it is.
- The off-diagonal errors are not random. They cluster in biologically related groups. LUSC and HNSC confusion makes sense — both squamous cell cancers. Kidney subtypes confuse each other — they are all renal tissue.
- The CRC block is clean. After the merge, there is zero within-colorectal confusion. That problem is completely gone.
- The remaining errors are the ones where even expert pathologists might disagree, because the cancers genuinely share molecular features. Eliminating them would require multi-omics data.

---

### Slide 19: Comparison to Mostavi et al.

**Headline:** Every improvement over Mostavi is intentional — each one addresses a specific identified gap.

**Content:**

| Aspect | Mostavi et al. (2020) | Our Work |
|--------|----------------------|----------|
| Architecture | Single CNN | 5× 1D-CNN with SE attention blocks |
| Cancer type labels | 33 classes (READ separate) | 32 classes (READ merged — biologically justified) |
| Ensemble | No | 5-model weighted ensemble (Nelder-Mead) |
| Data augmentation | Not reported | MixUp + Gaussian noise + label smoothing + SMOTE |
| Test-time augmentation | No | Yes — 10 passes per model (50 per sample) |
| Calibration | No | Temperature scaling (T = 1.006) |
| READ/COAD result | READ F1 = 0.40 | CRC F1 = 0.9924 |
| Test accuracy | ~90–93% | 94.33% |
| Macro F1 | ~0.88–0.92 | 0.9250 |
| Mean ROC-AUC | Not reported | 0.9957 |

**The hierarchy of improvements:**
1. READ/COAD merge — biggest single impact, transforms worst class to near-best
2. Ensemble (5 models) — +~2% accuracy from diversity alone
3. SE attention — channel-wise focus improves feature quality
4. Augmentation pipeline — MixUp/noise/smoothing improve generalization and calibration
5. TTA + calibration — reduces variance and makes confidence scores trustworthy

**Talking Points:**
- Let me put our work directly next to Mostavi since that is the primary comparison.
- The most important difference is the READ/COAD resolution. Their F1 = 0.40 for READ. Our CRC = 0.9924. That single biologically principled decision made the biggest difference.
- Second: we trained five models instead of one. Two percentage points of accuracy just from having diverse seeds.
- Third: the augmentation pipeline — MixUp, noise, label smoothing, SMOTE — each addressing a specific risk.
- Fourth: TTA and calibration, which make the model's uncertainty estimates trustworthy.
- None of these are random additions. Each one addresses a specific identified weakness in the prior work.

---

### Slide 20: Biomarker Discovery — What the Model Learned

**Headline:** The model recovers known clinical cancer biomarkers without being told what to look for.

**Content:**

**Method: Gradient-based gene importance**
- For each test sample: compute gradient of predicted class score with respect to every gene's expression value
- High gradient = model is highly sensitive to that gene — it is relying on it for this prediction
- Average gradients across all test samples per cancer type → ranked gene importance list per cancer
- Top 50 genes per cancer type saved to gene_importance_per_cancer.csv

**Known biomarkers recovered:**

| Cancer | Top Genes Found | Clinical Significance |
|--------|----------------|----------------------|
| Breast (BRCA) | GATA3, ESR1, PGR, ERBB2 | Hormone receptor + HER2 status — standard clinical test for treatment selection |
| Prostate (PRAD) | KLK3 (PSA), AR | PSA is the standard prostate screening blood test; AR drives tumor growth |
| Lung Adeno (LUAD) | TTF1, NKX2-1 | Standard IHC marker used by clinical pathologists to identify lung adenocarcinoma |
| Colorectal (CRC) | KRAS, SMAD4, MLH1, APC | Core CRC drivers; KRAS status guides anti-EGFR therapy eligibility |
| Kidney (KIRC) | VHL, PBRM1 | VHL loss is the founding event in clear cell kidney cancer |
| Leukemia (LAML) | FLT3, NPM1 | Standard AML mutation markers; FLT3 guides targeted therapy selection |

**What this confirms:** The model learned genuine biological signal, not statistical noise or batch artifacts.

**Talking Points:**
- High accuracy alone does not tell us whether the model learned real biology. We want to know what it is actually using for its predictions.
- We computed gradient-based gene importance — for each cancer type, we ranked genes by how sensitive the model's prediction is to small changes in their expression. Genes the model relies on get high scores.
- When we cross-reference the top genes per cancer type with the clinical literature, we find known biomarkers. For breast cancer: GATA3, ESR1, PGR, ERBB2 — exactly the genes oncologists test to determine hormone receptor status and HER2 status for treatment decisions. For prostate: KLK3, which is the gene that produces PSA — the protein measured in the PSA blood test used in routine prostate screening.
- The model was not told about any of these genes. It found them purely from learning to classify cancer types. That is strong evidence it learned real biology, not noise.

---

---

## SECTION 6: DISCUSSION & CLOSE (Slides 21–24)

---

### Slide 21: Challenges We Faced

**Headline:** Three real challenges — and what we did to address each one.

**Content:**

**Challenge 1 — Class Imbalance (27:1 ratio)**
- Problem: Early experiments showed the model learning common cancers well and failing on rare ones
- Solution: SMOTE to 400-sample minimum + weighted cross-entropy loss during training
- Result: Rare class performance improved substantially; model no longer ignores small classes

**Challenge 2 — Model Overconfidence**
- Problem: Neural networks tend to assign very high probabilities (0.99+) even on uncertain predictions
- This makes model outputs unreliable as confidence estimates — dangerous in medical contexts
- Solution: MixUp and label smoothing during training reduced overconfidence organically; temperature scaling fixed residual overconfidence post-hoc
- Result: T = 1.006 — model was already nearly perfectly calibrated out of training

**Challenge 3 — Ensemble Weight Optimization**
- Problem: Equal-weight averaging across 5 models is suboptimal — some models are stronger than others
- Solution: Nelder-Mead numerical optimization over validation set cross-entropy to find model-specific weights; 8 random restarts for robustness
- Result: Seed 789 (weight 0.3321) was strongest; Seed 123 (0.0901) weakest — not equal

**Talking Points:**
- Every project runs into real problems and I want to be honest about ours.
- The class imbalance was visible in early experiments. The model was giving up on rare cancers because it barely saw any gradient signal from them. SMOTE plus a weighted loss function fixed this.
- Overconfidence is a known neural network problem. Our training techniques pushed the model toward reasonable confidence levels, and temperature scaling gave us a clean post-hoc fix. The fact that T was only 1.006 tells us our training pipeline did most of the work already.
- For ensemble weights, we could not just use equal averaging and call it done — we ran optimization to find the best combination, which required multiple restarts to handle non-convexity.

---

### Slide 22: Training Behavior — Learning Curves

**Headline:** Models converged cleanly — training and validation curves stay close, no overfitting signs.

**Content:**

**[Insert training_curves.png — accuracy and loss over epochs, averaged across 5 seeds]**

**Key observations from the curves:**
- Training accuracy and validation accuracy both climb consistently throughout training
- No divergence between training and validation — no classic overfitting pattern
- Cosine warm restarts are visible as small periodic bumps in the loss curve — LR resets and model briefly explores before settling
- Early stopping triggered before 200-epoch limit in most seeds — models converged before budget ran out
- Results consistent across 5 seeds — curves are similar in shape, confirming stable training procedure and seed-independent convergence

**Talking Points:**
- The training curves confirm the model learned well without overfitting.
- Training and validation accuracy both improve together. The validation curve stays close to the training curve rather than flattening early — which would be the sign of overfitting.
- You can see the warm restart bumps in the loss curve — the learning rate resets periodically, giving the model a push to explore before it settles again.
- Early stopping triggered before 200 epochs in most runs, meaning the model found its best generalization naturally before exhausting the training budget.

---

### Slide 23: Limitations and Future Work

**Headline:** Four honest limitations — each one points to a concrete next step.

**Content:**

**Current Limitations:**

**1. ACC Performance (F1 = 0.5455)**
- Adrenocortical carcinoma has only 45 total TCGA samples
- SMOTE cannot manufacture the biological signal that is simply not there in 45 real examples
- Fix: More real ACC patient data, or few-shot learning techniques designed for extreme data scarcity

**2. Single Omics Only**
- We used only gene expression. TCGA also has DNA methylation, copy number variation, and somatic mutation data
- Some cancer type pairs are better separated by mutation patterns or methylation than by expression (e.g., IDH1 mutation separates LGG subtypes)
- Fix: Multi-omics integration — combine expression, methylation, and mutation data in a multi-input model

**3. TCGA-Only Validation**
- Trained and tested entirely within TCGA — different hospitals and sequencing platforms introduce batch effects
- We do not know if the model generalizes to external datasets from different institutions
- Fix: External validation on independent cohorts from GEO or ICGC before any clinical consideration

**4. Attribution Method**
- Gradient-based gene importance can be noisy — saturated gradients can underrate actually important genes
- Fix: Integrated Gradients or SHAP values for more robust, reliable biomarker analysis

**Talking Points:**
- We want to be honest about where this work has limits.
- ACC is the clearest failure and we know exactly why — 45 samples is not enough. SMOTE helped but you cannot generate biological signal that does not exist in 45 real patients. More data or specialized few-shot learning would be the path forward.
- We used only gene expression. Multi-omics is the natural next step — adding DNA methylation and mutation data would help for cancer pairs that are hard to separate by expression alone.
- The whole pipeline was trained and tested within TCGA. For any real clinical consideration, external validation on data from different hospitals and sequencing platforms would be required.

---

### Slide 24: Conclusion

**Headline:** Principled biology + modern deep learning = 94.33% accuracy, 0.9957 ROC-AUC across 32 cancer types.

**Content:**

**What we set out to do:**
- Build a reliable, well-calibrated pan-cancer classifier from gene expression data
- Address the known failures in prior work
- Validate that the model learned real biology

**What we achieved:**
1. READ/COAD merge — transformed prior work's worst class (F1 = 0.40) into near-perfect (CRC F1 = 0.9924) through biologically justified label correction
2. 1D-CNN with SE attention — architecture that learns which gene channels matter per prediction
3. 5-model ensemble + MixUp + TTA + calibration — 94.33% accuracy, 0.9250 Macro F1, 0.9957 Mean ROC-AUC
4. Biomarker recovery — model found known clinical markers (PSA for prostate, GATA3/ESR1 for breast, TTF1 for lung) purely from classification training

**What is next:**
- Multi-omics integration: add DNA methylation and copy number data
- Transformer architecture for long-range gene interaction modeling
- External cohort validation for generalizability
- SHAP-based interpretability for more reliable biomarker rankings

**Closing statement:**
This work shows that careful attention to biological label quality — not just model complexity — can make a larger difference than architectural improvements alone. The READ/COAD merge, grounded in published molecular evidence, was more impactful than any single technical addition.

**Talking Points:**
- To close: we started with a clear problem — prior work left a known failure unresolved and used single-model approaches that left performance on the table.
- We addressed both. The READ/COAD merge grounded in biology fixed the worst class. The ensemble of five SE-attention 1D-CNNs trained with MixUp, augmented with TTA, and calibrated with temperature scaling got us to 94.33% accuracy and 0.9957 ROC-AUC.
- The model also validated itself biologically — recovering known clinical markers without supervision.
- The clearest lesson from this project: knowing your data and understanding the biology matters as much as knowing your deep learning toolbox. A well-motivated label correction outperformed the entire technical pipeline in terms of impact.
- Thank you. Happy to take questions.

---


## KEY NUMBERS TO KNOW BY HEART

- Test Accuracy: **94.33%**
- Macro F1: **0.9250**
- Mean ROC-AUC: **0.9957**
- Dataset: **10,459 samples, 20,530 genes, 32 cancer types**
- Ensemble: **5 models × 10 TTA passes = 50 predictions per sample**
- READ F1 before merge (Mostavi): **0.40**
- CRC F1 after merge (ours): **0.9924**
- Single model accuracy (Seed 42): **92.42%**
- Ensemble accuracy: **94.33%** (+1.9% from diversity alone)
- Temperature: **T = 1.006** (near-perfect calibration)
- Highest ensemble weight: **Seed 789 = 0.3321**
- Classes with F1 = 1.0: **7** (THCA, PRAD, PCPG, LAML, LUAD, STAD, CHOL)
- Hardest class: **ACC F1 = 0.5455** (only 45 total samples)
- Class imbalance ratio: **27:1** (BRCA 1,218 vs CHOL 45)
- Training time: **~67 minutes on GPU**

---
