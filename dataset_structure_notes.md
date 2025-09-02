# Dataset Organization & Split Structure Notes

Date: 2025-09-02

These notes document how compositional datasets in this repo are laid out on disk and how the code (`datasets/composition_dataset.py`, `datasets/compute_embeddings.py`) loads & uses them. Includes: directory trees, file roles, split semantics, and data flow diagrams.

---
## 1. Top-Level Repo (Relevant Parts)
```
./data/
  celebA/
    images/                  # All CelebA images (aligned subset)
    compositional-split-natural/
      train_pairs.txt
      val_pairs.txt
      test_pairs.txt
    metadata_compositional-split-natural.t7
    TEXTemb_ViT-L-14_openai.pt
    IMGemb_ViT-L-14_openai.pt
    ... (possibly other model variants)
  mit-states/
    images/
    compositional-split-natural/{train_pairs.txt,val_pairs.txt,test_pairs.txt}
    metadata_compositional-split-natural.t7
    TEXTemb_ViT-L-14_openai.pt
    IMGemb_ViT-L-14_openai.pt
  ut-zappos/
    images/                  # Raw shoe images
    _images/ (sometimes intermediate)
    compositional-split-natural/{...}
    metadata_compositional-split-natural.t7
    TEXTemb_ViT-L-14_openai.pt
    IMGemb_ViT-L-14_openai.pt
  waterbird_complete95_forest2water2/  # Raw source directory (original)
  waterbirds/
    images/                  # Standardized images for composition task
    compositional-split-natural/{train_pairs.txt,val_pairs.txt,test_pairs.txt}
    metadata_compositional-split-natural.t7
    TEXTemb_ViT-L-14_openai.pt
    IMGemb_ViT-L-14_openai.pt
```

### Core File Roles
| File | Purpose |
|------|---------|
| `images/` | Raw RGB image files used for embedding extraction. |
| `compositional-split-natural/*.txt` | Plain-text lists of (attribute, object) pairs partitioned by role (train/val/test). |
| `metadata_compositional-split-natural.t7` | Torch-saved list of dicts: each entry has keys `{image, attr, obj, set}` tying an image to its labeled attribute, object, and split. |
| `TEXTemb_*.pt` | Precomputed CLIP text embeddings for prompts (pairs and/or class-only) with metadata. |
| `IMGemb_*.pt` | Precomputed CLIP image embeddings for each image referenced in metadata. |

---
## 2. Split Semantics
- `train_pairs.txt`: Attribute–object combinations with visual supervision (seen pairs).
- `val_pairs.txt`: Validation pairs (some may be seen if they appear also in train, but typically disjoint for zero-shot composition evaluation).
- `test_pairs.txt`: Test pairs (mixture of seen and unseen vs train).
- Global sets built by code:
  - `train_pairs`, `val_pairs`, `test_pairs`: loaded explicitly.
  - `full_pairs = product(attrs, objs)`: all possible synthetic combinations (may exceed union of listed pairs if some combos never appear).

Each *.txt file format (typical): one pair per line, whitespace or comma separated (the loader composes list of tuples). Attributes/objects aggregated across the three files to build global vocabularies.

---
## 3. Metadata File Structure
`metadata_compositional-split-natural.t7` (Torch serialized) → Python object like:
```
[
  {
    'image': 'XXXX.jpg',
    'attr': 'red',
    'obj': 'car',
    'set': 'train'
  },
  {
    'image': 'YYYY.jpg',
    'attr': 'blue',
    'obj': 'car',
    'set': 'test'
  },
  ...
]
```
Notes:
- 'attr' may be 'NA' for items excluded (code drops these).
- 'set' ∈ {'train','val','test','NA'}.
- Every usable record supplies the triplet (image path relative under images/, attr, obj).

---
## 4. Loading Flow (Code Path)
### In `CompositionDataset.__init__`:
1. `parse_split()` reads the three pair list files → returns:
   - `attrs`, `objs`, `pairs` (union of all), plus per-split pair lists.
2. Constructs `full_pairs = product(attrs, objs)`.
3. If `open_world=True`, sets `self.pairs = full_pairs` (expands candidate label space).
4. Calls `get_split_info()` to iterate metadata:
   - Builds `train_data`, `val_data`, `test_data` as lists of `[image, attr, obj]` only when `(attr,obj)` ∈ allowed `self.pairs` and `attr!='NA'`.
5. According to `phase` argument, selects subset for `self.data`.
6. Builds helpful maps: `attr2idx`, `obj2idx`, `pair2idx`, boolean `seen_mask` for candidate pairs.

### In `CompositionDatasetEmbeddings` (subclass):
- Adds paths to precomputed embedding files via naming pattern:
  - `TEXTemb_{arch}_{pretraining}.pt`
  - `IMGemb_{arch}_{pretraining}.pt`
- Methods:
  - `load_text_embs(pairs)` → returns embeddings for those pairs (can replicate entries if multiple prompt templates per pair).
  - `load_all_image_embs()` → loads image embeddings aligned in order to current phase's `self.data`.

---
## 5. Embedding File Internals
### Text (`TEXTemb_...pt`)
Torch-saved dict keys (typical):
```
{
  'embeddings': Tensor [N_text, D],
  'pairs': List[(attr,obj)] (attr may be None for class-only prompts),
  'prompt template': str or template description
}
```
### Image (`IMGemb_...pt`)
```
{
  'image_ids': List[str],          # Matches 'image' field in metadata
  'embeddings': Tensor [N_img, D],
  'pairs': zip(all_attrs, all_objs) or iterator (consumed on load)
}
```

---
## 6. Data Flow Diagrams
### 6.1 Split & Vocabulary Construction
```
train_pairs.txt  ┐
val_pairs.txt    ├─> parse_split() →
test_pairs.txt  ┘        attrs set
                         objs set
                         all_pairs (union)
                         full_pairs = attrs × objs
```

### 6.2 Metadata Filtering
```
metadata_compositional-split-natural.t7
         │
         ▼ (iterate)
   discard if attr=='NA' or (attr,obj) not in self.pairs
         │
         ├─ set=='train' → train_data (image,attr,obj)
         ├─ set=='val'   → val_data
         └─ set=='test'  → test_data
```

### 6.3 Embedding Assembly (Image Modality)
```
train_data images ─┐
val/test images    │ (phase selection in CompositionDatasetEmbeddings)
                   ▼
IMGemb_*.pt (load all) → index by image_id → phase image_embs aligned with all_pairs_true
```

### 6.4 Text Embedding Retrieval
```
TEXTemb_*.pt (embeddings, pairs)
         │
request pairs list (target label space)
         ▼
collect indices (may duplicate if multiple prompts per pair)
         ▼
(text_embs subset) → optional L2 normalize
```

### 6.5 LDE Primitive Construction (Image Modality)
```
(train image embeddings, train (attr,obj) labels)
          │
   optional within-pair weights
          │
   compute context C, attr means μ_a, obj means ν_o
          │
   pair prototype p(a,o)=μ_a+ν_o−C (for any candidate pair)
```

---
## 7. Key Index Mappings
| Mapping | Built From | Use |
|---------|------------|-----|
| `attr2idx` | Sorted unique attributes | Convert attr string → index for tensor ops |
| `obj2idx`  | Sorted unique objects    | Same for objects |
| `pair2idx` | Enumerated `self.pairs` (depends on open_world) | Label space alignment |
| `seen_mask`| Membership in `train_pairs` over `self.pairs` | Separate seen vs unseen predictions |

---
## 8. Open World vs Closed World
| Mode | Candidate `self.pairs` | Effect |
|------|------------------------|--------|
| Closed (default) | Union of split pairs | Only real (attr,obj) that appear in any split |
| Open World | full_pairs (A×O) | Adds synthetic combinations for zero-shot evaluation |

---
## 9. Dataset-Specific Attribute/Object Examples
| Dataset | Attribute Examples | Object Examples | Notes |
|---------|--------------------|-----------------|-------|
| MIT-States | adjectives/states (e.g., "wooden","broken") | base nouns (e.g., "chair","car") | Classic compositionality benchmark |
| UT-Zappos | properties ("red","leather","high") | shoe categories ("boot","sandal") | Some attributes sparse |
| CelebA | hair color, gender (if used), facial attributes | (Simplified) hair color object or treat gender as attribute | Debiasing prompts combine spurious (gender) + class (hair) |
| Waterbirds | background type (spurious: land/water) | bird class group (landbird/waterbird) | Debiasing pairs combine spurious + class |

(Note: For debiasing variants, one component can act as spurious attribute.)

---
## 10. Typical Pair Lifecycle
```
(pair in train_pairs) → appears in metadata with images → contributes to μ_a, ν_o
(pair only in test_pairs) → no images in train → prototype still composable via component means
(pair never in any split) → only exists if open_world and synthetic; prototype composable but no direct evaluation sample
```

---
## 11. Practical Checklist When Adding a New Dataset
1. Place images under `data/<newdataset>/images/`.
2. Create `compositional-split-natural/{train_pairs.txt,val_pairs.txt,test_pairs.txt}`.
3. Build `metadata_compositional-split-natural.t7` list of dicts (image relative path, attr, obj, set).
4. Run `compute_embeddings.py <dataset> ViT-L-14 openai` to generate TEXT/IMG embedding files.
5. Verify `parse_split()` returns expected counts (print dataset object).
6. Run classification script with `--experiment_name LDE --modality_IW text` for baseline.
7. Optionally enable open world with `--open_world`.

---
## 12. Common Pitfalls
| Issue | Cause | Fix |
|-------|-------|-----|
| Missing pair prototype | Attribute or object absent in train embeddings (image modality) | Ensure every attr/object appears in at least one train pair; otherwise fall back to text modality for missing components |
| Duplicate prompt entries | Multiple templates per pair in text embeddings | Average or deduplicate when constructing label embeddings |
| Unbalanced seen vs unseen counts | Split design skewed | Report harmonic mean and consider bias calibration |
| High hubness | Over-averaged object means | Consider weighting or residual refinement |

---
## 13. Minimal Pseudocode (Loading + Prototypes)
```python
# Load dataset embeddings (test phase)
dset_test = CompositionDatasetEmbeddings(root, phase='test')
# Train embeddings for primitives
train_embs, train_pairs = CompositionDatasetEmbeddings(root, phase='train').load_all_image_embs()
# Build LDE primitives
context, attr_IW, obj_IW = compute_attr_obj_means(train_embs, train_pairs)
# Compose prototype for arbitrary (a,o)
proto = attr_IW[attr2idx[a]] + obj_IW[obj2idx[o]] + context
```

---
## 14. ASCII End-to-End Overview
```
          +------------------------------+
          |  split .txt files (pairs)    |
          +---------------+--------------+
                          | parse_split()
                          v
                 attrs, objs, train/val/test pairs
                          |
                          | build full_pairs
                          v
+----------------------+   +----------------------------------+
| metadata .t7 records |-->| filter valid (attr!='NA' & pair) |--+--> train_data
+----------------------+   +----------------------------------+  +--> val_data
                                                             |    +--> test_data
                                                             v
                                                   (image list per phase)
                                                             |
                                                             | load IMGemb_*.pt (index by image_ids)
                                                             v
                                                    train image embeddings
                                                             |
                                           compute context / attr means / obj means (LDE)
                                                             |
                                +----------------------------+----------------------+
                                |         Candidate Label Space (closed or open)     |
                                +----------------------------+----------------------+
                                                             |
                                      compose prototypes for each (attr,obj)
                                                             |
                                      score test image embeddings (dot products)
                                                             |
                                      evaluate seen vs unseen accuracies
```

---
## 15. Summary
- Splits are pair-level; images link to splits via metadata file.
- `full_pairs` enables synthetic open-world compositions.
- Embedding files decouple expensive CLIP forward passes from experimentation.
- LDE prototypes require only component coverage in train; unseen pairs become accessible via additive recomposition.

(End of dataset notes)
