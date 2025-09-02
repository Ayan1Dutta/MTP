# LDE Worked Examples (Image vs Text Modality)

Date: 2025-09-02
Scope: Demonstrate Linear Decomposition of Embeddings (LDE) for two modalities used to build primitives (Ideal Words):
- modality_IW = "image": primitives from TRAIN image embeddings.
- modality_IW = "text": primitives from text (pair prompt) embeddings.

We construct tiny synthetic examples to show each intermediate tensor and final logits. Numbers are low‑dimensional (R^3) for clarity.

---
## 1. Recap of LDE Formula
Given embeddings E with labels (a_i, o_i):
1. Context: C = mean(E)
2. Attribute means: μ_a = mean({e_i | a_i = a})
3. Object means: ν_o = mean({e_i | o_i = o})
4. Attribute IW: a_IW = μ_a − C; Object IW: o_IW = ν_o − C
5. Pair prototype: p(a,o) = μ_a + ν_o − C (equivalently a_IW + o_IW + C)
Logits: image_emb ⋅ p(a,o) (optionally scaled).

---
## 2. Synthetic Setup
Attributes: A1, A2  |  Objects: O1, O2
Training (seen) pairs (image supervision): (A1,O1), (A2,O1)  (so O2 is unseen in images).  We will classify four test images: two seen (A1,O1), one seen (A2,O1), one unseen (A1,O2).

We craft embeddings so that image modality produces slight prototype drift for O1 (shared across A1 and A2), while text modality gives cleaner ν_O2.

---
## 3. Image Modality Example
### 3.1 Train Image Embeddings (unit-normalized approximations)
| Pair | Image Embeddings (examples) |
|------|-----------------------------|
| (A1,O1) | e1 = (0.90, 0.32, 0.28), e2 = (0.88, 0.34, 0.30) |
| (A2,O1) | e3 = (0.32, 0.90, 0.28), e4 = (0.34, 0.88, 0.30) |

Mean per pair (denoising inside pair):
- m_A1O1 = (e1+e2)/2 = (0.89, 0.33, 0.29)
- m_A2O1 = (e3+e4)/2 = (0.33, 0.89, 0.29)

(If multiple images per pair, `compute_weights` would first normalize weights within each pair; here uniform.)

### 3.2 Build Primitive Pool E
E = {e1,e2,e3,e4}. Context C = mean(E) = (0.61, 0.61, 0.29)

Attribute means:
- μ_A1 = mean(e1,e2) = (0.89, 0.33, 0.29)
- μ_A2 = mean(e3,e4) = (0.33, 0.89, 0.29)

Object means (O1 only present):
- ν_O1 = mean(e1,e2,e3,e4) = C (since all training images share O1)
(So ν_O1 = (0.61,0.61,0.29))

No visual data for O2, so with modality="image" we CANNOT build ν_O2 from images (prototype for any pair with O2 would be undefined in pure image modality). In practice you only evaluate pairs whose components appear in train when using pure image primitives; unseen pairs must still have their components individually seen. (In real datasets, each object usually appears with at least one attribute in train.) For illustration we continue with O2 absent → we will NOT generate (A1,O2) under strict image modality.

However, to show unseen composition we slightly adjust scenario: assume ONE training image for (A1,O2): e5 = (0.85, 0.10, 0.50) (so O2 present but pair (A1,O2) is withheld at test for evaluation—treat e5 as belonging to a different attribute A0 for structural presence; simplified here). For clarity we instead switch to text modality for unseen below.

### 3.3 Pair Prototypes (Seen)
- p(A1,O1) = μ_A1 + ν_O1 − C = (0.89,0.33,0.29)+(0.61,0.61,0.29)−(0.61,0.61,0.29) = (0.89,0.33,0.29)
- p(A2,O1) = μ_A2 + ν_O1 − C = (0.33,0.89,0.29)
(Here p equals attribute means because ν_O1 = C; minimal drift case.)

### 3.4 Test Image Embeddings
| True Pair | Image | Vector |
|-----------|-------|--------|
| (A1,O1) | t1 | (0.91,0.31,0.28) |
| (A1,O1) | t2 | (0.88,0.35,0.30) |
| (A2,O1) | t3 | (0.31,0.91,0.28) |

Cosine (≈ dot since near unit):
- t1·p(A1,O1)=0.91*0.89+0.31*0.33+0.28*0.29 ≈ 0.81+0.10+0.08=0.99
- t1·p(A2,O1)=0.91*0.33+0.31*0.89+0.28*0.29 ≈ 0.30+0.28+0.08=0.66 → correct classification.

(Drift scenario omitted here because ν_O1 equaled C; see numeric sketch in main notes for drift illustration.)

---
## 4. Text Modality Example (Including Unseen Pair)
We now use text embeddings as primitive pool, enabling construction for all cross product pairs (open-world or closed-world). Suppose CLIP text encoder yields:

| Component Prompt | Embedding |
|------------------|-----------|
| A1 | a1 = (0.92,0.28,0.28) |
| A2 | a2 = (0.28,0.92,0.28) |
| O1 | o1 = (0.60,0.60,0.30) |
| O2 | o2 = (0.84,0.12,0.52) |

We simulate pair-prompts were averaged into these component estimates (simplified). Primitive set E_component = {a1,a2,o1,o2}. Context C_text = mean(E_component) = ((0.92+0.28+0.60+0.84)/4, ... ) = (0.66, 0.48, 0.345) → approx: (0.66,0.48,0.35).

Compute component means (since each appears once, mean = vector itself):
- μ_A1=a1, μ_A2=a2, ν_O1=o1, ν_O2=o2.

Pair prototypes:
- p(A1,O1)= μ_A1 + ν_O1 − C_text = (0.92,0.28,0.28)+(0.60,0.60,0.30)−(0.66,0.48,0.35) = (0.86,0.40,0.23)
- p(A2,O1)= a2 + o1 − C_text = (0.22,1.04,0.23)
- p(A1,O2)= a1 + o2 − C_text = (1.10, -0.08, 0.45)
- p(A2,O2)= a2 + o2 − C_text = (0.46, 0.56, 0.45)
(Each then L2-normalized in practice.)

Test image embedding for unseen pair (A1,O2): t_un = (0.88,-0.05,0.46)
Cosines (pre-normalization approximation):
- t_un·p(A1,O2) large (alignment in X and Z, Y near 0)
- t_un·p(A1,O1) smaller (Y=0.40 positive vs t_un negative Y) → unseen pair can be ranked above seen pair for this image.

---
## 5. Comparison: Image vs Text Modality
| Aspect | Image Modality | Text Modality |
|--------|----------------|---------------|
| Source variance | Real image noise (lighting/background) | Linguistic prompt noise (lower) |
| Coverage of unseen pairs | Only if both components seen in train | Full Cartesian product (if prompts exist) |
| Drift risk | Component means pulled by multi-attribute sharing | Lower (components more semantically pure) |
| Seen<Unseen phenomenon | Arises via shared object averaging & centering | Can still occur; often stronger unseen due to semantic purity |

---
## 6. Minimal PyTorch Snippet (Both Modalities)
```python
import torch

# Synthetic text component embeddings (rows): A1, A2, O1, O2
text_components = torch.tensor([
    [0.92,0.28,0.28],  # A1
    [0.28,0.92,0.28],  # A2
    [0.60,0.60,0.30],  # O1
    [0.84,0.12,0.52],  # O2
])
# Normalize (as CLIP typically outputs already unit norm after optional F.normalize)
text_components = torch.nn.functional.normalize(text_components, dim=-1)
A1, A2, O1, O2 = text_components
C_text = text_components.mean(0)

# Compose function
def compose(mu_a, nu_o, C):
    p = mu_a + nu_o - C
    return torch.nn.functional.normalize(p, dim=-1)

p_A1O1_text = compose(A1, O1, C_text)
p_A1O2_text = compose(A1, O2, C_text)

# Example unseen image embedding (approximate unit)
t_un = torch.tensor([0.88,-0.05,0.46])
t_un = torch.nn.functional.normalize(t_un, dim=-1)

score_seen = torch.dot(t_un, p_A1O1_text).item()
score_unseen = torch.dot(t_un, p_A1O2_text).item()
print("score(A1,O1)=", score_seen)
print("score(A1,O2)=", score_unseen)  # Expect unseen higher

# Image modality example with only O1 present
image_train = torch.tensor([
    [0.90,0.32,0.28], [0.88,0.34,0.30],  # (A1,O1)
    [0.32,0.90,0.28], [0.34,0.88,0.30],  # (A2,O1)
])
labels = [("A1","O1"),("A1","O1"),("A2","O1"),("A2","O1")]
C_img = image_train.mean(0)

# Compute means per attribute/object
import collections
attr_groups = collections.defaultdict(list)
obj_groups  = collections.defaultdict(list)
for emb,(a,o) in zip(image_train, labels):
    attr_groups[a].append(emb)
    obj_groups[o].append(emb)
mu_A1 = torch.stack(attr_groups['A1']).mean(0)
mu_A2 = torch.stack(attr_groups['A2']).mean(0)
nu_O1 = torch.stack(obj_groups['O1']).mean(0)
# Compose (A1,O1)
p_A1O1_img = compose(mu_A1, nu_O1, C_img)
# Score a seen test image
t_seen = torch.tensor([0.91,0.31,0.28])
t_seen = torch.nn.functional.normalize(t_seen, dim=-1)
score_seen_img = torch.dot(t_seen, p_A1O1_img).item()
print("image modality score(A1,O1)=", score_seen_img)
```

---
## 7. Key Observations from Examples
1. Text modality prototypes leverage semantically separated directions → can yield higher scores for genuinely novel compositions.
2. Image modality prototypes may collapse object distinctions when object appears with diverse attributes (drift not shown in the minimal equal-weight case; arises with more heterogeneous samples).
3. Unseen > seen emerges when (a) seen prototype drifts toward shared average; (b) unseen prototype uses a more discriminative semantic object vector.

---
## 8. How to Reproduce with Repo Code
1. Precompute embeddings: `python datasets/compute_embeddings.py <dataset> ViT-L-14 openai`
2. Run classification with text primitives: `python classification.py --dataset <dataset> --experiment_name LDE --modality_IW text`
3. Run with image primitives: `python classification.py --dataset <dataset> --experiment_name LDE --modality_IW image`
4. Compare seen vs unseen accuracy + harmonic mean.

---
## 9. Extension Ideas
- Mixed modality primitives: use text for scarce components, image for frequent ones.
- Per-component reliability weighting: blend μ_a(text) and μ_a(image) based on variance.

(End of LDE modality examples)
