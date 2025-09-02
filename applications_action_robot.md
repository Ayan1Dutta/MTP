# Application Concepts: Compositional Action Recognition & Multi-Tool Skill Transfer

Date: 2025-09-02
Scope: Two focused application ideas extending CLIP + LDE (no GDE) to (1) video action recognition and (2) robotic multi-tool skill transfer.

---
## 1. Compositional Action Recognition (Manner–Object)
### 1.1 Problem Framing
Recognize fine-grained actions in video by factoring each action instance into two reusable primitives:
- Attribute ("manner" / motion modifier): e.g., fast, slow, wiping, pouring, slicing, rotating.
- Object (manipulated target): cup, pan, knob, cloth, door.
Action label ≈ (manner, object). Many (manner, object) pairs are unseen in training but component primitives appear individually (manner with other objects, object with other manners).

### 1.2 Mapping to LDE
Treat short video clip embeddings (segment-level) as analogous to image embeddings.
- Collect training segments with labels (manner, object).
- Compute LDE primitives: context C_vid, μ_manner, ν_object.
- Compose prototype for any (manner, object) via p = μ_manner + ν_object − C_vid.

### 1.3 Pipeline
1. Video Segmentation: Split videos into uniform or activity-detected segments (e.g., 16–32 frames).
2. Embedding Extraction: Frozen video-text model (e.g., CLIP-TimeSformer, ViT + temporal pooling).
3. Primitive Construction: LDE over training segment embeddings.
4. Prototype Generation: For all candidate (manner, object) combos (closed or open world).
5. Scoring: segment_emb · p(manner, object) (scaled dot product).
6. Temporal Smoothing: Optional HMM/CRF or simple majority over sliding window.

### 1.4 Zero-Shot Setting
Hold out a subset of (manner, object) pairs so that each manner and each object still appears elsewhere in training.

### 1.5 Example
Seen: (stirring, spoon), (pouring, bottle), (wiping, cloth)
Unseen target: (pouring, bowl). Both pouring and bowl appear with other partners.

### 1.6 Evaluation
Metrics:
- Seen Acc, Unseen Acc, HM (harmonic mean).
- mAP over all pairs.
- Per-manner and per-object recall.
- Early recognition (accuracy vs observed frames) to show compositional anticipation.

### 1.7 Baselines
- Direct pair text prompt embeddings (“a video of [manner] [object]”).
- Concatenated or averaged individual CLIP text tokens (no centering).
- Fine-tuned classifier head on full pair IDs (no compositional generalization).
- Prototype from pair mean only (no factorization).

### 1.8 Analyses
- Drift: angle between pair prototype and empirical seen pair mean.
- Component variance vs per-pair error.
- Hubness of object prototypes (appear frequency in top-k).
- Effect of removing centering (−C) on seen/unseen gap.

### 1.9 Expected Contributions / Novelty Hooks
- Demonstrate systematic zero-shot generalization across action-object space without finetuning backbone.
- Show LDE reduces prompt variance vs direct pair textual prompts.
- Provide new benchmark split for manner–object compositional action recognition.

### 1.10 Risks & Mitigations
| Risk | Mitigation |
|------|------------|
| Weak manner annotation quality | Use motion clustering to refine labels; prune ambiguous segments. |
| High intra-object visual diversity overshadowing manner | Normalize per-object variance; variance-aware scaling of μ_manner. |
| Temporal confounds (background) | Context subtraction; per-scene context clustering. |

### 1.11 Extension Options
- Third factor: phase (start/mid/end) creating triple-factor prototypes.
- Residual interaction term for complex synergy (μ_manner + ν_object + R_{m,o}).

### 1.12 Minimal Pseudocode Sketch
```python
# segment_embs: [N, D], labels: list[(manner, object)]
C = segment_embs.mean(0)
μ_manner = group_mean(segment_embs, manner_labels)
ν_object = group_mean(segment_embs, object_labels)
# classify new segment x
score(m,o) = (x @ (μ_manner[m] + ν_object[o] - C))
```

---
## 2. Multi-Tool Skill Transfer (Skill–Tool)
### 2.1 Problem Framing
Robotic manipulation from limited demonstrations where each demonstration pairs a reusable skill verb with a specific tool. Goal: execute an unseen (skill, tool*) combination where the tool shares affordances with known tools for that skill, or the skill has been applied to related tools.

- Attribute (skill): scoop, scrape, hammer, pry, stir, cut.
- Object (tool): spoon, spatula, screwdriver, ladle, knife.
Unseen example: (scrape, spoon) or (stir, spatula) given training coverage of skill and tool separately.

### 2.2 Mapping to LDE
Encode each demonstration (RGB frames, depth, or fused sensory observation) into an embedding e_demo using a frozen visuomotor representation (e.g., R3M, VC-1, CLIP video). Labels: (skill, tool).
Compute LDE primitives (μ_skill, ν_tool, context C_demo) over demonstration embeddings.
Compose prototype p(skill, tool) for planning or retrieval.

### 2.3 Usage Modes
1. Retrieval: For target (skill*, tool*), find nearest stored demonstration embedding to p(skill*, tool*) and adapt trajectory (retime / regrasp).
2. Policy Conditioning: Use p(skill*, tool*) as latent conditioning vector for a diffusion or transformer policy (frozen or lightly fine-tuned adapter).
3. Sequencing: For multi-step tasks, sum successive pair prototypes to build a task latent.

### 2.4 Data Requirements
- Each skill appears with ≥2 tools.
- Each tool appears with ≥2 skills.
- Balanced matrix coverage not required but improves stability.
- Add small validation set for hyperparameters (e.g., weighting, residual sharpening).

### 2.5 Prototype Composition
p(skill, tool) = μ_skill + ν_tool − C_demo.
Optional refinements:
- Affordance injecting: Embed tool mesh / point cloud; derive ν_tool = visual_tool + affordance_latent − C_combined.
- Reliability blending: p = α_skill μ_skill_img + (1−α_skill) μ_skill_text  + α_tool ν_tool_img + (1−α_tool) ν_tool_text − C.

### 2.6 Adaptation Strategy
Given retrieved demonstration d_retr (closest to p):
1. Align initial gripper pose via pose regression network.
2. Retime via dynamic time warping on intermediate visual embeddings.
3. Low-level controller tracks adapted keyframes (Cartesian impedance or velocity control).

### 2.7 Evaluation
Metrics:
- Zero-shot success rate on unseen (skill, tool) pairs.
- Generalization gap: Seen vs Unseen success and HM.
- Demonstration efficiency: Performance vs number of demos.
- Latent retrieval quality: cosine(p, nearest demo) vs success probability (calibration).
- Affordance transfer: success when substituting tool with similar affordances but different geometry.

### 2.8 Baselines
- Nearest neighbor without factorization (raw embedding to demo library).
- Direct textual prompt embedding of "skill tool" (no decomposition).
- Fine-tuned policy per pair (upper bound, data-heavy).
- Meta-imitation (context-conditioned transformer) without explicit composition.

### 2.9 Analyses
- Drift: angle between p(skill, tool) and empirical mean of demonstrations for seen pairs.
- Variance decomposition: % variance explained by tool vs skill vs residual.
- Ablation: remove centering; replace ν_tool with mesh encoder; add residual interaction term.
- Failure categorization: skill mismatch vs tool mismatch (gradient attribution to μ_skill or ν_tool).

### 2.10 Expected Contributions / Novelty Hooks
- Show compositional factorization drastically reduces needed demonstrations for unseen pair execution.
- Provide quantitative link between prototype–demo cosine and success likelihood (enabling active demo collection).
- Introduce reliability blending of text and sensory primitives for robotics.

### 2.11 Risks & Mitigations
| Risk | Impact | Mitigation |
|------|--------|-----------|
| Embedding not factorized (entangled skill & tool) | Poor zero-shot | Add contrastive regularizer pushing μ_skill sets apart while collapsing tool variance within a skill. |
| Tool geometry dominates (visual bias) | Low skill transfer | Affordance subspace extraction; subtract geometry principal components. |
| Covariate shift (lighting, background) | Prototype drift | Context clustering (multiple C’s) or adaptive centering per scene. |
| Retrieval latency with large demo library | Real-time failure | ANN index (FAISS) over composed prototypes or precomputed sums (μ_skill + ν_tool). |

### 2.12 Minimal Pseudocode Sketch
```python
# demo_embs: [N, D]; labels: list[(skill, tool)]
C = demo_embs.mean(0)
μ_skill = group_mean(demo_embs, skill_labels)
ν_tool  = group_mean(demo_embs, tool_labels)

def prototype(skill, tool):
    return μ_skill[skill] + ν_tool[tool] - C

# retrieval
query_p = prototype(target_skill, target_tool)
idx = torch.argmax(demo_embs @ query_p)
selected_demo = demos[idx]
```

### 2.13 Extension Options
- Add uncertainty: maintain per-component covariance; risk-aware selection.
- Online update: incremental mean updates when new demo added (O(1) per addition).
- Interaction residual: learn low-rank R_{s,t} to capture non-additive synergies.

### 2.14 Benchmark Proposal Sketch
Matrix Split Construction:
- Skills S, Tools T.
- Sample train pairs ensuring each s∈S and t∈T appears at least once.
- Reserve 20–30% pairs for zero-shot test.
Report: success_seen, success_unseen, HM, success_vs_demo_count curve.

### 2.15 Potential Publication Angles
- Robotics & Learning (CoRL / ICRA / RSS): emphasis on demo efficiency.
- Vision & Language Workshops: compositionality generalization with frozen encoders.
- Embodied AI Benchmarks: new dataset / split design + evaluation protocol.

---
## 3. Joint Considerations
| Theme | Action Recognition | Multi-Tool Skill Transfer |
|-------|--------------------|----------------------------|
| Primitive Noise | Motion intra-class variance | Demonstration style variance |
| Additional Signals | Optical flow, pose | Force/torque, proprioception |
| Center Strategy | Single C or per-video cluster | Single C or per-scene cluster |
| Zero-Shot Risk | Manner ambiguity | Tool affordance mismatch |

Shared Methods:
- Reliability weighting between textual prompts (semantic prior) and empirical visual embeddings.
- Residual sharpening for frequent seen pairs to reduce seen<unseen inversion.

---
## 4. Immediate Next Steps
1. Choose datasets: (EPIC-Kitchens subset) + small teleop tool dataset (e.g., spoon, spatula, screwdriver).
2. Implement generic LDE primitive builder for sequence/video embeddings (reuse image LDE logic).
3. Construct zero-shot splits ensuring component coverage.
4. Baseline experiments (pair prompt vs LDE) to confirm compositional gain.
5. Begin logging: drift metrics, seen/unseen accuracy or success.

---
## 5. Key Hypotheses (Testable)
H1: LDE prototypes reduce prompt variance and improve HM over direct pair prompts by ≥5 absolute points in zero-shot action recognition.
H2: Compositional prototypes enable ≥40% zero-shot success on unseen (skill, tool) pairs with ≤50% of full Cartesian demonstrations.

(End of applications document)
