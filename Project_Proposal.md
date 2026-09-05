# Interaction Aware Trajectory Prediction Under Real-Time Constraints
## Team and Responsibilities
Jacob Webster - Developer/Researcher
## Feedback Received and Responses

Dr. Ruchkin - "Simplify the project." --- Not sure how to proceed with this, I plan on following up with him to determine
what a reasonable scope would be

## Problem and Motivation

Current trajectory prediction systems treat agents independently. A pedestrian moves because a car drives by; a vehicle merges because it sees another vehicle. Independent models fail in these interactive scenarios.

However, modeling interactions adds latency. Autonomous vehicles operate under strict time constraints (~100ms per control cycle). Predictions that improve accuracy but exceed latency budgets are useless. The core challenge: Can we model multi-agent interactions accurately and meet real-time constraints?

## Research Questions and Hypotheses

- RQ1: How much more accurate is GNN compared to independent LSTM? 
- RQ2: What's the latency overhead? 
- RQ3: Which interactions actually matter for prediction?

- H1: GNN will be more accurate than independent LSTM in the majority of scenes
- H2: Latency will grow exponentially as more agents are being processed in each scene
- H3: All interactions will assist in more accurate predictions; some will be more influential than others

## Related Work

https://arxiv.org/abs/2001.03093 - Related work on trajectory forecasting provided by Dr. Ruchkin

## Proposed System or Approach

Baseline (Independent LSTM):

- Separate LSTM cell per agent processes historical trajectory (2s history).
- Outputs future positions via linear regression head.
- Inference time: ~30-50ms per agent.

Proposed Model (GNN):

- Agents modeled as graph nodes; edges connect agents within 50m (dynamic graph per scene).
- Graph Convolutional Network (2-3 layers) with message passing.
- Processes all agents jointly, allowing information flow across interactions.
- Same linear output head as baseline.

Implementation: PyTorch + PyTorch Geometric.

## Evaluation Plan

Metrics:

- Accuracy: Average Displacement Error (ADE) and Final Displacement Error (FDE) in meters.
- Latency: Wall-clock inference time (ms) on RTX 5080.
- Ablation: ADE change when removing specific edge types.

Independent Variables:

- Model type: LSTM baseline vs. GNN.
- Graph structure: fully-connected vs. k-nearest neighbors (k=5).
- Agent count: 2, 4, 6, 8+ agents per scene.
- Prediction horizon: 3s, 5s, 8s into future.

Baselines: Independent LSTM.

Statistical Rigor: 5 random seeds per configuration; t-tests for significance (p < 0.05).

Scenarios: Urban intersections, highway merges, parking lots (~1000 scenes per type from Waymo Motion).

## Expected Deliverables

- Benchmark harness (PyTorch): Reproducible code for training and evaluating LSTM and GNN models on Waymo Motion.
- Latency profiles: Detailed breakdowns showing where time is spent (data loading, forward pass, output computation).
- Accuracy-latency Pareto frontier plots: Clear visualization of the trade-off.
- Ablation study tables: Quantifying which agent interactions drive accuracy improvements.
- Technical report: pages analyzing results, failure cases, and recommendations.

## Timeline and Milestones

- Data loading pipeline reproduce LSTM baseline on Waymo Motion.
- Implement GNN model verify convergence on training data.
- Benchmark latency profile where time is spent.
- Run full ablation studies identify which interactions matter.
- Optimization pass (quantization/pruning if needed) error analysis.
- Write technical report finalize presentation and code.

## Risks and Mitigations

- GNN not faster than expected --- Latency > 200ms --- Pre-compute graph structure; use sparse convolution pivot to k-NN variant.
- GNN not more accurate	--- No accuracy gain --- interactions may not matter for this dataset. Still publishable result.
- GPU memory limits --- OOM errors on 5080 --- Reduce batch size or scene length profile memory early.

## Reproducibility Plan

- Code: Python 3.9+, PyTorch 2.0+, PyTorch Geometric 2.3+.
- Data: Waymo Motion dataset (requires academic credentials). Instructions for download included.
- Reproducibility: Fixed random seeds, version-controlled configs (YAML). All results generated via single bash script.
- Hardware: Tests run on RTX 5080.

## References

Waymo Open Dataset: Scalability and Classification Challenges. Accessed: 2024. https://waymo.com/open/