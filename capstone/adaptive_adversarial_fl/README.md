# Adaptive Attack-Defense Framework for Federated Learning

An adaptive attack-defense framework for studying the co-evolutionary arms race between attackers and defenders in federated phishing detection.

## Overview

This framework simulates the adaptive interplay between:
- **Attackers**: Who adapt their strategies to evade detection
- **Defenders**: Who adapt their defenses to detect emerging threats

The goal is to understand how both sides evolve over time and whether the system reaches an equilibrium.

## Attack Taxonomy

### 1. Defense-Aware Label Flip
- **Description**: Attacker knows about Byzantine-robust aggregation and adapts to stay within detection bounds
- **Evasion Strategy**:
  - `stay_under_bound`: Scale attacks to stay just below detection threshold
  - `distributed`: Distribute attack across multiple malicious clients
- **Adaptation**: Reduce flip ratio when detected, gradually increase when not

### 2. Defense-Aware Backdoor
- **Description**: Attacker knows about anomaly detection and uses gradual trigger injection
- **Trigger Types**:
  - `semantic`: Realistic phishing keywords (e.g., "urgent", "wire transfer")
  - `pixel`: Visual patterns (for image-based detection)
  - `word`: Word embedding patterns
- **Gradual Schedules**:
  - `linear`: Linear increase in injection rate
  - `exponential`: Exponential growth
  - `sigmoid`: Slow-start, rapid-middle, slow-end

### 3. Defense-Aware Model Poisoning
- **Description**: Attacker knows about gradient norm bounds and scales attacks accordingly
- **Strategies**:
  - `just_under_bound`: Scale to stay just under assumed bound
  - `adaptive`: Adapt based on detection history
- **Sybil Coordination**: Coordinate multiple malicious clients

### 4. Evasion-Poisoning Combo
- **Description**: Combines adversarial example crafting with model poisoning
- **Evasion Methods**:
  - `pgd`: Projected Gradient Descent
  - `fgsm`: Fast Gradient Sign Method
  - `cw`: Carlini-Wagner attack
- **Combo Strategies**:
  - `simultaneous`: Apply both together
  - `alternating`: Alternate between evasion and poisoning

## Defense Mechanisms

### 1. Multi-Round Anomaly Detection
- **Description**: Tracks client behavior over time to detect slow poisoning
- **Features**:
  - Window-based anomaly detection
  - Adaptive thresholds based on FP rate
  - Multiple baseline methods (moving average, exponential, percentile)

### 2. Honeypot Defense
- **Description**: Injects known-good updates from simulated honest clients
- **Features**:
  - Random or cluster-based honeypot placement
  - Deviation-based detection (cosine, euclidean, manhattan distance)
  - Configurable number of honeypots

### 3. Gradient Forensics
- **Description**: Analyzes gradient structure beyond norms
- **Analysis Methods**:
  - `pca`: Principal Component Analysis for coordination detection
  - `clustering`: DBSCAN clustering to find coordinated groups
  - `distance`: Distance-based coordination detection
  - `combined`: Multiple methods combined
- **Attack Type Classification**: Identifies attack type from gradient signatures

## Installation

```bash
# Clone repository
git clone <repository_url>
cd adaptive_adversarial_fl

# Install dependencies
pip install -r requirements.txt

# Install in development mode
pip install -e .
```

## Usage

### Basic Usage

```bash
# Run co-evolution experiment
python experiments/run_coevolution.py \
    --attack label_flip \
    --defense multi_round_anomaly \
    --rounds 20 \
    --output ./results \
    --visualize \
    --report
```

### Python API

```python
from src.attacks import DefenseAwareLabelFlip, AttackerKnowledge, AttackConfig
from src.defenses import MultiRoundAnomalyDetection, DefenderObservability
from src.coevolution import CoevolutionSimulator, CoevolutionConfig, CoevolutionAnalyzer

# Create attack
attack = DefenseAwareLabelFlip(
    attacker_knowledge=AttackerKnowledge(knows_defense=True),
    attack_config=AttackConfig(attack_type="label_flip"),
    flip_ratio=0.3,
)

# Create defense
defense = MultiRoundAnomalyDetection(
    defender_observability=DefenderObservability(),
    window_size=10,
    threshold_method="adaptive",
)

# Create simulator
config = CoevolutionConfig(num_rounds=20, num_clients=10, num_malicious=2)
simulator = CoevolutionSimulator(config, attack, defense)

# Run simulation
result = simulator.run()

# Analyze results
analyzer = CoevolutionAnalyzer(result)
print(analyzer.generate_summary())
```

## Configuration

### Attack Configuration

```yaml
# config/attacks.yaml
label_flip:
  flip_ratio: 0.3
  target_phishing_type: null  # "financial", "ecommerce", etc.
  evasion_strategy: "stay_under_bound"

backdoor:
  trigger_pattern: "semantic"
  injection_rate: 0.1
  gradual: true
  gradual_schedule: "linear"

model_poisoning:
  poison_strength: 5.0
  norm_bound_aware: true
  sybil_coordination: false

evasion_poisoning:
  evasion_method: "pgd"
  combo_strategy: "simultaneous"
  pgd_eps: 0.1
```

### Defense Configuration

```yaml
# config/defenses.yaml
multi_round_anomaly:
  window_size: 10
  threshold_method: "adaptive"
  baseline_method: "moving_average"

honeypot:
  num_honeypots: 3
  deviation_threshold: 2.0
  distance_metric: "cosine"

gradient_forensics:
  analysis_method: "pca"
  coordination_threshold: 0.9
  clustering_method: "dbscan"
```

## Co-evolution Results

### Equilibrium Analysis

The framework automatically detects when the system reaches equilibrium (when metrics stop changing significantly).

- **Equilibrium Detection**: Uses moving window to check if metrics have converged
- **Convergence Rate**: Ratio of equilibrium round to total rounds

### Arms Race Detection

Identifies oscillatory patterns characteristic of arms races:

```python
analyzer = CoevolutionAnalyzer(result)
arms_race = analyzer.identify_arms_race()

if arms_race["arms_race"]:
    print(f"Arms race detected with {arms_race['total_direction_changes']} direction changes")
```

### Metrics Tracked

- **Attack Success Rate (ASR)**: Proportion of successful attacks
- **Detection Rate (DR)**: Proportion of attacks detected
- **False Positive Rate (FPR)**: Proportion of honest clients falsely detected
- **Model Accuracy**: Accuracy of the federated model
- **Attacker Cost**: Computational cost of attack
- **Defender Cost**: Computational cost of defense
- **Defense Overhead**: Time overhead of defense mechanism

## Game-Theoretic Analysis

### Payoff Matrix

Compute payoff matrices for different attack-defense pairs:

```python
from src.game_theory import PayoffMatrix, AttackerUtility, DefenderUtility

payoff_matrix = PayoffMatrix(
    attack_actions=["label_flip", "backdoor", "model_poisoning"],
    defense_actions=["anomaly", "honeypot", "forensics"],
)
payoff_matrix.compute_payoffs(outcomes)
payoff_matrix.print_matrix()
```

### Nash Equilibrium

Find pure and mixed strategy Nash equilibria:

```python
from src.game_theory import NashEquilibriumAnalyzer

analyzer = NashEquilibriumAnalyzer(payoff_matrix)

# Pure Nash equilibrium
pure_eq = analyzer.find_pure_nash_equilibrium()

# Mixed Nash equilibrium
mixed_eq = analyzer.find_mixed_nash_equilibrium()

# Dominant strategies
dominant = analyzer.compute_dominant_strategies()
```

## Output Artifacts

### Visualizations

- `metrics_over_rounds.png`: All metrics over co-evolution rounds
- `costs_over_rounds.png`: Attacker and defender costs
- `attack_defense_dynamics.png`: ASR vs DR trajectory
- `equilibrium_analysis.png`: Moving average analysis
- `arms_race_heatmap.png`: Density heatmap of (ASR, DR) pairs

### Reports

- `coevolution_report.txt`: Text summary
- `coevolution_report.md`: Markdown report
- `coevolution_results.csv`: Per-round metrics
- `coevolution_summary_table.tex`: LaTeX summary table
- `coevolution_detailed_table.tex`: LaTeX detailed table

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run specific test category
pytest tests/test_attacks/ -v
pytest tests/test_defenses/ -v
pytest tests/test_coevolution/ -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html
```

## Research Contributions

1. **Adaptive Attacker Model**: Realistic attackers that adapt based on detection feedback
2. **Multi-Round Defense**: Defenses that track behavior over time
3. **Equilibrium Analysis**: Understanding when arms races stabilize
4. **Game-Theoretic Framework**: Nash equilibrium analysis for attack-defense games

## Citation

```bibtex
@software{adaptive_adversarial_fl,
  title={Adaptive Attack-Defense Framework for Federated Learning},
  author={Research Team},
  year={2025},
  url={https://github.com/your-repo/adaptive-adversarial-fl}
}
```

## License

MIT License

## Acknowledgments

This framework was developed for research on adversarial robustness in federated learning for phishing detection, directly relevant to PhD research on adaptive defense mechanisms.
