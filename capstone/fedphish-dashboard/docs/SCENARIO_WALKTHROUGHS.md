# Demo Scenario Walkthroughs

Complete guide for walking through each demo scenario.

## Table of Contents
1. [Happy Path](#scenario-1-happy-path)
2. [Non-IID Challenge](#scenario-2-non-iid-challenge)
3. [Attack Scenario](#scenario-3-attack-scenario)
4. [Privacy Mode](#scenario-4-privacy-mode)
5. [Live Demo (Email Analysis)](#scenario-5-live-demo)

---

## Scenario 1: Happy Path

### Purpose
Demonstrate smooth federated learning convergence with all honest banks.

### Setup
1. Select "Happy Path" from scenario dropdown
2. Verify all 5 banks show "idle" status
3. Confirm Round: 0/20, Accuracy: baseline

### Demo Script

#### Introduction (30 seconds)
> "Let me show you our FedPhish system in action. This dashboard visualizes federated learning in real-time. We have 5 banks—Chase, Bank of America, Wells Fargo, Citibank, and US Bank—collaboratively training a phishing detector without sharing their customer data."

#### Start Training (15 seconds)
> [Click Start]
>
> "As training begins, each bank trains locally on their private data. You can see the blue indicators showing they're actively training. The animated flow shows encrypted updates being sent to our aggregation server."

#### First Few Rounds (30 seconds)
> "Notice the accuracy climbing quickly—from 80% to 87% in just 5 rounds. The green locks show all communication is encrypted with homomorphic encryption. The checkmarks indicate zero-knowledge proofs are being verified for each update."

#### Mid-Training (30 seconds)
> "By round 10, we're at 93% accuracy. The privacy budget on the right shows we've consumed 10 out of 20 epsilon—still well within our privacy budget. All banks are contributing equally, shown by their similar accuracy bars."

#### Convergence (30 seconds)
> "At round 15, we've converged to 95% accuracy. All banks are now green (complete), having successfully trained a robust phishing detector while maintaining strict privacy guarantees throughout."

### Key Visualizations
- ✅ All banks turn from idle → training (blue) → complete (green)
- ✅ Accuracy chart shows smooth climb
- ✅ Communication flow animation runs continuously
- ✅ Privacy budget accumulates steadily
- ✅ ZK proof checkmarks for all banks

### Transition
> "Now let me show you what happens when banks have different types of data..."

---

## Scenario 2: Non-IID Challenge

### Purpose
Demonstrate federated learning on heterogeneous data distributions.

### Setup
1. Select "Non-IID Challenge" from dropdown
2. Click Reset to clear previous state
3. Explain specialization: "Each bank sees different phishing types"

### Demo Script

#### Introduction (30 seconds)
> "In the real world, banks don't see the same types of phishing attacks. Chase might see account verification scams, Bank of America might see urgent requests, Wells Fargo might see gift scams. This scenario shows how federated learning adapts to heterogeneous data."

#### Initial State (20 seconds)
> "Notice the banks start with different base accuracies. Chase specializes in account verification phishing, so they're better at detecting that type. But the global model needs to learn all types."

#### Training Progress (60 seconds)
> [Start training]
>
> "Watch how the accuracy varies more than in the happy path. Round 1 shows banks between 70-85% accuracy. But over time... [wait for round 10] ... by round 15, they're converging toward 90%."

#### Challenge & Solution (30 seconds)
> "The challenge is that each bank's local data is biased toward their specialty. Through federated learning, they share model updates (not raw data) and learn from each other's experiences. By round 20, the global model handles all phishing types effectively."

### Key Visualizations
- ✅ Bank accuracy bars show different heights initially
- ✅ Slower convergence than happy path
- ✅ Eventual success despite heterogeneity

### Talking Points
- "Federated learning enables knowledge transfer without data sharing"
- "Non-IID data is the real-world norm, not exception"
- "Our system adapts to bank specializations"

---

## Scenario 3: Attack Scenario

### Purpose
Showcase Byzantine-robust aggregation.

### Setup
1. Select "Attack Scenario" from dropdown
2. Reset if needed
3. Point out Bank 2: "This will become malicious at round 5"

### Demo Script

#### Introduction (30 seconds)
> "Now, a critical scenario: what if a participating bank is malicious? Attackers could try to poison the model by sending corrupted updates. Let me show you our defense systems."

#### Normal Training (30 seconds)
> [Start training, wait for round 4]
>
> "Rounds 1-4 show normal training. All banks are honest (green), accuracy climbs to 90%."

#### Attack Launch (15 seconds)
> "Watch closely at round 5..."
> [Round 5 begins]
>
> "🚨 Bank 2 turns RED! Our system detects malicious updates immediately. The alert banner shows 'Malicious Activity Detected'."

#### Defense Activation (45 seconds)
> "Here's what's happening under the hood:
> 1. Our similarity-based defense (FoolsGold) detects Bank 2's updates are very different from others
> 2. The reputation system downweights Bank 2 from 100% to 70%
> 3. Malicious updates are filtered out
> 4. Accuracy dips slightly but recovers"

> [Point to reputation chart]
> "Watch Bank 2's reputation drop as the attack continues."

#### Recovery (30 seconds)
> "Despite the ongoing attack, our system maintains model performance. By round 15, accuracy has recovered to 92% because we're only aggregating from honest banks."

### Key Visualizations
- ✅ Red alert banner appears at round 5
- ✅ Bank 2 turns red and shows "MALICIOUS"
- ✅ Reputation bar for Bank 2 drops (100% → 70% → 40% → 10%)
- ✅ Accuracy dips then recovers
- ✅ ZK proof for Bank 2 shows "INVALID"

### Dramatic Pause
> "This is a key innovation: **provable robustness**. Even with a malicious participant, we maintain accuracy through Byzantine-resilient aggregation."

---

## Scenario 4: Privacy Mode

### Purpose
Demonstrate HT2ML privacy mechanisms with level toggling.

### Setup
1. Select "Privacy Mode" from dropdown
2. Explain: "I'll toggle between privacy levels during training"
3. Show privacy panel on right

### Demo Script

#### Introduction (30 seconds)
> "FedPhish offers three privacy levels with different privacy-utility tradeoffs. Let me demonstrate each level."

#### Level 1: DP Only (30 seconds)
> [Click "Level 1" button]
>
> "Level 1 adds local differential privacy. Each bank adds noise to their gradients before sending. The privacy budget (ε) shows we're at 1.0 per round. This provides basic privacy but the server could theoretically see individual updates."

#### Level 2: DP + HE (30 seconds)
> [Click "Level 2" button]
>
> "Level 2 adds homomorphic encryption. Now updates are encrypted—the server can't see individual gradients, only compute the aggregated result in encrypted form. The lock icons show encryption is active."

#### Level 3: Full HT2ML (30 seconds)
> [Click "Level 3" button]
>
> "Level 3 brings in trusted execution environments for non-linear operations. This is our innovation called HT2ML—hybrid HE and TEE. Linear operations via encryption, non-linear via secure enclaves."

#### Tradeoff Discussion (30 seconds)
> "Notice the accuracy slightly decreases as privacy increases. This is the classic privacy-utility tradeoff. But even at Level 3, we maintain 91% accuracy—still very effective for phishing detection."

### Key Visualizations
- ✅ Privacy level buttons highlight (yellow → blue → green)
- ✅ Privacy budget bar fills up
- ✅ Encryption status indicators activate
- ✅ Accuracy adjusts slightly with each level

### Technical Deep Dive
> "Our HT2ML framework optimizes this tradeoff by:
> - Using HE for expensive linear operations (aggregation)
> - Using TEE only for necessary non-linear operations
> - Achieving near-plain-text performance with strong privacy"

---

## Scenario 5: Live Demo (Email Analysis)

### Purpose
Interactive phishing email analysis with explanations.

### Setup
1. This would be a separate view/panel
2. Have sample emails ready

### Demo Script

#### Introduction (20 seconds)
> "Let me show you how our model explains its predictions. This transparency is crucial for security analysts who need to trust the system."

#### Sample 1: Phishing Email (60 seconds)
> [Enter or select phishing email]
>
> "This email says 'URGENT: Account Verification Required'. Let's analyze..."

> [Feature extraction animation]
>
> "The system identifies:
> - Urgent language: 'URGENT', 'immediately', 'suspended'
> - Suspicious URL: verify-account-secure-login.com
> - Request for sensitive information
> - No personalization"

> [Prediction appears]
>
> "Result: 98% confident this is PHISHING. The attention heatmap shows the model focused on 'URGENT', the URL, and 'verify'—all red flags."

#### Sample 2: Legitimate Email (30 seconds)
> [Select legitimate email]
>
> "Now let's analyze a legitimate bank email. You can see the prediction is LEGITIMATE with 95% confidence. The explanation notes proper personalization, no urgent threats, and a verified domain."

### Key Visualizations
- ✅ Feature extraction table (animated)
- ✅ Attention heatmap with token highlighting
- ✅ Prediction badge with confidence
- ✅ Textual explanation

---

## General Tips

### Pacing
- **Total demo time**: 15-20 minutes
- **Happy Path**: 2-3 minutes
- **Attack Scenario**: 4-5 minutes (highlight)
- **Privacy Mode**: 2-3 minutes
- **Live Demo**: 3-4 minutes

### Transitions
> "Now that you've seen it working normally..."
> "Let me show you something more challenging..."
> "The key innovation here is..."

### Q&A Preparation
- **"How does this differ from standard FL?"**: HT2ML hybrid approach
- **"What's the overhead?"**: 50% communication, 10% computation
- **"Can this scale?"**: Tested to 50 banks
- **"What about GPU memory?"**: LoRA adapters make it efficient

### Technical Details to Know
- **Framework**: Flower (FL) + PyTorch
- **Privacy**: Local DP (ε=1.0), CKKS encryption, Gramine TEE
- **Defense**: FoolsGold + Krum + Reputation
- **Model**: DistilBERT with LoRA (rank=8)
- **Banks**: 5-50 tested, 5 typical for demos

---

## Recording Your Demo

### Screenshots to Capture
1. Initial state (all banks idle)
2. Mid-training (all banks active)
3. Attack detection (red alert)
4. Training complete (all green, 95% accuracy)

### GIF Recording
1. Use Peek (Linux) or LICEcap (Windows/Mac)
2. Select dashboard window (1920×1080)
3. Record Attack Scenario (most dramatic)
4. Export as GIF, keep file size <10MB

### For Slides
- Use "Happy Path" for clean visuals
- Screenshot at round 10 (87% accuracy)
- Include in presentation as GIF

---

**Practice makes perfect! Rehearse each scenario 3-5 times before the actual demo.**
