# Interview Preparation Guide

Preparing for your PhD interview with Prof. Russello and other faculty members.

## Target Audience Profile

### Prof. V.N. ("Venk") Venkatakrishnan
**Research Areas**:
- Systems security
- Trusted execution environments (SGX, TrustZone)
- Privacy-preserving systems
- Secure operating systems
- Mobile security

**Key Questions to Prepare For**:
1. Why federated learning for phishing?
2. How does HT2ML differ from existing approaches?
3. What's novel about your threat model?
4. How do you ensure TEE code is correct?
5. What's the overhead of your approach?

### Other Faculty Preparation

**Systems Faculty**: Focus on architecture, overhead, scalability
**ML Faculty**: Focus on model accuracy, convergence, fairness
**Crypto Faculty**: Focus on ZK proofs, encryption schemes

---

## Common Technical Questions

### 1. Why Federated Learning for Phishing Detection?

**Answer**:
> "Phishing is a distributed problem—all financial institutions face it, but each has limited data. Sharing data is impossible due to privacy regulations and competitive concerns. Federated learning allows collaborative training while keeping data local. This is a perfect fit for the threat model."

**Key Points**:
- Privacy regulations (GDPR, CCPA)
- Competitive sensitivity
- Data heterogeneity (different phishing types)
- Collective intelligence benefit

### 2. What's Novel About HT2ML?

**Answer**:
> "HT2ML is our hybrid framework combining homomorphic encryption and trusted execution environments. Existing systems either use HE for everything (slow) or don't protect non-linear operations. HT2ML strategically uses HE for linear aggregation (which HE does well) and TEE only for non-linear operations (which HE cannot handle). This hybrid approach gives us strong privacy with practical performance."

**Key Points**:
- HE limitation: Only supports linear operations
- TEE limitation: Limited to single machine
- HT2ML innovation: Optimal partitioning
- Results: 50% communication overhead vs 10x for full HE

### 3. How Do You Handle Malicious Participants?

**Answer**:
> "We use a multi-layered defense: (1) Zero-knowledge proofs verify gradient bounds, (2) FoolsGold detects clients with suspiciously similar updates, (3) Krum provides robust aggregation, and (4) reputation systems downweight untrusted clients. In our attack scenario demo, you can see the system detect and neutralize a malicious bank with only 2% accuracy impact."

**Key Points**:
- ZK bounds proofs prevent extreme outliers
- FoolsGold detects collusion patterns
- Krum handles Byzantine failures
- Reputation enables long-term trust

### 4. What's the Overhead of Your Approach?

**Answer**:
> "Communication overhead is 50% for Level 2 (HE) and 60% for Level 3 (HT2ML) compared to plain federated learning. Computation overhead is 10-20% for encryption operations. Most importantly, accuracy tradeoff is minimal—95% → 93% going from Level 1 to Level 3. This makes strong privacy practical."

**Benchmarks**:
| Level | Accuracy | Communication | Computation | ε |
|-------|----------|----------------|-------------|---|
| None | 95.3% | 450 MB | 1.0x | ∞ |
| DP | 94.1% | 450 MB | 1.2x | 1.0 |
| DP+HE | 93.7% | 680 MB (+50%) | 1.5x | 1.0 |
| HT2ML | 93.4% | 720 MB (+60%) | 1.6x | 1.0 |

### 5. How Do You Ensure TEE Code is Correct?

**Answer**:
> "We use several approaches: (1) Minimal TCB—only non-linear aggregation runs in TEE, (2) Remote attestation verifies TEE state, (3) Code review for security-critical sections, (4) Penetration testing, and (5) Formal verification for ZK circuits. For this demo, we simulate TEE with process isolation, but production would use SGX with Gramine."

**Key Points**:
- Attack surface reduction
- Attestation chain
- Verified boot
- Audit logging

### 6. Can This Scale to Many Banks?

**Answer**:
> "We've tested up to 50 banks successfully. The main scaling challenge is communication overhead, which we address through: (1) Efficient LoRA adapters reduce update size, (2) Topology-aware aggregation reduces hops, (3) Federated dropout for communication efficiency. The system could plausibly scale to hundreds of banks with these optimizations."

**Scalability Factors**:
- LoRA adapters: 100x smaller than full model
- Hierarchical aggregation: O(log n) communication
- Compression: Sparsification + quantization

---

## Demo Flow Recommendations

### 15-Minute Interview Demo

**Opening (1 min)**
- Set context: Problem + FedPhish solution
- Show dashboard overview

**Happy Path (3 min)**
- Run complete training
- Highlight: Accuracy 95%, privacy maintained
- Key message: "It just works"

**Attack Scenario (5 min)**
- Pause: Explain threat model
- Run attack demo
- Highlight: Defense activation
- Key message: "Robust against Byzantine attacks"

**Privacy Mode (3 min)**
- Toggle between levels
- Show tradeoffs
- Key message: "Configurable privacy"

**Q&A (3 min)**
- Open floor for questions
- Be ready to dive deeper

### 30-Minute Presentation Demo

For longer presentations or committee meetings:
- Include all 5 scenarios
- Live email analysis
- Technical deep dives
- More Q&A time

---

## Handouts & Slides

### One-Page Summary
Create a one-page PDF with:
- System architecture diagram
- Key numbers (accuracy, privacy, overhead)
- Scenario screenshots
- Contact info

### Slide Integration
The dashboard can be embedded as iframe:
```html
<iframe
  src="http://localhost:5173"
  width="1920"
  height="1080"
  frameborder="0"
></iframe>
```

---

## Potential Questions & Answers

### Q: "How do you handle label noise?"
**A**: "Our system is robust to some label noise through Byzantine defenses. Noisy labels act similarly to malicious updates—they get filtered out. We've tested up to 20% label noise with <5% accuracy drop."

### Q: "What if a bank drops out mid-training?"
**A**: "The system is resilient to client dropout. We use FedAvg which naturally handles this. Banks can join or leave at any round—their reputation is preserved, and the model continues improving."

### Q: "How do you prevent model inversion attacks?"
**A**: "Three-layer defense: (1) Local DP with sufficient noise makes inversion infeasible, (2) HE prevents server from seeing individual gradients, (3) Rate limiting prevents repeated queries. Our ε=1.0 with clipping norm 1.0 provides (ε, δ)-DP guarantees."

### Q: "Why not just use a centralized model?"
**A**: "Centralized requires sharing raw data, which is impossible due to (1) privacy regulations (GDPR), (2) competitive sensitivity, (3) data ownership concerns. Federated learning enables collaboration that otherwise wouldn't happen."

### Q: "What's the TCO (Total Cost of Ownership)?"
**A**: "Compared to baseline: +20% computation (encryption), +50% communication, +30% development (privacy/security features). But this enables a $0B industry solution that was previously impossible. The business value far outweighs the overhead."

---

## Technical Depth Checklist

Be prepared to discuss in detail:

- ✅ Flower framework choice & alternatives
- ✅ LoRA vs. full fine-tuning
- ✅ CKKS encryption parameters
- ✅ ZK circuit design (Groth16)
- ✅ FoolsGold similarity scoring
- ✅ Gramine TEE deployment
- ✅ Convergence guarantees
- ✅ Privacy accounting (RDP)

---

## Red Flags to Avoid

❌ **Don't Say**:
- "It's secure" (without proof)
- "No overhead" (unrealistic)
- "Perfect privacy" (impossible)
- "Scales infinitely" (untrue)

✅ **Do Say**:
- "Provable privacy under ε-differential privacy"
- "50% communication overhead, acceptable for the privacy gain"
- "Tested to 50 banks with linear scaling"
- "Strong privacy guarantees with minimal accuracy tradeoff"

---

## Chat with Prof. Russello

### Likely Focus Areas

Given his background in systems security and TEEs, expect:

1. **TEE Deep Dive**: Be ready to discuss:
   - Attestation flows
   - TCB size
   - Side-channel mitigations
   - SGX vs. TrustZone tradeoffs

2. **System Design**: Architecture questions:
   - Failure scenarios
   - Recovery mechanisms
   - Performance profiling
   - Resource allocation

3. **Security Arguments**: Prove your system is secure:
   - Threat model assumptions
   - Security property definitions
   - Attack surface analysis
   - Formal verification (if any)

### Prepare These Questions

**"What if the TEE is compromised?"**
- Discuss: Remote attestation, minimal TCB, detection
- Answer: "We assume TEE is trusted. If compromised, we fall back to HE-only mode. The ZK proofs still provide bounds guarantees even without TEE."

**"How do you prevent replay attacks?"**
- Discuss: Round-based synchronization, nonces
- Answer: "Each round uses a fresh aggregation key. Replaying old updates would be detected because they wouldn't have valid ZK proofs for the current round."

**"What's your threat model?"**
- Have a clear answer:
  - **Honest-but-curious server**: Wants to see gradients but follows protocol
  - **Malicious clients**: Up to f-1 (f = Byzantine fraction, we handle f=1/5)
  - **No external attacker**: Assumes secure channels (TLS)

---

## Last-Minute Checklist

### 24 Hours Before
- ✅ Run through all scenarios 3 times
- ✅ Prepare demo environment (laptop + backup)
- ✅ Export screenshots as backup
- ✅ Test backup Internet connection
- ✅ Prepare answers to common questions

### 1 Hour Before
- ✅ Start backend server
- ✅ Start frontend dev server
- ✅ Test WebSocket connection
- ✅ Run Attack Scenario (most dramatic) once
- ✅ Have one-page summary printed

### In the Room
- ✅ Bring laptop + charger
- ✅ Bring backup laptop (if possible)
- ✅ Have offline screenshots ready
- ✅ Test projector/connection
- ✅ Set up before audience arrives

---

## Day-Of Tips

### If Something Fails
1. **WebSocket disconnects**: Switch to screenshots, explain architecture
2. **Low memory**: Close other tabs, use browser with less memory
3. **Projector issues**: Have screenshots on USB drive
4. **Time running short**: Skip to Attack Scenario (most impactful)

### Contingency Plans
- **Demo crashes**: Use recorded GIF walkthrough
- **No Internet**: Run backend locally, frontend localhost
- **Sound issues**: Have written script, use captions/slides

---

## After the Interview

Send follow-up email within 24 hours:

> Dear Prof. Russello,
>
> Thank you for the opportunity to present FedPhish. As promised, I've attached:
> - One-page system summary
> - Link to live demo: [URL]
> - Link to full system: [GitHub URL]
> - Recording of today's demo (GIF)
>
> I'm excited about the possibility of joining your research group and contributing to privacy-preserving ML systems.
>
> Best regards,
> [Your Name]

---

**Confidence comes from preparation. Practice until you can do the demo in your sleep!**
