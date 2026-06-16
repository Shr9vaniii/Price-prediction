# 🏷️ Amazon Price Prediction Agent

> A multi-agent agentic system with a QLoRA fine-tuned Llama 3.2 ensemble that **outperforms GPT-5.1** on Amazon product price prediction — achieving **$39.85 MAE** vs GPT-5.1's **$44.74 MAE**.

---

## 🏆 Results

| Model | MAE ($) | MSE | R² |
|---|---|---|---|
| **Fine-tuned Llama 3.2 Ensemble (800K dataset)** | **$39.85** | — | — |
| GPT-5.1 (frontier baseline) | $44.74 | — | — |
| Fine-tuned Llama 3.2 Ensemble (20K subset) | $56.90 ± 11.62 | 10,267 | 53.3% |

> The full 800K model was trained on the complete Amazon product dataset using QLoRA fine-tuning. The 20K subset demonstrates the same architecture under compute constraints, already competitive with frontier models. Trained on the full dataset, our ensemble beats GPT-5.1 by ~$5 MAE.

---

## 🧠 Architecture

The system is a **multi-agent pipeline** with 6 specialized agents coordinated by a Planning Agent:

```
                         ┌─────────────────────────────────┐
                         │         Planning Agent          │
                         │  Orchestrates the full pipeline │
                         └──────┬──────────────┬───────────┘
                                │              │
                    ┌───────────▼──┐    ┌──────▼──────────┐
                    │ Scanner Agent│    │ Ensemble Agent  │
                    │ (GPT-5-mini) │    │ (weighted avg)  │
                    │ Scrapes RSS  │    └──┬───────┬───┬───┘
                    │ feeds, picks │       │       │   │
                    │ top 5 deals  │   ┌───▼──┐ ┌─▼──┐│┌──────────────┐
                    └──────────────┘   │Spec- │ │Fron││ Neural Network│
                                       │ialist│ │tier│││    Agent     │
                                       │Agent │ │Agt ││└──────────────┘
                                       │(Llama│ │(RAG│││  80% weight  │
                                       │3.2   │ │+   │││  to Frontier │
                                       │Fine- │ │Chro│││  10% each    │
                                       │tuned)│ │maDB│││  to others   │
                                       └──────┘ └────┘│└──────────────┘
                                        10%      80%        10%
                                           │       │          │
                                           └───────┴──────────┘
                                                   │
                                        ┌──────────▼──────────┐
                                        │   Messaging Agent   │
                                        │ Sends push alert if │
                                        │  discount > $50     │
                                        └─────────────────────┘
```

### Agent Breakdown

| Agent | Role |
|---|---|
| **PlanningAgent** | Master orchestrator — runs the full pipeline end to end |
| **ScannerAgent** | Scrapes RSS deal feeds, uses GPT-5-mini structured output to select top 5 deals with clear prices |
| **EnsembleAgent** | Combines 3 pricers using weighted average (Frontier 80%, Specialist 10%, NN 10%) |
| **SpecialistAgent** | QLoRA fine-tuned Llama 3.2-3B deployed on Modal (T4 GPU) |
| **FrontierAgent** | RAG pipeline — queries ChromaDB for similar products, feeds context to LLM |
| **NeuralNetworkAgent** | Deep neural network trained on Amazon dataset |
| **MessagingAgent** | Sends push notification when estimated discount exceeds $50 |

---

## 📦 Tech Stack

| Layer | Technology |
|---|---|
| Base Model | Meta Llama 3.2-3B |
| Fine-tuning | QLoRA (4-bit NF4, bitsandbytes) |
| Training Framework | HuggingFace PEFT + Transformers |
| GPU Deployment | Modal (T4 GPU, persistent HF volume cache) |
| Vector Store / RAG | ChromaDB |
| Text Preprocessing | GPT-5-mini (via OpenAI Structured Outputs) |
| Neural Network | PyTorch (deep_neural_network.py) |
| Visualization | Gradio UI + Plotly 3D t-SNE of ChromaDB embeddings |
| Agent Memory | Persistent JSON (memory.json) with Opportunity objects |
| Dataset | Amazon Products Dataset (~800K items, 8 categories) |

---

## 🗂️ Project Structure

```
├── notebooks/
│   ├── data_curation.ipynb              # Dataset sourcing and cleaning
│   ├── data preprocessing.ipynb         # Feature engineering & prep
│   ├── Baseline and traditional ML.ipynb # Baselines, NN, and evaluation
│   └── results.ipynb                    # Benchmarks and model comparison
│
├── agents/
│   ├── agent.py                         # Base Agent class with logging
│   ├── planning_agent.py                # Master orchestrator
│   ├── scanner_agent.py                 # RSS scraper + GPT-5-mini deal selector
│   ├── ensemble_agent.py                # Weighted ensemble (80/10/10)
│   ├── specialist_agent.py              # Fine-tuned Llama 3.2 pricer
│   ├── frontier_agent.py                # RAG pricer (ChromaDB + LLM)
│   ├── neural_network_agent.py          # Deep neural network pricer
│   ├── messaging_agent.py               # Push notification on deal found
│   ├── deals.py                         # Deal, Opportunity, ScrapedDeal models
│   ├── items.py                         # Item schema
│   ├── preprocessor.py                  # Text preprocessing pipeline
│   └── evaluator.py                     # Evaluation utilities
│
├── services/
│   ├── pricer_service.py                # Modal warm service (cached model)
│   ├── pricer_service2.py               # Modal cls with HF volume caching
│   └── pricer_ephemeral.py              # Modal cold-start function
│
├── deal_agent_framework.py              # Main entry point + t-SNE visualization
├── price_is_right.py                    # Gradio UI with real-time agent logs
├── deep_neural_network.py               # Neural network training script
├── llama.py                             # Raw Llama inference test script
├── log_utils.py                         # Colored console logging
├── memory.json                          # Agent memory (persisted opportunities)
├── requirements.txt
└── .env.example
```

---

## 🚀 Getting Started

### 1. Clone & install dependencies

```bash
git clone https://github.com/Shr9vaniii/Amazon-price-prediction-agent.git
cd Amazon-price-prediction-agent
pip install -r requirements.txt
```

### 2. Set up environment variables

```bash
cp .env.example .env
# Fill in your OPENAI_API_KEY and HUGGINGFACE_TOKEN
```

### 3. Set up Modal for GPU inference

```bash
pip install modal
modal setup
modal secret create huggingface-secret HUGGINGFACE_TOKEN=your_token_here
```

### 4. Deploy the Llama inference service

```bash
# Warm service with HF volume caching (recommended)
modal deploy pricer_service2.py

# Or cold-start (simpler, slower)
modal run pricer_ephemeral.py
```

### 5. Populate the ChromaDB vector store

```bash
# Run the data curation notebook first, then:
python deal_agent_framework.py
```

### 6. Launch the Gradio UI

```bash
python price_is_right.py
```

---

## 🤖 Fine-tuned Model

The QLoRA fine-tuned weights are on HuggingFace:

🔗 [`SmellyCat-pheobe/price-2026-02-08_09.18.49`](https://huggingface.co/SmellyCat-pheobe/price-2026-02-08_09.18.49)

| Config | Value |
|---|---|
| Base model | `meta-llama/Llama-3.2-3B` |
| Method | QLoRA (4-bit NF4) |
| Training dataset | Amazon Products (~800K descriptions) |
| Task | Predict price to nearest dollar from text description |
| Revision | `96c63b581a7e107d660a62cac2fa48cdc5e70efa` |

---

## 💡 How the Ensemble Works

The `EnsembleAgent` combines three specialized pricers using a **learned weighted average**:

```python
combined = frontier * 0.8 + specialist * 0.1 + neural_network * 0.1
```

- **Frontier Agent (80%)** — RAG pipeline: embeds the product description, retrieves the most similar products from ChromaDB, feeds them as context to an LLM for a grounded price estimate
- **Specialist Agent (10%)** — The fine-tuned Llama 3.2-3B, which has learned price distributions directly from 800K Amazon products
- **Neural Network Agent (10%)** — A deep neural network trained on structured product features

The `ScannerAgent` also uses GPT-5-mini with **Structured Outputs** to parse RSS deal feeds and extract only deals with clear, unambiguous prices before they enter the pipeline.

---

## 📊 Product Categories

The system handles 8 Amazon product categories tracked in the ChromaDB vector store:

`Appliances` · `Automotive` · `Cell Phones & Accessories` · `Electronics` · `Musical Instruments` · `Office Products` · `Tools & Home Improvement` · `Toys & Games`

---

## 🔭 Future Work

- [ ] Transfer fine-tuned weights to personal HuggingFace account with model card
- [ ] Retrain on full 800K dataset with A100 for further MAE reduction
- [ ] Add real-time deal alerting as a deployable web service
- [ ] Evaluate on out-of-domain product categories
- [ ] Add confidence intervals to ensemble output
- [ ] Replace hardcoded ensemble weights with a learned meta-model

---

## 📄 License

MIT