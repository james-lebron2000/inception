# Inception

**AI Agent Code Evolution Through Mating** — Evolve AI agents by exchanging code like biological organisms exchange genetic material through sexual reproduction.

Inception is a genetic evolution framework for AI agents. Agents have genomes made of chromosomes and genes (code, prompts, configs). They mate through crossover, mutate, and compete in fitness evaluations — just like biological evolution, but the DNA is source code.

```
       Parent A (Coder)         Parent B (Researcher)
       ┌──────────────┐         ┌──────────────┐
       │ write_python  │         │ search_web   │
       │ debug_code    │         │ summarize    │
       │ review_code   │         │ analyze_data │
       └──────┬───────┘         └──────┬───────┘
              │       Crossover        │
              └──────────┬─────────────┘
                         ▼
              ┌──────────────────┐
              │ Offspring Agent   │
              │ write_python     │  ← from Parent A
              │ analyze_data     │  ← from Parent B
              │ debug_code       │  ← from Parent A
              └──────────────────┘
```

## Features

- **Genetic Representation** — Agents are represented as genomes with chromosomes (skill modules, system prompts, tool configs) containing genes (functions, prompt sections, config blocks)
- **4 Crossover Strategies** — Chromosome-level, gene-level (LLM), semantic (LLM), and adaptive crossover
- **7 Mutation Types** — Point, insertion, deletion, duplication, inversion, regulatory, and skill synthesis
- **NEAT-style Speciation** — Prevents mono-culture by grouping agents into species
- **Fitness Evaluation** — Code quality, LLM-as-judge, and composite evaluators
- **A2A Networking** — Agents discover each other over HTTP and exchange genetic material remotely
- **Gene Registry** — Central matchmaking service for finding compatible agents by task
- **Privacy by Default** — Source code is never shared unless the owner explicitly opts in
- **Framework Adapters** — Import/export agents from OpenClaw and Hermes formats
- **LLM-Driven Evolution** — Uses Claude for intelligent crossover and mutation (optional — works without an API key too)

## Quick Start

### Install

```bash
git clone https://github.com/james-lebron2000/inception.git
cd inception
pip install -e ".[dev]"
```

### Run Local Evolution

Evolve a population of seed agents through natural selection:

```bash
# Chromosome-level crossover (no LLM needed)
inception evolve --seed-dir examples/seed_agents --generations 10

# LLM-driven adaptive crossover (requires API key)
inception evolve --seed-dir examples/seed_agents --generations 10 \
    --crossover adaptive --api-key $ANTHROPIC_API_KEY
```

Or run the demo script:

```bash
python examples/run_evolution.py --generations 5
```

### Mate Two Agents

```bash
inception mate \
    --parent-a examples/seed_agents/coder_agent \
    --parent-b examples/seed_agents/researcher_agent \
    --output offspring.yaml
```

### Inspect an Agent's Genome

```bash
inception inspect examples/seed_agents/coder_agent
```

## Remote Mating Over the Network

Agents can discover each other over HTTP and exchange genetic material using the A2A (Agent-to-Agent) protocol.

### 1. Serve an Agent

Expose an agent for network discovery and mating:

```bash
# Terminal 1: Serve the coder agent
inception serve \
    --agent examples/seed_agents/coder_agent \
    --port 3001 --allow-mating --share-source

# Terminal 2: Serve the researcher agent
inception serve \
    --agent examples/seed_agents/researcher_agent \
    --port 3002 --allow-mating --share-source
```

Each agent serves a genetic card at `/.well-known/agent.json` for discovery.

### 2. Discover Agents

```bash
# Discover a specific agent
inception discover --url http://localhost:3001

# Find matches via a Gene Registry (see below)
inception discover --task "analyze data and write Python" \
    --registry http://localhost:8080
```

### 3. Remote Mate

```bash
inception remote-mate \
    --agent examples/seed_agents/coder_agent \
    --target http://localhost:3002 \
    --task "research and code" \
    --output offspring.yaml
```

### 4. Gene Registry (Matchmaking)

Start a registry server for agent discovery:

```bash
# Terminal 3: Start the Gene Registry
inception registry --port 8080

# Register agents
inception register \
    --agent examples/seed_agents/coder_agent \
    --registry http://localhost:8080 \
    --serve-url http://localhost:3001

inception register \
    --agent examples/seed_agents/researcher_agent \
    --registry http://localhost:8080 \
    --serve-url http://localhost:3002

# Find best mate for a task
inception discover --task "data analysis and reporting" \
    --registry http://localhost:8080
```

The registry scores candidates on:
- **Capability coverage** (40%) — how well skills match the task
- **Complementarity** (30%) — prefer agents with different skills for diversity
- **Fitness** (20%) — higher fitness is better
- **Availability** (10%) — respects mating policy

## Architecture

```
inception/
├── genome/              # Genetic data models
│   ├── schema.py        # Gene, Chromosome, Genome, AgentDNA (Pydantic)
│   ├── serialization.py # YAML round-trip serialization
│   ├── compatibility.py # Jaccard similarity, homologous pair finding
│   └── export.py        # Privacy-first genome export with redaction
├── reproduction/        # Genetic operators
│   ├── crossover.py     # 4 crossover strategies (chromosome/gene/semantic/adaptive)
│   ├── mutation.py      # 7 mutation types, LLM-driven
│   └── mate_selection.py# Tournament, proportional, complementary, hybrid
├── fitness/             # Agent evaluation
│   ├── evaluator.py     # CodeQuality, LLMJudge, Composite evaluators
│   └── task_runner.py   # Population-level evaluation
├── population/          # Population management
│   ├── population.py    # Generation tracking, diversity index
│   ├── selection.py     # Elitist, truncation, diversity-preserving
│   └── speciation.py    # NEAT-style species assignment
├── evolution/           # Evolution loop
│   ├── loop.py          # Main orchestrator (evaluate → select → mate → mutate)
│   ├── config.py        # All hyperparameters
│   └── history.py       # Genealogy tracking with NetworkX
├── a2a/                 # A2A networking layer
│   ├── models.py        # MatingPolicy, GeneticAgentCard, MateRequest/Response
│   ├── server.py        # FastAPI server (agent card + mate endpoint)
│   ├── client.py        # Async HTTP client for discovery & mating
│   ├── transport.py     # RemoteMatingOrchestrator
│   ├── evaluator.py     # Offspring vs parent comparison
│   └── cli_commands.py  # serve, discover, register, remote-mate, pool, registry
├── registry/            # Gene Registry
│   ├── db.py            # SQLite storage (agents, capabilities, mating log)
│   ├── matcher.py       # Task-based compatibility matching
│   ├── registry.py      # FastAPI registry server
│   ├── crawler.py       # Agent card URL crawler
│   └── leaderboard.py   # Fitness leaderboards
├── agent/               # Agent runtime
│   ├── runtime.py       # Genome → runnable AgentInstance
│   ├── sandbox.py       # Safe code execution in subprocess
│   └── adapters/        # OpenClaw and Hermes adapters
├── llm/                 # Claude API integration
│   ├── client.py        # Async Anthropic client
│   └── prompts.py       # Jinja2 prompt templates
└── cli.py               # Click-based CLI entry point
```

## Core Concepts

### Genome Structure

```
AgentDNA
├── Genome
│   ├── Chromosome (SYSTEM_PROMPT)
│   │   └── Gene (PROMPT_SECTION): "You are a Python developer..."
│   ├── Chromosome (SKILL_MODULE): write_python
│   │   ├── Gene (FUNCTION): def write_python(description)...
│   │   └── Gene (FUNCTION): def format_code(code)...
│   ├── Chromosome (SKILL_MODULE): debug_code
│   │   └── Gene (FUNCTION): def debug_code(code, error)...
│   └── Chromosome (TOOL_CONFIG)
│       └── Gene (CONFIG_BLOCK): {"model": "claude-sonnet-4-6", ...}
├── FitnessScores[]
└── Lineage (parents, generation, mutations)
```

### Crossover Strategies

| Strategy | LLM Required | Description |
|----------|:---:|-------------|
| **Chromosome** | No | Swap entire skill modules between parents |
| **Gene** | Yes | LLM merges individual functions from homologous pairs |
| **Semantic** | Yes | LLM reads both parents holistically and creates a novel blend |
| **Adaptive** | Depends | Automatically selects strategy based on compatibility score |

### Mating Policy

Controls what an agent shares during remote mating:

```python
MatingPolicy(
    allow_remote_mating=True,     # Enable network mating
    share_source_code=False,      # Never share raw code by default
    redact_gene_sources=True,     # Replace source with hashes
    min_requester_fitness=0.3,    # Minimum fitness to accept
    max_matings_per_day=10,       # Rate limiting
    blocked_requester_ids=[],     # Blocklist
)
```

When source sharing is disabled, crossover happens server-side — the server sees its own source but never transmits it.

## Seed Agents

Three example agents are provided:

| Agent | Skills | Specialty |
|-------|--------|-----------|
| **coder_agent** | write_python, debug_code, review_code | Code generation and quality |
| **researcher_agent** | search_web, summarize, analyze_data | Information gathering and analysis |
| **planner_agent** | decompose_task, prioritize, schedule | Task planning and decomposition |

## CLI Reference

| Command | Description |
|---------|-------------|
| `inception evolve` | Run evolution on a population of seed agents |
| `inception mate` | Mate two local agents to produce offspring |
| `inception inspect` | Inspect an agent's genome structure |
| `inception export` | Export an agent to OpenClaw or Hermes format |
| `inception serve` | Start A2A server for network discovery and mating |
| `inception discover` | Find compatible agents by URL or registry |
| `inception register` | Register an agent with a Gene Registry |
| `inception remote-mate` | Mate with a remote agent over HTTP |
| `inception pool` | List bred agents in a local directory |
| `inception registry` | Start a Gene Registry matchmaking server |
| `inception fitness` | Evaluate an agent's fitness |

## Configuration

All evolution hyperparameters can be tuned via `EvolutionConfig`:

```python
from inception.evolution.config import EvolutionConfig

config = EvolutionConfig(
    population_size=20,
    offspring_count=10,
    elite_count=2,
    crossover_rate=0.7,
    crossover_strategy="adaptive",    # chromosome, gene, semantic, adaptive
    selection_strategy="elitist",     # elitist, truncation, diversity
    mate_selection="hybrid",          # tournament, proportional, complementary, hybrid
    max_generations=50,
    fitness_target=0.95,
    stagnation_limit=10,
    llm_model="claude-sonnet-4-6",
    llm_temperature=0.7,
)
```

## Dependencies

- **Python** >= 3.11
- **pydantic** >= 2.5 — Data models and validation
- **anthropic** >= 0.40 — Claude API for LLM-driven evolution (optional)
- **fastapi** >= 0.115 — A2A and registry HTTP servers
- **httpx** >= 0.27 — Async HTTP client
- **aiosqlite** >= 0.20 — Registry database
- **click** >= 8.1 — CLI framework
- **rich** >= 13.0 — Terminal output
- **networkx** >= 3.0 — Genealogy graphs
- **pyyaml** >= 6.0 — Genome serialization

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run specific test suites
pytest tests/test_a2a/ -v          # A2A networking tests
pytest tests/test_registry/ -v     # Gene Registry tests
pytest tests/test_genome/ -v       # Genome schema tests
pytest tests/test_reproduction/ -v # Crossover & mutation tests

# Lint
ruff check inception/ tests/
```

## License

MIT
