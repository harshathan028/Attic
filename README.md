# AI Content Factory - Custom Multi-Agent Orchestrator

> **Implemented adaptive multi-agent orchestration with live data ingestion and cross-run learning memory enabling strategy reuse and performance-driven agent behavior.**

A production-ready content generation system featuring:
- 🤖 **Multi-Agent Orchestration**: Research, Write, Fact-check, and Optimize.
- 🌐 **Real-Time Web Research**: Integrated DuckDuckGo search for live accuracy.
- 📡 **Live Data Ingestion**: RSS, CSV, PDF, and REST API support.
- 🧠 **Recursive Learning**: Memory system that improves strategies over time.
- 💾 **Vector Memory**: ChromaDB-powered semantic retrieval
- 🔌 **Multi-LLM Support**: Gemini (cloud) or Ollama (local)
- 📊 **Quality Evaluation**: Readability, structure, keyword density scoring

**No LangChain, CrewAI, or AutoGen** - Pure custom orchestration architecture.

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        CLI Interface                            │
│                 python main.py --topic "..."                    │
└─────────────────────────────┬───────────────────────────────────┘
                              │
┌─────────────────────────────▼───────────────────────────────────┐
│                     ContentPipeline                             │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐  │
│  │  Strategy   │  │    Run      │  │     Performance         │  │
│  │   Memory    │◄─┤   History   │◄─┤      Tracker            │  │
│  └─────────────┘  └─────────────┘  └─────────────────────────┘  │
└─────────────────────────────┬───────────────────────────────────┘
                              │
┌─────────────────────────────▼───────────────────────────────────┐
│                      AgentManager                               │
│                                                                 │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐        │
│  │ LiveData │  │ Research │  │  Writer  │  │FactCheck │  ...   │
│  │  Agent   │──▶│  Agent   │──▶│  Agent   │──▶│  Agent   │       │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘        │
│       │             │             │             │               │
└───────┼─────────────┼─────────────┼─────────────┼───────────────┘
        │             │             │             │
┌───────▼─────────────▼─────────────▼─────────────▼───────────────┐
│                         Tool Layer                              │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────────────────┐ │
│  │   LLM   │  │  Vector │  │  Search │  │    Live Data Tools  │ │
│  │ Client  │  │  Store  │  │  Tool   │  │  RSS│API│PDF│CSV    │ │
│  └─────────┘  └─────────┘  └─────────┘  └─────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🚀 Quick Start

### Installation

```bash
cd ai_content_factory

# Install dependencies
pip install -r requirements.txt

# For PDF support (optional)
pip install pymupdf
```

### Basic Usage

```bash
# Using Ollama (local, no API limits)
ollama serve  # Start Ollama in another terminal
ollama pull llama3.2

python main.py --topic "AI in Healthcare"
```

### With Gemini API

```bash
export GEMINI_API_KEY="your-api-key"
# Edit config/agent_config.yaml: provider: "gemini"

python main.py --topic "AI in Healthcare"
```

---

## �️ ATTIC Interactive CLI Mode

**ATTIC** (Adaptive Tool-driven Intelligent Content Orchestrator) provides an OpenCode-style interactive terminal experience.

### Launch ATTIC

```bash
python run_attic.py
```

### ATTIC Start Screen

```
     █████╗ ████████╗████████╗██╗ ██████╗
    ██╔══██╗╚══██╔══╝╚══██╔══╝██║██╔════╝
    ███████║   ██║      ██║   ██║██║     
    ██╔══██║   ██║      ██║   ██║██║     
    ██║  ██║   ██║      ██║   ██║╚██████╗
    ╚═╝  ╚═╝   ╚═╝      ╚═╝   ╚═╝ ╚═════╝

    Adaptive Tool-driven Intelligent Content Orchestrator

    Ask anything… "summarize AI policy from rss"

attic >
```

### Natural Language Prompts

Just type naturally—no flags required! ATTIC auto-detects intent and data sources:

```
attic > summarize latest ai chip news
attic > analyze this csv sales.csv
attic > research climate policy from rss
attic > write report from pdf market.pdf
attic > explain quantum computing
```

### Built-in Commands

| Command | Description |
|---------|-------------|
| `help` | Show help information |
| `history` | Show command history |
| `repeat` | Repeat the last command |
| `stats` | Show session statistics |
| `config` | Show current configuration |
| `clear` | Clear the screen |
| `exit` | Exit ATTIC |

### Keyboard Shortcuts

| Key | Action |
|-----|--------|
| `↑/↓` | Navigate command history |
| `Ctrl+C` | Cancel current operation |
| `Ctrl+D` | Exit ATTIC |

### Live Execution View

During pipeline execution, ATTIC shows real-time progress:

```
╭─ 🚀 Pipeline Execution ─────────────────────╮
│ ● Live Data Agent    ████████████████ 100% │
│ ● Research Agent     ████████████████ 100% │
│ ● Writer Agent       ████████░░░░░░░░  50% │
│ ○ Fact Check Agent   ░░░░░░░░░░░░░░░░   0% │
│ ○ Optimizer Agent    ░░░░░░░░░░░░░░░░   0% │
╰─────────────────────────────────────────────╯
```

---

## �📡 Live Data Agent Mode

Ingest real external data sources before research begins.

### Supported Sources

| Type | Description | Example |
|------|-------------|---------|
| **RSS** | News feeds, blogs | `--data-source-url "https://news.google.com/rss"` |
| **API** | REST JSON endpoints | `--api-endpoint "https://api.example.com/data"` |
| **PDF** | Documents (local/URL) | `--data-file report.pdf` |
| **CSV** | Datasets | `--data-file sales.csv` |

### Usage Examples

```bash
# RSS Feed
python main.py --topic "Climate Policy" --live-data \
  --data-source-url "https://news.google.com/rss/search?q=climate"

# CSV Dataset
python main.py --topic "Sales Analysis" --live-data \
  --data-file data/sales_2024.csv

# API Endpoint
python main.py --topic "Weather Report" --live-data \
  --api-endpoint "https://api.weather.gov/gridpoints/OKX/35,37/forecast"

# PDF Document
python main.py --topic "Research Summary" --live-data \
  --data-file papers/research.pdf
```

### How It Works

1. **LiveDataAgent** runs first when `--live-data` is enabled
2. Fetches data from all configured sources
3. Normalizes content into documents
4. Stores in ChromaDB vector store
5. **ResearchAgent** retrieves relevant context
6. Pipeline continues with enriched knowledge

---

## 🧠 Long-Term Learning Memory

The system learns from each run and improves over time.

### Components

| Component | Purpose | Storage |
|-----------|---------|---------|
| **RunHistoryStore** | Tracks every pipeline execution | SQLite |
| **StrategyMemory** | Remembers successful strategies | SQLite |
| **PerformanceTracker** | Agent metrics and failure patterns | SQLite |

### What Gets Learned

- ✅ Which prompts produce best results
- ✅ Which tools succeed for which topics
- ✅ Optimal agent execution order
- ✅ Topic category patterns
- ✅ Performance trends over time

### View Learning Statistics

```bash
python main.py --show-stats
```

### Disable Learning

```bash
python main.py --topic "Quick Test" --no-learning
```

---

## 📁 Project Structure

```
ai_content_factory/
├── agents/                    # Agent implementations
│   ├── __init__.py
│   ├── base_agent.py         # Abstract base agent
│   ├── research_agent.py     # Research specialist
│   ├── writer_agent.py       # Content writer
│   ├── factcheck_agent.py    # Fact checker
│   ├── optimizer_agent.py    # Content optimizer
│   └── live_data_agent.py    # Live data ingestion
│
├── orchestrator/              # Pipeline orchestration
│   ├── __init__.py
│   ├── pipeline.py           # Main pipeline
│   └── agent_manager.py      # Agent execution
│
├── tools/                     # Tool implementations
│   ├── __init__.py
│   ├── llm_client.py         # Gemini/Ollama clients
│   ├── search_tool.py        # Abstract/Mock search
│   ├── ddg_search.py         # Real DuckDuckGo search
│   ├── vector_store.py       # ChromaDB wrapper
│   ├── evaluator.py          # Content scoring
│   ├── live_data_tools.py    # Unified data interface
│   ├── rss_reader.py         # RSS/Atom parser
│   ├── api_client.py         # REST API client
│   ├── pdf_loader.py         # PDF extraction
│   └── csv_loader.py         # CSV loading
│
├── memory/                    # Learning system
│   ├── __init__.py
│   ├── run_history_store.py  # Execution history
│   ├── strategy_memory.py    # Strategy learning
│   └── performance_tracker.py # Metrics tracking
│
├── attic_cli/                 # ATTIC Interactive CLI
│   ├── __init__.py
│   ├── banner.py             # ASCII art banner
│   ├── command_parser.py     # Natural language parsing
│   ├── rich_layout.py        # UI components
│   ├── session_state.py      # Session management
│   └── interactive_shell.py  # Main shell
│
├── config/
│   └── agent_config.yaml     # Configuration
│
├── data/                      # SQLite databases
│   ├── run_history.db
│   ├── strategy_memory.db
│   └── performance.db
│
├── runs/                      # Run output logs
│   └── run_<timestamp>/
│       ├── config.json
│       ├── strategy_used.json
│       ├── evaluation.json
│       ├── agent_outputs/
│       └── live_data_raw/
│
├── main.py                    # Flag-based CLI
├── run_attic.py               # ATTIC interactive CLI
├── requirements.txt
└── README.md
```

---

## ⚙️ Configuration

Edit `config/agent_config.yaml`:

```yaml
# LLM Provider
llm:
  provider: "ollama"  # or "gemini"
  model: "llama3.2"
  temperature: 0.7
  max_tokens: 2000

# Feature Flags
live_data_enabled: false      # Default for all runs
learning_memory_enabled: true # Enable learning

# Tool Permissions
tool_permissions:
  live_data_agent:
    - rss
    - api
    - pdf
    - csv
```

---

## 📊 CLI Reference

```
usage: main.py [-h] --topic TOPIC [--live-data] [--data-source-url URL]
               [--data-file FILE] [--api-endpoint URL] [--no-learning]
               [--output FILE] [--verbose] [--config PATH] [--show-stats]

Options:
  --topic TOPIC           Topic to generate content about (required)
  --live-data             Enable live data ingestion
  --data-source-url URL   URL for live data (RSS, API, PDF)
  --data-file FILE        Local file path (CSV, PDF)
  --api-endpoint URL      REST API endpoint
  --no-learning           Disable learning memory
  --output, -o FILE       Save output to file
  --verbose, -v           Enable debug logging
  --config PATH           Custom config file path
  --show-stats            Show learning statistics
```

---

## 🔧 Development

### Adding New Agents

1. Create `agents/new_agent.py` extending `BaseAgent`
2. Implement `execute()` method
3. Register in `pipeline.py`

### Adding New Data Sources

1. Create `tools/new_loader.py`
2. Add to `LiveDataTools` class
3. Update `DataSourceType` enum

---

## 📈 Performance

Typical execution times (with Ollama llama3.2):

| Mode | Agents | Time |
|------|--------|------|
| Standard | 4 | ~2-3 min |
| Live Data | 5 | ~3-4 min |

Quality scores typically range 75-95 depending on topic complexity.

---

## 📝 License

MIT License - See LICENSE file

---

## 🙏 Acknowledgments

Built with:
- [Ollama](https://ollama.ai/) - Local LLM inference
- [Google Gemini](https://ai.google.dev/) - Cloud LLM API
- [ChromaDB](https://www.trychroma.com/) - Vector database
- [Rich](https://rich.readthedocs.io/) - Terminal UI
