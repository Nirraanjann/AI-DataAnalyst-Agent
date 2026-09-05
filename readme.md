# AI Data Analyst Agent

An agentic AI system that answers natural-language business questions (e.g. *"Why did revenue decline in Q2?"*) by orchestrating SQL, Python/Pandas analysis, and RAG over a real e-commerce dataset — served via FastAPI, with LangGraph coordinating tool selection and verification.

This is a phase-gated build. **Phase 1 (Data Foundation) is complete.** Phases 2–10 (deterministic tools, LLM-to-SQL, the agent itself, RAG, the API, Docker/CI/CD, cloud deployment) are documented in the project roadmap and not yet built.

## Why this project exists

The goal is to close the gap between research/ML skills and production software engineering: backend development, SQL depth, containerization, CI/CD, and cloud deployment — while producing a real, evaluable AI application rather than a notebook or chatbot demo.

## Dataset

**Olist Brazilian E-Commerce Public Dataset** — ~100K real orders placed on the Olist marketplace between 2016 and 2018, across 9 source CSVs.

- **Source:** [Kaggle — Olist Brazilian E-Commerce Public Dataset](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce)
- **License:** CC BY-NC-SA 4.0
- **Scale:** ~99,441 orders, ~112,650 order line items, ~99,441 customers, ~3,095 sellers, ~32,951 products

Of the 9 source files, **7 are loaded** into this project's database. `geolocation` (a many-to-many zip-code lookup table, not an analytical entity) and `order_reviews` (deferred to a later RAG phase) are intentionally excluded from Phase 1.

## Database schema

PostgreSQL, 7 tables, FK-constrained:

```
category_translation ──┐
                        ├──> products ──┐
sellers ────────────────────────────────┤
                                         ├──> order_items ──> orders ──> customers
                                         │                       │
                                         └───────────────────────┘
                                                                  └──> order_payments
```

Full DDL: [`sql/schema.sql`](sql/schema.sql)

**Key schema decisions:**
- `order_items` and `order_payments` use **composite primary keys** (`order_id` + a sequence column), because a single order can have multiple line items and multiple split payments.
- `customers.customer_unique_id` (a real person) is distinct from `customers.customer_id` (order-scoped) — the same person can appear multiple times under different `customer_id`s. Any per-customer aggregation in this project uses `customer_unique_id`.
- Date columns (`order_purchase_timestamp`, etc.) are cast to real `TIMESTAMP` types at load time — the source CSVs store them as strings.
- `products.product_category_name` is a **nullable** FK: ~610 products in the source data have no category assigned. This is left as `NULL` rather than imputed, since it's a genuine data-quality characteristic of the source, not a loading error.

## Setup

### Prerequisites
- Python 3.11+
- PostgreSQL 16+ (a local install, or `docker run --name olist-pg -e POSTGRES_PASSWORD=<pw> -e POSTGRES_DB=ecommerce_analyst -p 5432:5432 -d postgres:16`)
- The 9 Olist CSVs downloaded from Kaggle into `data/raw/`

### Install

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

Copy `.env.example` to `.env` and fill in your database credentials:

```
DB_HOST=localhost
DB_PORT=5432
DB_NAME=ecommerce_analyst
DB_USER=postgres
DB_PASSWORD=<your_password>
```

### Reproduce the full pipeline

```bash
# 1. Create the schema
psql -h localhost -U postgres -d ecommerce_analyst -f sql/schema.sql

# 2. Load all 7 tables (idempotent-unsafe: drops+reloads via schema.sql first if re-running)
python scripts/load_data.py

# 3. Inspect the 10 analytical queries
#    (run interactively via psql or a SQL client — see sql/analytical_queries.sql)

# 4. Regenerate ground-truth results for the evaluation set
python scripts/generate_ground_truth.py > ground_truth_output.json
```

`load_data.py` reports source-vs-loaded row counts for every table and exits non-zero on any mismatch or failure, so a clean run is a reliable signal the database matches the source data.

## Analytical questions & evaluation set

Ten business questions were defined as the initial evaluation target for the eventual AI agent — see [`sql/analytical_queries.sql`](sql/analytical_queries.sql) for the SQL and [`evaluation_questions.json`](evaluation_questions.json) for documented ground truth, assumptions, and findings for each.

**Global revenue definition** (applies to every question): `revenue = SUM(order_items.price)`, restricted to `orders.order_status = 'delivered'`. Freight is excluded — it's a pass-through shipping cost, not merchandise revenue. This assumption is repeated in every query's comments and in `evaluation_questions.json` so it can't be silently forgotten in later phases.

One deviation from the original question set: **Q8 was changed from "profit margin by category" to "average order value by category,"** since the Olist dataset has no product cost data — profit margin isn't computable from what's actually available. Documented in `evaluation_questions.json`.

### Notable findings from the evaluation pass

- **Peak revenue month:** November 2017 (R$987,765), consistent with Brazil's Black Friday timing.
- **Q10 (revenue decline root cause):** revenue fell -12.4% from May to June 2018. This was confirmed as a genuine demand decline (not a delivery-status artifact — total orders placed also fell -10.3%), broad-based across ~10 unrelated categories, and concentrated ~85% in São Paulo and Rio de Janeiro. It corresponds to Brazil's nationwide truckers' strike (May 21 – June 1, 2018), which disrupted retail logistics nationwide and prompted São Paulo to declare a state of emergency. Full reasoning and caveats (not every category's decline shares this cause) are documented in `evaluation_questions.json`.

## Repository structure

```
ai-data-analyst/
├── sql/
│   ├── schema.sql                  # CREATE TABLE DDL, 7 tables
│   ├── analytical_queries.sql      # The 10 ground-truth business questions
│   ├── q10_diagnostics.sql         # Root-cause investigation queries for Q10
│   └── q10_diagnostics_step4.sql   # Seasonality check for Q10
├── scripts/
│   ├── load_data.py                # Loads all 7 CSVs into Postgres, verifies row counts
│   └── generate_ground_truth.py    # Re-runs ground-truth queries, outputs clean JSON
├── data/
│   └── raw/                        # Source Olist CSVs (not committed — see .gitignore)
├── evaluation_questions.json       # Ground truth, assumptions, and findings per question
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

## Data quality notes

- 2 product categories (`pc_gamer`, `portateis_cozinha_e_preparadores_de_alimentos`) exist in `products.csv` but were missing from the category-translation lookup file; these were backfilled into `category_translation` at load time so the FK constraint doesn't silently drop affected products.
- ~2,965 orders have no `order_delivered_customer_date` — confirmed to correlate entirely with non-`delivered` order statuses (canceled, in-flight), not a data error.
- `order_payments` has more rows (103,886) than `orders` (99,441) because some orders have split payments across multiple rows (`payment_sequential > 1`).
- Delivered-order data thins out after **August 2018** — this reflects the dataset's extraction cutoff (later orders hadn't reached "delivered" status yet at extraction time), not a real business trend. Any time-series analysis should treat the last 1–2 months with caution.

## Roadmap

| Phase | Status | Description |
|---|---|---|
| 1 | ✅ Done | Data foundation — schema, load, ground-truth queries |
| 2 | Next | Deterministic Python analyst tools (no LLM yet) |
| 3 |  | Basic LLM-to-SQL analyst |
| 4 |  | LangGraph agent |
| 5 |  | Verification and safety (SQL allowlisting, query limits) |
| 6 |  | Python analysis and visualization |
| 7 |  | RAG over supporting business documents |
| 8 |  | FastAPI production backend |
| 9 |  | Docker + testing + CI/CD |
| 10 |  | Cloud deployment + monitoring |

## License

Project code: for personal/portfolio use. Dataset: Olist Brazilian E-Commerce Public Dataset, CC BY-NC-SA 4.0 — see the [Kaggle page](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce) for full terms.