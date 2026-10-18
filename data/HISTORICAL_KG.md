# Historical Knowledge Graph

This directory contains the validated precomputed Historical Knowledge Graph
used by Knowdit audits. Keeping the database in the repository avoids rerunning
the long and expensive C4 learning pipeline after a fresh clone.

## Baseline

- Database: `historical.db`
- Learned projects: 66
- Audit findings: 1,887
- Semantic rows: 5,255
- Globally linked findings: 1,887/1,887
- Cross-project links: 5,563
- Validation status: passed

Validate after cloning with:

```bash
./engine/target/release/knowdit \
  learn --database-url "sqlite:///absolute/path/to/data/historical.db" \
  validate-db
```

The database is a binary artifact and should be treated as a versioned
baseline. SQLite WAL, SHM, and journal files are runtime-only and ignored.
