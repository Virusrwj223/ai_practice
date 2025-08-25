PRAGMA foreign_keys=ON;

CREATE TABLE IF NOT EXISTS town (
  id    INTEGER PRIMARY KEY AUTOINCREMENT,
  name  TEXT NOT NULL UNIQUE
);

CREATE TABLE IF NOT EXISTS resale_transaction (
  id                     INTEGER PRIMARY KEY AUTOINCREMENT,
  month                  DATE NOT NULL,
  town_id                INTEGER NOT NULL REFERENCES town(id),
  block                  TEXT NOT NULL,
  street_name            TEXT NOT NULL,
  flat_type              TEXT NOT NULL CHECK (flat_type IN
                         ('1 ROOM','2 ROOM','3 ROOM','4 ROOM','5 ROOM','EXECUTIVE','MULTI-GENERATION')),
  flat_model             TEXT,
  storey_low             INTEGER NOT NULL,
  storey_high            INTEGER NOT NULL,
  floor_area_sqm         REAL NOT NULL,
  lease_commence_year    INTEGER,
  remaining_lease_months INTEGER,
  resale_price           REAL NOT NULL,
  source_file            TEXT,
  source_rownum          INTEGER,
  created_at             TEXT DEFAULT (datetime('now')),
  UNIQUE (month, town_id, block, street_name, flat_type, flat_model,
          storey_low, storey_high, floor_area_sqm, lease_commence_year, resale_price)
);

CREATE INDEX IF NOT EXISTS idx_resale_month ON resale_transaction(month);
CREATE INDEX IF NOT EXISTS idx_resale_town_flat ON resale_transaction(town_id, flat_type);
CREATE INDEX IF NOT EXISTS idx_resale_storey ON resale_transaction(storey_low, storey_high);
