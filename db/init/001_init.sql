CREATE TABLE IF NOT EXISTS datasets (
  name TEXT PRIMARY KEY,
  description TEXT,
  source_uri TEXT,
  created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS molecules (
  dataset TEXT NOT NULL,
  molecule_id INTEGER NOT NULL,
  qm9_id INTEGER,
  smiles TEXT,
  n_atoms INTEGER,
  environment TEXT,
  source_path TEXT NOT NULL,
  source_row INTEGER NOT NULL,
  data_format TEXT NOT NULL,
  created_at TIMESTAMPTZ DEFAULT NOW(),
  PRIMARY KEY (dataset, molecule_id)
) PARTITION BY LIST (dataset);

CREATE TABLE IF NOT EXISTS molecules_qm9s PARTITION OF molecules
  FOR VALUES IN ('qm9s');

CREATE TABLE IF NOT EXISTS molecules_ext_val PARTITION OF molecules
  FOR VALUES IN ('ext_val');

CREATE TABLE IF NOT EXISTS molecules_ext_val_env PARTITION OF molecules
  FOR VALUES IN ('ext_val_env');

CREATE INDEX IF NOT EXISTS molecules_qm9s_smiles_idx ON molecules_qm9s (smiles);
CREATE INDEX IF NOT EXISTS molecules_ext_val_smiles_idx ON molecules_ext_val (smiles);
CREATE INDEX IF NOT EXISTS molecules_ext_val_env_smiles_idx ON molecules_ext_val_env (smiles);
