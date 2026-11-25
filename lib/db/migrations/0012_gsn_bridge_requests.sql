CREATE TABLE IF NOT EXISTS gsn_bridge_requests (id uuid PRIMARY KEY, created_at timestamptz NOT NULL DEFAULT now(), finished_at timestamptz);
CREATE INDEX IF NOT EXISTS gsn_bridge_requests_created ON gsn_bridge_requests (created_at);
