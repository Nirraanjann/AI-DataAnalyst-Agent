-- scripts/create_readonly_role.sql
--
-- Run ONCE, as a superuser (e.g. postgres), against the
-- ecommerce_analyst database:
--   psql -U postgres -d ecommerce_analyst -f scripts/create_readonly_role.sql
--
-- Creates a read-only role for Phase 5's SQLQueryTool, so
-- LLM-generated SQL executes with SELECT-only privileges at the
-- database level -- defense-in-depth beneath the application-level
-- allowlist in app/tools/sql_guardrails.py.
--
-- CHANGE_ME: replace with a real password, then add these to .env:
--   DB_READONLY_USER=analyst_readonly
--   DB_READONLY_PASSWORD=<the password you set below>

CREATE ROLE analyst_readonly WITH LOGIN PASSWORD 'niranjan';
GRANT CONNECT ON DATABASE ecommerce_analyst TO analyst_readonly;
GRANT USAGE ON SCHEMA public TO analyst_readonly;
GRANT SELECT ON ALL TABLES IN SCHEMA public TO analyst_readonly;

-- Ensures the role can still SELECT from any table added later
-- (e.g. if the schema grows in a future phase) without rerunning this.
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT SELECT ON TABLES TO analyst_readonly;