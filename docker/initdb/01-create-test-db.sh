#!/bin/sh
# Runs once, the first time the database volume is created.
# Creates a second database used only by the integration tests.
set -e
psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" <<-EOSQL
    CREATE DATABASE "${POSTGRES_TEST_DB:-fincomb_test}" OWNER "$POSTGRES_USER";
EOSQL
