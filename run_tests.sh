#!/usr/bin/env bash
set -e

GREEN='\033[0;32m'
NC='\033[0m'

echo "================================================="
echo "        RUNNING actf-core PLATFORM TEST SUITE       "
echo "================================================="

# PYTHONDONTWRITEBYTECODE=1 & -o cache_dir=/tmp prevents pytest from 
# attempting to write cache files into :ro read-only volume mounts.

echo -e "\n${GREEN}[1/3] Running Layer 1: Airflow DAG Integrity Tests...${NC}"
docker exec -e PYTHONDONTWRITEBYTECODE=1 -it actf-core-airflow-webserver \
  python3 -m pytest -o cache_dir=/tmp/.pytest_cache /opt/airflow/tests/test_dag_integrity.py -v

echo -e "\n${GREEN}[2/3] Running Layer 2: Raw Ingestion Unit Tests...${NC}"
docker exec -e PYTHONDONTWRITEBYTECODE=1 -it actf-core-ray-head \
  python3 -m pytest -o cache_dir=/tmp/.pytest_cache /home/ray/workspace/1-raw-data-ingest/tests/ -v

echo -e "\n${GREEN}[3/3] Running Layer 2: Data Prep Unit Tests...${NC}"
docker exec -e PYTHONDONTWRITEBYTECODE=1 -it actf-core-ray-head \
  python3 -m pytest -o cache_dir=/tmp/.pytest_cache /home/ray/workspace/2-data-prep/tests/ -v

echo -e "\n${GREEN}================================================="
echo -e "      ALL TEST SUITES PASSED SUCCESSFULLY!       "
echo -e "=================================================${NC}\n"