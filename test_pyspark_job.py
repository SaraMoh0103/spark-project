import os
import sys

os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable
import pytest
from pyspark.sql import SparkSession
from pyspark_job import clean_data


@pytest.fixture(scope="module")
def spark_local():
    spark = (
        SparkSession.builder
        .master("local[2]")
        .appName("test-customer-orders")
        .config("spark.ui.enabled", "false")
        .getOrCreate()
    )

    yield spark

    spark.stop()

COLUMNS = ["transaction_id", "date", "name", "amount"]

def make_df(spark,rows):
    return spark.createDataFrame(rows, COLUMNS)


def test_clean_data_keeps_valid_rows(spark_local):
    df = make_df(spark_local, [
        ("1001", "2026-09-01", "Alice", "150.00"),
        ("1004", "2026-09-04", "Omar", "75.50"),
    ])
    transformed_df = clean_data(df)
    results = transformed_df.collect()
    assert len(results) == 2

def test_clean_data_calculates_amount_with_tax(spark_local):
    df = make_df(spark_local, [("1001", "2026-09-01","Alice", "150.00")])
    row = clean_data(df).collect()[0]
    assert row["amount_with_tax"] == pytest.approx(180.0)

def test_clean_data_adds_amount_with_tax_column(spark_local):
    df = make_df(spark_local, [("1001", "2026-09-01", "Alice", "150.00")])
    assert clean_data(df).columns == COLUMNS + ["amount_with_tax"]


def test_clean_data_removes_zero_and_negative_values(spark_local):
    df = make_df(spark_local, [
        ("1001", "2026-09-01", "Alice", "150.00"),
        ("1002", "2026-09-02", "Bob", "-20.00"),
        ("1003", "2026-09-03", "Sara", "0"),
    ])

    transformed_df = clean_data(df)

    results = transformed_df.collect()

    assert len(results) == 1
    assert results[0]["transaction_id"] == "1001"
    assert results[0]["name"] == "Alice"
    assert results[0]["amount"] == 150.0

def test_clean_data_removes_null_name(spark_local):
    df = make_df(spark_local,[
        ("1001", "2026-09-01", None, "150.00"),
        ("1002", "2026-09-02", "Bob", "20.00"),
    ])
    transformed_df = clean_data(df)
    results = transformed_df.collect()
    assert len(results) == 1
    assert results[0]["transaction_id"] == "1002"

def test_clean_data_removes_blank_or_whitespace_name(spark_local):
    df = make_df(spark_local, [ 
        ("1001", "2026-09-01", "   ", "150.00"),
        ("1002", "2026-09-02", "", "150.00"),
        ("1003", "2026-09-03", "Sara", "150.00"),])
    transformed_df = clean_data(df)
    results = transformed_df.collect()
    assert len(results) == 1

def test_clean_data_removes_non_numeric_amount(spark_local):
    df = make_df(spark_local, [
        ("1001", "2026-09-01", "Alice", "abc"),
        ("1002", "2026-09-02", "Bob", "$99.99"),
        ("1003", "2026-09-03", "Sara", "10.00"),
    ])
    transformed_df = clean_data(df)
    results = transformed_df.collect()
    assert len(results) == 1
    assert results[0]["transaction_id"] == "1003"

def test_clean_data_keeps_smalled_positive_amount(spark_local):
    df = make_df(spark_local, [("1009", "2026-09-09", "Nour", "0.01")])
    assert clean_data(df).count() == 1

def test_clean_data_all_rows_invalid_returns_empty(spark_local):
    df = make_df(spark_local, [
        ("1001", "2026-09-01", "Alice", "-1"),
        ("1002", "2026-09-02", None, "10"),
        ("1003", "2026-09-03", "Sara", "0"),
    ])
    assert clean_data(df).count() == 0