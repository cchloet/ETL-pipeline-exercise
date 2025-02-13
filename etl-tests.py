import pytest
from pyspark.sql import SparkSession
from pyspark.sql.types import *
from pyspark.sql.functions import col
import ETLscript as etl
from datetime import datetime

@pytest.fixture(scope="session")
def spark():
    """Create a Spark session for testing."""
    return SparkSession.builder \
        .appName("ETLTestingSuite") \
        .master("local[*]") \
        .getOrCreate()

@pytest.fixture(scope="function")
def sample_df(spark):
    """Create a sample DataFrame for testing."""
    data = [
        (1, "2024-01-01", "C001", "M", 25, "Electronics", 2, 100.0, 200.0),
        (2, "2024-01-02", "C002", "F", 30, "Beauty", 1, 50.0, 50.0),
        (3, "2024-01-03", "C003", "M", 35, "Clothing", 3, 75.0, 225.0)
    ]
    return spark.createDataFrame(data, schema=etl.CONFIG["SCHEMA"])

class TestDuplicateDetection:
    """Test suite for duplicate detection functionality."""
    
    def test_contains_no_duplicates(self, sample_df):
        """Test that contains_duplicates correctly identifies no duplicates."""
        assert not etl.contains_duplicates(sample_df, "TransactionID")

    def test_contains_duplicates(self, spark):
        """Test that contains_duplicates correctly identifies duplicates."""
        data = [
            (1, "2024-01-01", "C001", "M", 25, "Electronics", 2, 100.0, 200.0),
            (1, "2024-01-02", "C002", "F", 30, "Beauty", 1, 50.0, 50.0)  # Duplicate TransactionID
        ]
        df = spark.createDataFrame(data, schema=etl.CONFIG["SCHEMA"])
        assert etl.contains_duplicates(df, "TransactionID")

###############################################################
##########################  1  ################################
#################### Missing Test Case ########################
###############################################################
class TestDateCleaning:
    """Test suite for date cleaning functionality."""
    ################################################
    ############ Add Test Case Here ################
    ################################################ 

    def test_clean_date_valid(self):
        """Test clean_date with valid date."""
        assert etl.clean_date("2024-01-01", "2024-01-02", "2024-01-03") == "2024-01-02"

    def test_clean_date_invalid_with_valid_next(self):
        """Test clean_date with invalid date but valid next date."""
        assert etl.clean_date(None, "invalid_date", "2024-01-03") == "2024-01-03"

    def test_clean_date_all_invalid(self):
        """Test clean_date with all invalid dates."""
        assert etl.clean_date(None, "invalid_date", None) == etl.CONFIG["DATE_DEFAULT"]

class TestAgeValidation:
    """Test suite for age validation functionality."""
    
    def test_validate_age_valid(self, spark):
        """Test validate_age with valid ages."""
        data = [(1, 25), (2, 0), (3, 120)]  # Valid ages
        df = spark.createDataFrame(data, ["id", "Age"])
        result = etl.validate_age(df)
        assert result.filter(col("Age").isNull()).count() == 0

    def test_validate_age_invalid(self, spark):
        """Test validate_age with invalid ages."""
        data = [(1, -1), (2, 121), (3, 30)]
        df = spark.createDataFrame(data, ["id", "Age"])
        result = etl.validate_age(df)
        assert result.filter(col("Age").isNull()).count() == 2

class TestCategoryValidation:
    """Test suite for category validation functionality."""
    
    def test_validate_categories_valid(self, sample_df):
        """Test validate_categories with valid categories."""
        result = etl.validate_categories(sample_df)
        assert result.filter(col("ProductCategory").isNull()).count() == 0

    def test_validate_categories_invalid(self, spark):
        """Test validate_categories with invalid categories."""
        data = [
            (1, "Electronics"),
            (2, "Invalid_Category"),
            (3, "Beauty")
        ]
        df = spark.createDataFrame(data, ["id", "ProductCategory"])
        result = etl.validate_categories(df)
        assert result.filter(col("ProductCategory").isNull()).count() == 1


###############################################################
##########################  2  ################################
#################### Missing Test Case ########################
###############################################################
class TestPriceValidation:
    """Test suite for price validation functionality."""
    ################################################
    ############ Add Test Case Here ################
    ################################################ 

    def test_validate_price_per_unit_correct(self):
        """Test validate_price_per_unit with correct prices."""
        row = (1, "2024-01-01", "C001", "M", 25, "Electronics", 2, 100.0, 200.0)
        result = etl.validate_price_per_unit(row)
        assert result[7] == 100.0  # Price should remain unchanged

    
    def test_validate_price_per_unit_null_handling(self):
        """Test validate_price_per_unit with null values."""
        row = (1, "2024-01-01", "C001", "M", 25, "Electronics", 2, None, 200.0)
        result = etl.validate_price_per_unit(row)
        assert result[7] == 100.0  # Price should be calculated from total/quantity



###############################################################
##########################  3  ################################
#################### Missing Test Case ########################
###############################################################
class TestNullValueCounting:
    """Test suite for null value counting functionality."""
    ################################################
    ############ Add Test Case Here ################
    ################################################ 


    def test_count_null_values_no_nulls(self, sample_df):
        """Test count_null_values_per_column with no null values."""
        result = etl.count_null_values_per_column(sample_df)
        assert all(count == 0 for count in result.values())

   

class TestETLPipeline:
    """Test suite for the main ETL pipeline."""
    
    def test_run_etl_pipeline_clean_data(self, spark, tmp_path):
        """Test run_etl_pipeline with clean data."""
        # Create a temporary CSV file with clean data
        test_data = [
            (1, "2024-01-01", "C001", "M", 25, "Electronics", 2, 100.0, 200.0),
            (2, "2024-01-02", "C002", "F", 30, "Beauty", 1, 50.0, 50.0)
        ]
        test_df = spark.createDataFrame(test_data, schema=etl.CONFIG["SCHEMA"])
        test_file = tmp_path / "test_data.csv"
        test_df.write.csv(str(test_file), header=True)
        
        # Run pipeline
        result_df = etl.run_etl_pipeline(spark, str(test_file))
        
        # Verify results
        assert result_df.count() == 2
        assert not etl.contains_duplicates(result_df, "TransactionID")
        assert etl.count_null_values_per_column(result_df)["TransactionID"] == 0