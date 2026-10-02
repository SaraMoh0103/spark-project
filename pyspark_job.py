import sys
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

# cleaning data by removing rows with zero or negative values and calculating amount with tax
def clean_data(df):
    return(
    df
    .withColumn("amount", F.col("amount").cast("double"))
    .filter(F.col("amount")>0)
    .filter(F.col("name").isNotNull() & (F.trim(F.col("name")) != ""))
    .withColumn("amount_with_tax", F.col("amount") * 1.20)
    )
def main(csv_path):
    spark = SparkSession.builder.appName("Data Cleaning Job").getOrCreate()
    try:
        df = (
            spark.read
            .option("header", "true")
            .option("inferSchema", "true")
            .option("mode", "PERMISSIVE")
            .csv(csv_path)
        )
        cleaned_df = clean_data(df)
        cleaned_df.show(truncate=False)

        (
            cleaned_df.write
            .mode("overwrite")
            .option("header", "true")
            .csv("silver_data.csv")
        )
        print("Data cleaning and saving completed successfully.")
    finally:
        spark.stop()

if __name__ == "__main__":
    main(sys.argv[1])