from __future__ import annotations
from pyspark.sql import SparkSession
import os

def get_spark(app_name: str = "HybridRecommender"):
    os.environ.setdefault("JAVA_HOME", "/usr/lib/jvm/java-17-openjdk-amd64")
    builder = (SparkSession.builder
               .appName(app_name)
               .master("local[*]")
               .config("spark.driver.memory", "6g")
               .config("spark.sql.shuffle.partitions", "200")
               .config("spark.sql.adaptive.enabled", "true"))
    spark = builder.getOrCreate()
    return spark
