from pyspark.sql import SparkSession
import os 

input_folder = '../../data'

data_files = os.listdir(input_folder)

spark = SparkSession.builder.appName("FuturesData").getOrCreate()

sdf = spark.read.parquet(os.path.join(input_folder, data_files[0]))
