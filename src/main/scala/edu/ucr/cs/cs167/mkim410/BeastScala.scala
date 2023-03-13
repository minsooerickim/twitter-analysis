package edu.ucr.cs.cs167.mkim410

import edu.ucr.cs.bdlab.beast.geolite.{Feature, IFeature}
import org.apache.spark.SparkConf
import org.apache.spark.beast.SparkSQLRegistration
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{DataFrame, SaveMode, SparkSession}

import scala.collection.Map

/**
 * Scala examples for Beast
 */
object BeastScala {
  def main(args: Array[String]): Unit = {
    // Initialize Spark context

    val conf = new SparkConf().setAppName("Twitter Analysis")
    // Set Spark master to local if not already set
    if (!conf.contains("spark.master"))
      conf.setMaster("local[*]")

    val spark: SparkSession.Builder = SparkSession.builder().config(conf)

    val sparkSession: SparkSession = spark.getOrCreate()
    val sparkContext = sparkSession.sparkContext
    SparkSQLRegistration.registerUDT
    SparkSQLRegistration.registerUDF(sparkSession)

    val inputFile: String = args(0)
    try {
      // Import Beast features
      import edu.ucr.cs.bdlab.beast._
      val t1 = System.nanoTime()

      // Load the given input file using the json format.
      val tweetsDF = sparkSession.read.format("json")
        .option("sep", "\t")
        .option("inferSchema", "true")
        .option("header", "true")
        .load(inputFile)

      // used for validating Q1
      tweetsDF.show()
      tweetsDF.printSchema()

      // Keep only the following attributes {id, text, entities.hashtags.txt, user.description, retweet_count, reply_count, and quoted_status_id}
      // TODO: entities.hashtags is wrong
      val convertedDF: DataFrame = tweetsDF.selectExpr("id", "text", "explode(entities.hashtags)", "user.description", "retweet_count", "reply_count", "quoted_status_id")
      convertedDF.show()
      convertedDF.printSchema()

      convertedDF.write.mode(SaveMode.Overwrite).json("tweets_clean")

      val t2 = System.nanoTime()

      println(s"Operations on file '$inputFile' took ${(t2 - t1) * 1E-9} seconds")
    } finally {
      sparkSession.stop()
    }
  }
}