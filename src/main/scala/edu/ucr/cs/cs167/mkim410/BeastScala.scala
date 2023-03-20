package edu.ucr.cs.cs167.mkim410

import edu.ucr.cs.bdlab.beast.geolite.{Feature, IFeature}
import org.apache.spark.SparkConf
import org.apache.spark.beast.SparkSQLRegistration
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.functions.{arrays_zip, col, collect_list, explode}
import org.apache.spark.sql.{DataFrame, SaveMode, SparkSession}
import org.apache.spark.sql.functions._
import scala.collection.mutable._
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

      // used for validation
      // tweetsDF.show()
      // tweetsDF.printSchema()

      // Keep only the following attributes {id, text, entities.hashtags.txt, user.description, retweet_count, reply_count, and quoted_status_id}
      val clean_tweets_df: DataFrame = tweetsDF.selectExpr("id", "text", "transform(entities.hashtags, x -> x.text) AS hashtags", "user.description AS user_description", "retweet_count", "reply_count", "quoted_status_id")

      // run a top-k SQL query to select the top 20 most frequent hashtags as follows.
      clean_tweets_df.createOrReplaceTempView("clean_tweets")
      val frequent_hashtags = sparkSession.sql(
        """
        SELECT hashtag, count(*) as count
        FROM (
          SELECT explode(hashtags) as hashtag
          FROM clean_tweets
        ) t
        GROUP BY hashtag
        ORDER BY count DESC
        LIMIT 20
      """)

      val keywords: Array[String] = frequent_hashtags.select("hashtag").rdd.map(row => row.getString(0)).collect()

      keywords.foreach(println)

      //  used for validation
      //  frequent_hashtags.show()
      //  clean_tweets_df.show()
      //  clean_tweets_df.printSchema()

      // Store the output in a new JSON file named tweets_clean
      clean_tweets_df.write.mode(SaveMode.Overwrite).json("tweets_clean")

      val t2 = System.nanoTime()

      println(s"Operations on file '$inputFile' took ${(t2 - t1) * 1E-9} seconds")

      //end task 1
      //TASK 2
      val t3 = System.nanoTime()
      //retrieve tweets topics
      clean_tweets_df.createOrReplaceTempView("tweets_clean")
      //convert keywords to an array separated with , so it can be used for array intersect in a query
      val topics: String = "'"+ keywords.mkString("','") + "'"

      //dataframe


      val topics_df: DataFrame = sparkSession.sql(
        s"""
        SELECT id, text,element_at(t1.tweet_topic,1), user_description, retweet_count, reply_count, quoted_status_id
        AS topic FROM ( SELECT *, array_intersect(hashtags, array($topics)) AS tweet_topic FROM tweets_clean) AS t1 WHERE size(tweet_topic) > 0;
         """)


      //write to json
      topics_df.write.json("tweets_topic.json")
      val t4 = System.nanoTime()

      println(s"Operations on file '$inputFile' took ${(t4 - t3) * 1E-9} seconds")
    } finally {
      sparkSession.stop()
    }
  }
}
