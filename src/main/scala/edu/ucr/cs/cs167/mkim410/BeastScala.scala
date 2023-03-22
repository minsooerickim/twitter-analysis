package edu.ucr.cs.cs167.mkim410

import edu.ucr.cs.bdlab.beast.geolite.{Feature, IFeature}
import org.apache.spark.SparkConf
import org.apache.spark.ml.classification.{LogisticRegression, LogisticRegressionModel, MultilayerPerceptronClassifier}
import org.apache.spark.ml.evaluation.{BinaryClassificationEvaluator, MulticlassClassificationEvaluator}
import org.apache.spark.ml.feature.{HashingTF, StringIndexer, Tokenizer}
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.tuning.{ParamGridBuilder, TrainValidationSplit, TrainValidationSplitModel}
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.beast.SparkSQLRegistration
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.functions._
import org.apache.spark.sql.{DataFrame, Dataset, Row, SaveMode, SparkSession}

import scala.collection.mutable._
import scala.collection.{Map, mutable}

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
        FROM ( SELECT *, array_intersect(hashtags, array($topics)) AS tweet_topic FROM tweets_clean) AS t1 WHERE size(tweet_topic) > 0;
         """)


      //write to json
      topics_df.write.json("tweets_topic.json")
      // topics_df.show()
      val t4 = System.nanoTime()

      //END TASK 2
      //BEGIN TASK 3

      val tokenizer = new Tokenizer().setInputCol("text").setOutputCol("words")

      val hashingTF = new HashingTF().setInputCol("words").setOutputCol("features")

      val stringIndexer = new StringIndexer().setInputCol("element_at(tweet_topic, 1)").setOutputCol("label").setHandleInvalid("skip")

      val logisticRegression = new LogisticRegression()

      val pipeline = new Pipeline().setStages(Array(tokenizer, hashingTF, stringIndexer, logisticRegression))

      val Array(trainingData, testData) = topics_df.randomSplit(Array(0.7, 0.3))

      val logisticModel = pipeline.fit(trainingData)

      val predictions = logisticModel.transform(testData)

      // predictions.select("id", "text", "element_at(tweet_topic, 1)", "user_description", "label", "prediction").show(10)
      predictions.select("id", "text", "element_at(tweet_topic, 1)", "user_description", "label", "prediction")

      // Compute the number of true positives, false positives, and false negatives for each class
      val tp = (0 to 10).map(c => predictions.filter(col("label") === c && col("prediction") === c).count()).sum
      val fp = (0 to 10).map(c => predictions.filter(col("label") =!= c && col("prediction") === c).count()).sum
      val fn = (0 to 10).map(c => predictions.filter(col("label") === c && col("prediction") =!= c).count()).sum

      // Compute overall precision and recall
      val overallPrecision = tp.toDouble / (tp + fp)
      val overallRecall = tp.toDouble / (tp + fn)

      println(s"Overall Precision: $overallPrecision, Overall Recall: $overallRecall")

      println(s"Operations on file '$inputFile' took ${(t4 - t3) * 1E-9} seconds")
    } finally {
      sparkSession.stop()
    }
  }
}
