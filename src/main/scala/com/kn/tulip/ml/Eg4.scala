package com.kn.tulip.ml

import org.apache.spark.ml.regression.LinearRegression
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.tuning.{ParamGridBuilder, TrainValidationSplit}
import org.apache.spark.sql.SparkSession
/** 训练校验分类来优化模型
  * Created by wangcunxin on 2018/1/20.
  */
object Eg4 {
  def main(args: Array[String]): Unit = {
    val spark = SparkSession
      .builder()
      .master("local[2]")
      .getOrCreate()
    spark.sparkContext.setLogLevel("WARN")

    val data = spark.read.format("libsvm").load("file:/Users/wangcunxin/temp/input/sample_linear_regression_data.txt")
    val Array(training, test) = data.randomSplit(Array(0.9, 0.1), seed=12345)
    // 评估器
    val lr = new LinearRegression()
    // 评估器参数集合:正则参数0=L2、0.5、1=L1
    val paramGrid = new ParamGridBuilder().
      addGrid(lr.elasticNetParam, Array(0.0,0.5,1.0)).
      addGrid(lr.fitIntercept).
      addGrid(lr.regParam, Array(0.1,0.01)).
      build()

    val trainValidationSplit = new TrainValidationSplit().
      setEstimator(lr).
      setEstimatorParamMaps(paramGrid).
      setEvaluator(new RegressionEvaluator).
      setTrainRatio(0.8)

    val model = trainValidationSplit.fit(training)

    model.transform(test).select("features","label","prediction").show()
    spark.stop()
  }
}
