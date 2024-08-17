import json
import os
from pydoc import describe
from typing import List, Union

import pendulum
import pyspark.sql.functions as func
from loguru import logger
from pyspark import SparkConf
from pyspark.sql import SparkSession

from plot_images import FeatPlatReportPlot


def to_hbase_array(x, column: str = "value", key: List[str] = None):
    _key = "\001".join(str(x[_]) or "" for _ in key)
    return [_key, column, x["feature"], str(x["value"])]


def feishu_bot_bot_message():
    from feishu_bot import FeishuBotTemplate

    logger.info("Start to report to feishu bot")

    feishu = FeishuBotTemplate()
    data = feishu.get_feishu_card_template()

    namespace, name = os.getenv("TARGET_NAMESPACE"), os.getenv("FEATURE_NAME")

    describe_image_key = FeatPlatReportPlot.plot_statistic(
        json.loads(os.getenv("STATIC_DATA")), json.loads(os.getenv("DATA_DTYPES"))
    )

    elements = [
        feishu._get_two_col_text(
            data1={
                "名称": os.getenv("FEATURE_NAME"),
                "数据源(HIVE)": os.getenv("FEATURE_TABLE"),
                "目标表(HBASE)": f"{ namespace }:{ name }",
            },
            data2={
                "创建时间⏱️": os.getenv("JOB_START_DATETIME"),
                "结束时间⏱️": os.getenv("JOB_END_DATETIME"),
            },
        ),
        feishu._get_text(data={"请求体": "dafsdf"}),
        {"tag": "hr"},
        feishu._get_two_col_text(
            data1={
                "特征数量🦊": os.getenv("DATA_DTYPES_LENGTH"),
            },
            data2={
                "样本数量": os.getenv("DATA_SIZE"),
                "Unique Key数量": os.getenv("DATA_UNIQUEKEY_SIZE"),
            },
        ),
        feishu._get_text(data={"数据类型": json.loads(os.getenv("DATA_DTYPES"))}),
        {"tag": "hr"},
        feishu._get_img(
            name="缺失值",
            img_id=FeatPlatReportPlot.plot_missing(
                json.loads(os.getenv("MISSING_DATA")), int(os.getenv("DATA_SIZE"))
            ),
            comment="缺失值图表",
        ),
        feishu._get_img(
            name="数据分布",
            img_id=describe_image_key,
            comment="数据分布图表",
        ),
    ]
    value_freq = json.loads(os.getenv("VALUE_FREQ"))
    value_freq_feat_name = os.getenv("VALUE_FREQ_FEAT_NAME")
    value_freq_feat_dtype = json.loads(os.getenv("VALUE_FREQ_FEAT_DTYPE"))
    for feat_name, feat_data, dtype in zip(
        value_freq_feat_name.split(","), value_freq, value_freq_feat_dtype
    ):
        elements.append(
            feishu._get_img(
                name=feat_name,
                img_id=FeatPlatReportPlot.plot_distribution(
                    feat_data, feat_name, dtype
                ),
                comment="特征分布图表",
            )
        )
    logger.info(json.loads(os.getenv("MISSING_DATA")))
    logger.info(json.loads(os.getenv("VALUE_FREQ")))

    data["elements"].extend(elements)
    logger.info(data)
    feishu.sendCard(data=data)


class SparkSync(object):
    def __init__(self, app_name: str = "default-featplat-sync"):
        self.app_name = app_name
        self.key_split = "\001"

    def start_spark_session(self):
        conf = SparkConf().setAppName(self.app_name)
        conf.set("spark.sql.execution.arrow.pyspark.enabled", "true")

        spark = SparkSession.builder.config(conf=conf).enableHiveSupport().getOrCreate()
        spark.sparkContext.setLogLevel("Error")

        return spark

    def get_qualifier(self, x0, x1):
        return "\001".join([x0, x1])

    @logger.catch
    def sync(self, query, namespace, name, key, col="value"):

        spark = self.start_spark_session()
        data = spark.sql(query)
        data.printSchema()

        cols = [_ for _ in data.dtypes if _[0] not in key]
        cols_size = len(cols)

        stack_str = ",".join(
            map(lambda x: f"'{self.get_qualifier(x[0],x[1])}', {x[0]}", cols)
        )

        res = (
            data.select(*[func.col(_).astype("string") for _ in data.columns])
            .selectExpr(*key, f"stack({cols_size}, {stack_str}) as (feature, value)")
            .where(func.col('value') != func.lit('None'))
            .rdd.map(lambda x: to_hbase_array(x, col, key))
            .map(lambda x: (x[0], x))
        )


        data.unpersist()

        hbase_host = os.getenv("HBASE")
        hbase_port = os.getenv("HBASE_PORT")

        hbase_url = f"{hbase_host}:{hbase_port}"
        target_table = f"{namespace}:{name}"

        logger.info(f"hbase url : {hbase_url}")
        logger.info(f"target table : {target_table}")

        write_conf = {
            "hbase.zookeeper.quorum": hbase_url,
            "hbase.mapred.outputtable": target_table,
            "mapreduce.outputformat.class": "org.apache.hadoop.hbase.mapreduce.TableOutputFormat",
            "mapreduce.job.output.key.class": "org.apache.hadoop.hbase.common.ImmutableBytesWritable",
            "mapreduce.job.output.value.class": "org.apache.hadoop.io.Writable",
        }

        res.saveAsNewAPIHadoopDataset(
            conf=write_conf,
            keyConverter="org.apache.spark.examples.pythonconverters.StringToImmutableBytesWritableConverter",
            valueConverter="org.apache.spark.examples.pythonconverters.StringListToPutConverter",
        )

        logger.info("Sync Process Is Done")

        os.environ["JOB_END_DATETIME"] = pendulum.now().to_datetime_string()
        os.environ["DATA_DTYPES"] = json.dumps({k: v for k, v in data.dtypes})
        os.environ["DATA_DTYPES_LENGTH"] = str(len(data.dtypes))
        os.environ["DATA_SIZE"] = str(data.count())
        os.environ["DATA_UNIQUEKEY_SIZE"] = str(
            data.select(os.getenv("FEATURE_KEY").split(",")).distinct().count()
        )

        missing_count = (
            data.select(
                [
                    func.count(
                        func.when(func.isnan(c) | func.col(c).isNull(), c)
                    ).alias(c)
                    for c in data.columns
                ]
            )
            .collect()[0]
            .asDict()
        )

        os.environ["MISSING_DATA"] = json.dumps(missing_count)

        data.unpersist()
        # feishu_bot_bot_message()
        spark.stop()

    def compile_query(self) -> str:

        features = os.getenv("FEATURE_LIST")
        table = os.getenv("FEATURE_TABLE")
        key = os.getenv("FEATURE_KEY")

        query = f"""
        select {key}, {features} from {table}
        where acc_create_time = '2022-08-01'
        and game_id = '10043'
        and observe = 7
        """

        logger.info(query)
        return query


def main():

    os.environ["JOB_START_DATETIME"] = pendulum.now().to_datetime_string()

    namespace = os.getenv("TARGET_NAMESPACE")
    name = os.getenv("FEATURE_NAME")
    col = os.getenv("TARGET_COLUMN")

    key = os.getenv("FEATURE_KEY").split(",")
    key.sort()

    logger.info(f"namespace : {namespace}")
    logger.info(f"table : {name}")
    logger.info(f"column : {col}")

    job = SparkSync(f"featplatsync-{name}-app")
    job.sync(job.compile_query(), namespace, name, key, col)


if __name__ == "__main__":

    for i in range(5):
        try:
            main()
            quit()
        except Exception as e:
            logger.info(str(e))
            # if i < 5:
            #     logger.info(str(e))
            # else:
            #     raise e
