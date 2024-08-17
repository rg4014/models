import base64
import hashlib
import hmac
import json
import os
from collections.abc import Mapping
from datetime import datetime

import requests
from loguru import logger

from requests_toolbelt import MultipartEncoder


class FeishuBotTemplate(object):
    @property
    def text_format(self):
        return dict(
            bold=lambda x: f"**{x}**",
            italic=lambda x: f"**{x}**",
            delete=lambda x: f"~~{x}~~",
            at=lambda x: f"<at id={x}></at>",
        )

    @property
    def FEISHU_IMG_URL(self):
        return (
            os.getenv("FEISHU_IMG_URL")
            or "https://open.feishu.cn/open-apis/im/v1/images"
        )

    @property
    def FEISHU_TENANT_ACCESS_TOKEN(self):
        resp = requests.post(
            "https://open.feishu.cn/open-apis/auth/v3/tenant_access_token/internal",
            data={
                "app_id": os.getenv("FEISHUBOT_APPID"),
                "app_secret": os.getenv("FEISHUBOT_APPSECRET"),
            },
        )
        assert resp.status_code == 200, resp.content
        return json.loads(resp.content)["tenant_access_token"]

    @property
    def FEISHU_REPORT_BOT(self):
        return dict(
            webhook=os.getenv("FEISHU_REPORT_BOT_WEBHOOK"),
            secret=os.getenv("FEISHU_REPORT_BOT_SECRET"),
        )

    def _dict_to_plain_text(self, data):
        def _value_to_plain_text(x):

            if x is None:
                return ""
            elif isinstance(x, list):
                return "\n".join([str(_) for _ in x])
            elif isinstance(x, Mapping):
                return "\n".join([str(k) + ":" + str(v) for k, v in x.items()])
            else:
                return str(x)

        res = []
        for key, value in data.items():
            res.extend([self.text_format["bold"](key), _value_to_plain_text(value)])
        return "\n".join(res)

    def _get_two_col_text(self, data1, data2):
        element = {
            "tag": "div",
            "fields": [
                {
                    "is_short": True,
                    "text": {
                        "tag": "lark_md",
                        "content": self._dict_to_plain_text(data1),
                    },
                },
                {
                    "is_short": True,
                    "text": {
                        "tag": "lark_md",
                        "content": self._dict_to_plain_text(data2),
                    },
                },
            ],
        }
        return element

    def _get_text(
        self,
        data,
    ):
        return {"tag": "markdown", "content": self._dict_to_plain_text(data)}

    def _get_img(self, name, img_id, comment):
        return {
            "tag": "img",
            "title": {"tag": "lark_md", "content": self.text_format["bold"](name)},
            "img_key": img_id,
            "alt": {"tag": "plain_text", "content": comment},
        }

    def uploadImage(
        self,
        img_path=None,
    ):

        form = {
            "image_type": "message",
            "image": (open(img_path, "rb")),
        }
        multi_form = MultipartEncoder(form)
        headers = {
            "Authorization": f"Bearer {self.FEISHU_TENANT_ACCESS_TOKEN}",
        }
        headers["Content-Type"] = multi_form.content_type
        resp = requests.request(
            "POST", self.FEISHU_IMG_URL, headers=headers, data=multi_form
        )

        assert resp.status_code == 200, resp.content
        return json.loads(resp.content)

    def generateFeishuBotSecret(self, timestamp, secret):
        string_to_sign = f"{timestamp}\n{secret}"
        hmac_code = hmac.new(
            string_to_sign.encode("utf-8"), digestmod=hashlib.sha256
        ).digest()
        return base64.b64encode(hmac_code).decode("utf-8")

    def sendCard(self, data=None):

        timestamp = timestamp = int(round(datetime.now().timestamp()))
        sign = self.generateFeishuBotSecret(timestamp, self.FEISHU_REPORT_BOT["secret"])
        resp = requests.post(
            self.FEISHU_REPORT_BOT["webhook"],
            json={
                "msg_type": "interactive",
                "timestamp": timestamp,
                "sign": sign,
                "card": json.dumps(data),
            },
        )
        assert resp.status_code == 200, resp.content
        print(resp.content)

    def get_feishu_card_template(self, name="🥳特征平台数据同步脚本运行报告"):
        return {
            "config": {"wide_screen_mode": True},
            "header": {
                "template": "turquoise",
                "title": {"content": name, "tag": "plain_text"},
            },
            "elements": [],
        }


# def get_feishu_card(feishu, dtype_str, images_keys: list):
#     data = {
#         "config": {"wide_screen_mode": True},
#         "header": {
#             "template": "turquoise",
#             "title": {"content": "🥳特征平台数据同步脚本运行报告", "tag": "plain_text"},
#         },
#         "elements": [
#             feishu._get_two_col_text(
#                 data1={"名称": "feature_group", "数据源": "ml_stage.device_model_mapping",},
#                 data2={"创建时间⏱️": "2022-01-10", "结束时间⏱️": "2022-01-10",},
#             ),
#             feishu._get_text(data={"请求体": "dafsdf"}),
#             {"tag": "hr"},
#             feishu._get_two_col_text(
#                 data1={"特征数量🦊": 12,}, data2={"样本数量": 10000, "Unique Key数量": 9989,}
#             ),
#             feishu._get_two_col_text(data1=["acc_id"], data2=["string"]),
#             {"tag": "hr"},
#             feishu._get_img("特征缺失", images_keys[0], "test"),
#             feishu._get_img("特征分布", images_keys[1], "test"),
#         ],
#     }
#     return data


# if __name__ == "__main__":
#     from prettytable import PrettyTable, PLAIN_COLUMNS
#     import pendulum

#     print(pendulum.now())
#     json.dumps(dict(a=pendulum.now().to_datetime_string()))

#     tenant_access_token = "t-g1048tbd5RBJJSFJHQJGEC7VWBL2ABTWQVS2I7JF"
#     secret = "4Qwp3L3W1Vz9Pt4Jkvm3Md"
#     webhook = "https://open.feishu.cn/open-apis/bot/v2/hook/8aaa5604-6bca-42c4-9417-0803399c2285"

#     os.environ["FEISHU_IMG_URL"] = "https://open.feishu.cn/open-apis/im/v1/images"
#     os.environ["FEISHU_TENANT_ACCESS_TOKEN"] = tenant_access_token
#     os.environ["FEISHU_REPORT_BOT_WEBHOOK"] = webhook
#     os.environ["FEISHU_REPORT_BOT_SECRET"] = secret

#     feishu = FeishuBotTemplate()

#     img_path = "C:\\Users\\caoxu\\Downloads\\v2-7e25ed49687b8c3c11e5d99c6c499256_b.jpg"

#     # resp = feishu.uploadImage(img_path)
#     image_key = "img_v2_c90c13e7-006f-4db1-82f8-86d3a82f289g"

#     dtype_str = PrettyTable(["字段", "类型"])
#     dtype_str.add_row(["acc_id", "string"])
#     dtype_str.add_row(["asadfasdfewcc_id", "string"])
#     dtype_str.set_style(PLAIN_COLUMNS)
#     dtype_str.border = True
#     dtype_str.padding_width = 4
#     print(dtype_str.get_string())

#     dtype_str = """• 适应力：其适应能力强，不会因为环境的改变而改变。\n• 性格：大胆好奇，但很温柔，不会发脾气，更不会乱吵闹。"""

#     data = get_feishu_card(
#         feishu, dtype_str=dtype_str, images_keys=[image_key, image_key],
#     )
#     # print(json.dumps(data))

#     feishu.sendCard(data=data,)
