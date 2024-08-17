import grpc
from idl.featplat.featplat_pb2 import FeatureRequest
from idl.featplat.featplat_pb2_grpc import FeaturePlatServerStub
from typing import Dict, List


def get_feature(stub, name, key, config):
    return stub.Get(FeatureRequest(name=name, key=key, config=config))


def make_config(column_name: str, feature_configs: List[Dict]):
    config_str = ""
    config_str += f"column_name: {column_name}"
    for featture_config in feature_configs:
        config_str += "\n"
        config_str += "; ".join([f"{k}={v}" for k, v in featture_config.items()])
    return config_str


def grpc_test():
    host = "featplat-online-dev.tensorpipes"
    
    port = 13661
    # host, port = "localhost", 13723
    channel = grpc.insecure_channel(f"{host}:{port}")
    stub = FeaturePlatServerStub(channel)
    key = {
        "app_id": "com.lilithgames.rok.gpkr",
        "account": "4302654",
    }

    config = make_config("source:dappltv2", [
        {
            "feature_name": "f_country",
            "depend": "country",
            "method": "Direct"
        },
        {
            "feature_name": "f_game_id",
            "depend": "game_id",
            "method": "Direct"
        },
        {
            "feature_name": "f_login_last",
            "depend": "login_last",
            "method": "Direct"
        },
        {
            "feature_name": "f_ltv",
            "depend": "ltv",
            "method": "Direct"
        },
        {
            "feature_name": "f_device_type",
            "depend": "device_type",
            "method": "Hash"
        },
    ])
    feature = get_feature(stub, "test", key, config)
    print(feature)
    # key = {}
    # feature = get_feature(stub, "test", key, config)


if __name__ == '__main__':
    grpc_test()