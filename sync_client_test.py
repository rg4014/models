import grpc
from idl.featplat.featplat_pb2 import SourceDescriptor
from idl.featplat.featplat_pb2_grpc import FeatureSyncServerStub


def sync_source(stub, name, features, key, table, condition):
    s = SourceDescriptor(
        name=name,
        features=features,
        key=key,
        table=table,
        condition=condition,
    )
    resp = stub.SyncSource(s)
    return resp


def grpc_test():
    channel = grpc.insecure_channel("featplat-sync-dev.tensorpipes:13723")
    # channel = grpc.insecure_channel("localhost:13723")
    stub = FeatureSyncServerStub(channel)
    name = "dappltv2"
    features = [
        "country",
        "login_last",
        "ltv",
        "device_type",
        "observe",
        "game_id",
        "acc_create_time",
    ]
    key = ["account", "apg_app_id"]
    table = "ml_stage.dap_pltv_feature_transpkg_trino"
    condition = {"acc_create_time": "2022-03-03", "game_id": "10043"}
    resp = sync_source(stub, name, features, key, table, condition)
    print(resp)
    # key = []
    # ERROR: key is required
    # print(sync_source(stub, name, features, key, table, condition))


def http_test():
    pass


def test(type="grpc"):
    if type == "grpc":
        grpc_test()
    else:
        http_test()


if __name__ == "__main__":
    test("grpc")
