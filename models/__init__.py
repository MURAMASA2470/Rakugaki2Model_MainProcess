from models.gru_net import GRUNet
from models.res_gru_net import ResidualGRUNet

MODELS = (GRUNet, ResidualGRUNet)


def get_models():
    """Returns a tuple of sample models."""
    return MODELS


def load_model(name):
    """Creates and returns an instance of the model given its class name.
    The created model has a single placeholder node for feeding images.
    """
    # Find the model class from its name
    all_models = get_models()
    # print('モデル書き出し：', all_models)
    mdict = {model.__name__: model for model in all_models}
    if name not in mdict:
        print('Invalid model index. Options are:')
        # Display a list of valid model names
        for model in all_models:
            print('\t* {}'.format(model.__name__))
        return None
    # else:
    #     print('エラー？')
    #     print("入力されたモデルデータ名と一致しません")
    #     print("登録名一覧：", mdict)
    #     print("入力されたモデル名：", name)
    NetClass = mdict[name]
    # print("返り値：", NetClass)

    return NetClass
