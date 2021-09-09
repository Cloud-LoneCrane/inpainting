"""
Author: CraneLone
email: wang-zhizhen@outlook.com

file: tools
date: 2021/8/31 0031 下午 11:17
desc: 
"""


class Dict(dict):
    __setattr__ = dict.__setitem__
    __getattr__ = dict.__getitem__


def dict_to_object(dictObj):
    if not isinstance(dictObj, dict):
        return dictObj
    inst = Dict()
    for k, v in dictObj.items():
        inst[k] = dict_to_object(v)
    return inst


if __name__ == '__main__':
    database = {
        "name": "18D_Block",
        "xcc": {
            "component": {
                    "core": [],
                    "platform": []
                },
        },
        "uefi": {
            "component": {
                "core": [],
                "platform": []
            },
        }
    }

    # 转换字典成为对象，可以用"."方式访问对象属性
    res = dict_to_object(database)
    print(res.name)
    print(res.xcc)
    print(res.xcc.component)
    print(res.xcc.component.core)
