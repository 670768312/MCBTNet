def initialize():
    global atten_weight
    global r_idx
    atten_weight = []
    r_idx = []


def set_value(value1,value2):
        # 定义一个全局变量
        r_idx.append(value1)
        atten_weight.append(value2)
def get_len( ):
    print("len of idx:", len(r_idx))
    print("len of atten:",len(atten_weight))
def get_value():
    return r_idx,atten_weight