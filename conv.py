weight = [0.02, 0.25, 0.42]
input = [11, 30, 25]
# 假定，在气压=11， 湿度等于30， 温度等于25的情况下，降雨概率为30%，也就是0.3
real_answer = 0.3

# 设置调整步数
step_amount = 0.01

# 卷积计算
def conv(f, g):
    result = 0
    for i in range(len(f)):
        result += f[i] * g[i]
    return result

# 让权重增加一定的值
def addWeight(w, num):
    result = []
    for x in w:
        result.append(x + num)
    return result

# 让权重减少一定的值
def subWeight(w, num):
    result = []
    for x in w:
        result.append(x - num)
    return result

step = 0
for i in range(1000):
    step += 1
    # 计算降雨概率
    predict = conv(input, weight)
    # 衡量误差
    error = (predict - real_answer) ** 2
    print(f"训练步数：{step}, 误差：{error}, 预测结果：{predict}")

    # 测试需要怎么调整权重
    # 计算调高权重后的预测结果
    upWeight = addWeight(weight, step_amount)
    upPredict = conv(input, upWeight)
    # 计算误差
    uperror = (upPredict - real_answer) ** 2

    # 计算降低权重后的预测结果
    downWeight = subWeight(weight, step_amount)
    downPredict = conv(input, downWeight)
    # 计算误差
    downerror = (downPredict - real_answer) ** 2

    if downerror < uperror:
        weight = downWeight
    else:
        weight = upWeight
