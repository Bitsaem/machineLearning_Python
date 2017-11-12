# -*- coding: utf-8 -*-
"""
Created on Sat Sep  9 15:43:15 2017

@author: Dongwuk
"""
import numpy as np

I_1 = [1, 1]
I_2 = [1, 0]
I_3 = [0, 1]
I_4 = [0, 0]
I = [I_1, I_2, I_3, I_4]
Target = [0, 1, 1, 0]
"""
XOR Initial input and (target) output setting 
"""

weight = [0.7, 0.7, -2, 0.7, 0.7]
theta = [0.5, 0.5]
"""
initial weight function
"""

alpha = 0.19
backpropagation_rate = 0.1


def hidden_result(weigtht, theta, num):
    hid_result = I[num][0] * weight[0] + I[num][1] * weight[1] - theta[0]

    if hid_result > 0:
        hid_result = 1
    else:
        hid_result = 0

    return hid_result


def final_result(weight, theta, num):
    hid_result = hidden_result(weight, theta, num)
    result = I[num][0] * weight[3] + I[num][1] * weight[4] + weight[2] * hid_result - theta[1]
    """
    i[0] 은 입력 1 , i[1] 은 입력 2 r
    """
    if result > 0:
        result = 1
    else:
        result = 0
    return result


def learning(weight, theta, num):
    result = final_result(weight, theta, num)
    hid_result = hidden_result(weight, theta, num)
    d_weight = alpha * (Target[num] - result)

    if d_weight == 0:
        return True, weight, theta

    if hid_result == 1:
        """
        hid_result 값에 따라 weight 피드백을 달리함
        """
        theta[0] = theta[0] - d_weight * backpropagation_rate
        theta[1] = theta[1] - d_weight

        if num == 0 or num == 3:
            weight[0] = weight[0] + d_weight * backpropagation_rate
            weight[1] = weight[1] + d_weight * backpropagation_rate
            weight[2] = weight[2] + d_weight
            weight[3] = weight[3] + d_weight
            weight[4] = weight[4] + d_weight

        elif num == 1:
            weight[0] = weight[0] + d_weight * backpropagation_rate
            weight[2] = weight[2] + d_weight
            weight[3] = weight[3] + d_weight

        elif num == 2:
            weight[1] = weight[1] + d_weight * backpropagation_rate
            weight[2] = weight[2] + d_weight
            weight[4] = weight[4] + d_weight

    if hid_result == 0:

        theta[1] = theta[1] - d_weight

        if num == 0 or num == 3:
            weight[3] = weight[3] + d_weight
            weight[4] = weight[4] + d_weight

        elif num == 1:
            weight[3] = weight[3] + d_weight

        elif num == 2:
            weight[4] = weight[4] + d_weight

    return False, weight, theta


if __name__ == "__main__":
    print('****Initial****')
    print('weight is :', weight)
    print('theta is :', theta)
    for i in range(1000):
        is_corrects = []
        for z in range(4):
            is_correct, weight, theta = learning(weight, theta, z)
            is_corrects.append(is_correct)
        print('****Iter {} ****'.format(i + 1))
        print('weight is :', weight)
        print('theta is :', theta)
        if all(is_corrects):
            print("Done at iteration {}!!".format(i + 1))
            break