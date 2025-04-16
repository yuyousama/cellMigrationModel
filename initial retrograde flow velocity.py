#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2025/3/16 20:15
# @Author  : Qingdoors
# @File    : initial retrograde flow velocity.py
# @Software: PyCharm

import pylab as pl

if __name__ == "__main__":
    t = 0           #Time, min
    v = 0           #Normalized retrograde flow velocity, or (flow velocity / unloaded velocity)
    Fs = 1          #Normalized Force from substrate
    Fm = 1          #Normalized Force from Myosin
    delta_Fm =0.029525    #The increase rate of Fm, as a fitting parameter, arises from cyclic mechanical regulation that induces rapid pMyosin accumulation (Yang et al., Cell Stem Cell, 2025), thereby driving accelerated Fm augmentation.
    delta_t = 0.1   #delta t, min

    tt = []
    vv = []

    # Simulation cycle
    while t < 720:
        tt.append(t)
        vv.append(v)

        t += delta_t
        Fm += delta_Fm
        v = 1 -Fs/Fm

    # print (vv)

    # Draw pictures.
    # Since the retrograde flow velocity increases extremely rapidly, the maximum unloaded velocity value is directly adopted in the cell migration simulation to simplify the model.

    pl.plot(tt, vv)
    pl.show()
