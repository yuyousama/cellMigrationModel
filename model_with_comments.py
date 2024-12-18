# -*- coding: utf-8 -*-
"""
@Time    : 2024/1/4 16:42
@Author  : Qingdoors
"""

import numpy as np
import pylab as plt
import gc

class Model:
    def __init__(self, Settings):
        self.nc0 = Settings['nc0']  # init number of integrin
        self.r0_on = Settings['r0_on']  # Binding rate, 1 s-1,, 0.001 ms-1
        self.r0_off = Settings['r0_off']  # Zero force dissociation rate, 0.1 s-1, 0.0001 ms-1
        self.fcr = Settings['fcr']  # Threshold Reinforcement force, 3 pN
        self.alpha = Settings['alpha']  # Integrin density increment rate, 0.2 pN-1, fitting parameter
        self.fb = Settings['fb']  # Characteristic breakage force, 2 pN
        self.delta_t = Settings['delta_t']  # time step, 5 ms
        self.vr = Settings['vr']  # retrograde flow velocity
        self.kc = Settings['kc']  # stiffness of clutch, 5 pN/nm
        self.ks = (Settings['ks'] * np.pi * 0.15)/(1+0.5)  # ECM stiffness
        self.r0 = Settings['r0']  # Initial spreading radius, 5 μm, or 5000 nm
        self.ri = Settings['ri']    # radius when steady
        self.km = Settings['km']  # effective stiffness of cytoskeleton, 0.1 pN nm-1
        self.eta_m = Settings['eta_m']  # effective viscosity of cytoskeleton, 1 pN s nm-1, or 1000 pN ms nm-1
        self.h = Settings['h']  # Thickness of lamellipodium, 200 nm
        self.vp = Settings['vp']  # poly speed of actin
        self.init_state = Settings['init_state']    # init state of substrate, hard? soft?
        self.stiffness_factor = Settings['stiffness_factor']    # how much stiffness change
        self.integrin_factor = Settings['integrin_factor']      # how many integrin change
        self.integrin_engage = Settings['integrin_engage']      # speed of integrin engage, ms-1

        self.statu = {
            'nc' : self.nc0,    # number of integrin
            'clutch_pb' : np.zeros(self.nc0)+(self.r0_on/(self.r0_on+self.r0_off)), # problity of clutch remain engaged
            'clutch_bind' : np.ones(self.nc0)>0, # 1 for binding integrin, 0 for unbind
            'clutch_force' : np.zeros(self.nc0), # force on clutch
            'clutch_x' : np.zeros(self.nc0), # displacement of clutch
            'xs': np.zeros(self.nc0),  # substrate displacement
            'r': self.ri,  # radius of cell
            'last_r': self.ri,  # last rasius of cell, for calculating fr
            'fr': 0.0,  # resistance force
            'fs': 0.0,  # force on substrate, shoule be eqaul with fr+fc
            'state' : self.init_state,   # state of substrate, hard? soft?
            'time': 0  # time
        }

    def update(self):
        # 1 update Pb and clutch statu
        if np.sum(self.statu['clutch_bind']) > 0:
            f_a = np.sum(self.statu['clutch_force']*self.statu['clutch_bind'])/np.sum(self.statu['clutch_bind'])
        else:
            f_a = 0
        if f_a > self.fcr:
            r_on = self.r0_on * (1 + self.alpha * (f_a-self.fcr))
        else:
            r_on = self.r0_on
        r_off = self.r0_off * np.exp(self.statu['clutch_force'] / self.fb)
        self.statu['clutch_pb'] = r_on/(r_on+r_off)
        self.statu['clutch_bind'] = np.random.rand(self.statu['nc']) < self.statu['clutch_pb']

        # 2 update integrin number
        new_nc = int(self.nc0 + self.statu['time'] * self.integrin_engage)
        if new_nc > self.statu['nc']:
            add = new_nc - self.statu['nc']
            self.statu['nc'] = new_nc
            self.statu['clutch_pb'] = np.pad(self.statu['clutch_pb'],(0,add),'constant', constant_values=(0,self.r0_on/(self.r0_on+self.r0_off)))
            self.statu['clutch_bind'] = np.pad(self.statu['clutch_bind'],(0,add),'constant', constant_values=(0,1))
            self.statu['clutch_force'] = np.pad(self.statu['clutch_force'],(0,add),'constant', constant_values=(0,0))
            self.statu['clutch_x'] = np.pad(self.statu['clutch_x'],(0,add),'constant', constant_values=(0,0))
            self.statu['xs'] = np.pad(self.statu['xs'],(0,add),'constant', constant_values=(0,0))

        # 3 update dispalcement and force
        self.statu['clutch_x'][self.statu['clutch_bind']==True] += self.vr * self.delta_t
        self.statu['clutch_x'][self.statu['clutch_bind']==False] = 0
        self.statu['xs'][self.statu['clutch_bind'] == False] = 0

        fc = (self.statu['clutch_x']-self.statu['xs'])*self.kc*self.statu['clutch_bind']
        fs = self.statu['xs']*self.ks*self.statu['clutch_bind']
        delta_xs = (fc - fs) / (self.kc + self.ks)
        self.statu['xs'] += delta_xs

        self.statu['clutch_force'] = self.kc * (self.statu['clutch_x']-self.statu['xs'])

        # 4 calculate the membrane/cyroskeleton strain and resistance force
        r = (self.statu['r'] - self.r0) / self.r0
        dr = (self.statu['r'] - self.statu['last_r']) / self.r0
        a = self.km * r
        b = (self.eta_m * dr) / self.delta_t
        self.statu['fr'] = self.h * (a + b)
        # print('a',a)
        # print('b',b)
        self.statu['fs'] = (self.statu['fr'] + np.sum(self.statu['clutch_force']))/np.sum(self.statu['clutch_bind'])

        # 5 update r
        vs = self.vp - self.vr
        self.statu['last_r'] = self.statu['r']
        self.statu['r'] += vs * self.delta_t

        # 6 update time
        self.statu['time'] += self.delta_t

    def change(self):
        if self.statu['state'] == 'hard':
            #soften
            self.ks *= (1 - self.stiffness_factor)
            self.statu['state'] = 'soft'
        else:
            #harden
            self.ks /= (1 - self.stiffness_factor)
            self.statu['state'] = 'hard'
        fc = (self.statu['clutch_x']-self.statu['xs'])*self.kc*self.statu['clutch_bind']
        fs = self.statu['xs']*self.ks*self.statu['clutch_bind']
        delta_xs = (fc - fs) / (self.kc + self.ks)
        self.statu['xs'] += delta_xs

        self.statu['clutch_force'] = self.kc * (self.statu['clutch_x']-self.statu['xs'])

if __name__ == '__main__':
    Settings = {
        'nc0': 75,  # number of integrin, 75?
        'r0_on': 0.001,  # Binding rate, 1 s-1,, 0.001 ms-1
        'r0_off': 0.0001,  # Zero force dissociation rate, 0.1 s-1, 0.0001 ms-1
        'fcr': 3.0,  # Threshold Reinforcement force, 3 pN
        'alpha': 0.2,  # Integrin density increment rate, 0.2 pN-1, fitting parameter
        'fb': 2.0,  # Characteristic breakage force, 2.0 pN
        'delta_t': 5,  # time step, 5 ms
        'vr': 0.12,  # retrograde flow velocity
        'kc': 5.0,  # stiffness of clutch, 5 pN/nm
        'ks': 2.2,  # ECM stiffness
        'r0': 5000,  # Initial spreading radius, 5 μm, or 5000 nm
        'ri': 20000,    # radius when steady
        'km': 0.1,  # effective stiffness of cytoskeleton, 0.1 pN nm-1
        'eta_m': 100000,  # effective viscosity of cytoskeleton, 1 pN s nm-1, or 1000 pN ms nm-1
        'h': 200.0,  # Thickness of lamellipodium, 200 nm
        'vp': 0.127,  # unloaded poly speed of actin
        'init_state': 'hard',   # init state of substrate
        'stiffness_factor': 0.40,   # how much stiffness change
        'integrin_factor': 0.40,    # how many integrin change
        'integrin_engage': 75*5e-8,       # speed of integrin engage, ms-1
    }

    for k in range(1000):
        n = k
        print(n)
        for change_factor in [20]:
            if change_factor == 20:
                Settings['vp'] = 0.13
                Settings['stiffness_factor'] = 0.28
                Settings['integrin_factor'] = 0.18
                time_limit = 121
            else:
                Settings['vp'] = 0.137
                Settings['stiffness_factor'] = 0.4
                Settings['integrin_factor'] = 0.4
                time_limit = 30
            f = open(f'./multi2/{str(n).zfill(4)}_factor_{change_factor}.csv', 'w', encoding='utf8')
            data = {}
            parameter_list = ['clutch_bind','clutch_force','clutch_x','xs','r','fs','time']

            for parameter in parameter_list:
                data[parameter] = []
                f.write(f'{change_factor}_{parameter},')
            f.write('\n')

            cell = Model(Settings)
            while True:
                statu = cell.statu
                t = statu['time']
                if t > time_limit*(1000 * 60):
                    break
                if t % (1000*60) == 0 and t != 0:
                    cell.change()
                if t % 1000 == 0:
                    for parameter in parameter_list:
                        data[parameter].append(np.average(statu[parameter]))
                    for parameter in parameter_list:
                        f.write(f'{np.average(statu[parameter])},')
                    f.write('\n')

                cell.update()

            TT = np.array(data['time'])/(1000*60)
            fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(19.2,10.8))
            for i, ax in enumerate(axes.flatten()):
                ax.plot(TT, data[parameter_list[i]])
                ax.set_title(f'MAX:{np.max(data[parameter_list[i]])}')
                ax.set_xlabel('time/min')
                ax.set_ylabel(parameter_list[i])
            plt.tight_layout()
            # plt.show()
            plt.savefig(f'./multi2/{str(n).zfill(4)}_factor_{change_factor}.png')
            f.close()
            plt.cla()
            plt.close("all")

            del cell,fig,axes
            gc.collect()