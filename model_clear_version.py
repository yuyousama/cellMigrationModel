# -*- coding: utf-8 -*-
"""
@Time    : 2024/10/8 14:40
@Author  : Qingdoors
"""


import numpy as np
import pylab as plt
import gc

class Model:
    def __init__(self, Settings):
        self.nc0 = Settings['nc0']
        self.r0_on = Settings['r0_on']
        self.r0_off = Settings['r0_off']
        self.fcr = Settings['fcr']
        self.alpha = Settings['alpha']
        self.fb = Settings['fb']
        self.delta_t = Settings['delta_t']
        self.vr = Settings['vr']
        self.kc = Settings['kc']
        self.ks = (Settings['ks'] * np.pi * 0.15)/(1+0.5)
        self.r0 = Settings['r0']
        self.ri = Settings['ri']
        self.km = Settings['km']
        self.eta_m = Settings['eta_m']
        self.h = Settings['h']
        self.vp = Settings['vp']
        self.init_state = Settings['init_state']
        self.stiffness_factor = Settings['stiffness_factor']
        self.integrin_factor = Settings['integrin_factor']
        self.integrin_engage = Settings['integrin_engage']

        self.statu = {
            'nc' : self.nc0,
            'clutch_pb' : np.zeros(self.nc0)+(self.r0_on/(self.r0_on+self.r0_off)), 
            'clutch_bind' : np.ones(self.nc0)>0,
            'clutch_force' : np.zeros(self.nc0),
            'clutch_x' : np.zeros(self.nc0),
            'xs': np.zeros(self.nc0),
            'r': self.ri,
            'last_r': self.ri,
            'fr': 0.0,
            'fs': 0.0,
            'state' : self.init_state,
            'time': 0
        }

    def update(self):

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


        new_nc = int(self.nc0 + self.statu['time'] * self.integrin_engage)
        if new_nc > self.statu['nc']:
            add = new_nc - self.statu['nc']
            self.statu['nc'] = new_nc
            self.statu['clutch_pb'] = np.pad(self.statu['clutch_pb'],(0,add),'constant', constant_values=(0,self.r0_on/(self.r0_on+self.r0_off)))
            self.statu['clutch_bind'] = np.pad(self.statu['clutch_bind'],(0,add),'constant', constant_values=(0,1))
            self.statu['clutch_force'] = np.pad(self.statu['clutch_force'],(0,add),'constant', constant_values=(0,0))
            self.statu['clutch_x'] = np.pad(self.statu['clutch_x'],(0,add),'constant', constant_values=(0,0))
            self.statu['xs'] = np.pad(self.statu['xs'],(0,add),'constant', constant_values=(0,0))


        self.statu['clutch_x'][self.statu['clutch_bind']==True] += self.vr * self.delta_t
        self.statu['clutch_x'][self.statu['clutch_bind']==False] = 0
        self.statu['xs'][self.statu['clutch_bind'] == False] = 0

        fc = (self.statu['clutch_x']-self.statu['xs'])*self.kc*self.statu['clutch_bind']
        fs = self.statu['xs']*self.ks*self.statu['clutch_bind']
        delta_xs = (fc - fs) / (self.kc + self.ks)
        self.statu['xs'] += delta_xs

        self.statu['clutch_force'] = self.kc * (self.statu['clutch_x']-self.statu['xs'])


        r = (self.statu['r'] - self.r0) / self.r0
        dr = (self.statu['r'] - self.statu['last_r']) / self.r0
        a = self.km * r
        b = (self.eta_m * dr) / self.delta_t
        self.statu['fr'] = self.h * (a + b)

        self.statu['fs'] = (self.statu['fr'] + np.sum(self.statu['clutch_force']))/np.sum(self.statu['clutch_bind'])


        vs = self.vp - self.vr
        self.statu['last_r'] = self.statu['r']
        self.statu['r'] += vs * self.delta_t


        self.statu['time'] += self.delta_t

    def change(self):
        if self.statu['state'] == 'hard':
            self.ks *= (1 - self.stiffness_factor)
            self.statu['state'] = 'soft'
        else:
            self.ks /= (1 - self.stiffness_factor)
            self.statu['state'] = 'hard'
        fc = (self.statu['clutch_x']-self.statu['xs'])*self.kc*self.statu['clutch_bind']
        fs = self.statu['xs']*self.ks*self.statu['clutch_bind']
        delta_xs = (fc - fs) / (self.kc + self.ks)
        self.statu['xs'] += delta_xs

        self.statu['clutch_force'] = self.kc * (self.statu['clutch_x']-self.statu['xs'])

if __name__ == '__main__':
    Settings = {
        'nc0': 75,
        'r0_on': 0.001,
        'r0_off': 0.0001,
        'fcr': 3.0,
        'alpha': 0.2,
        'fb': 2.0,
        'delta_t': 5,
        'vr': 0.12,
        'kc': 5.0,
        'ks': 2.2,
        'r0': 5000,
        'ri': 20000,
        'km': 0.1,
        'eta_m': 100000,
        'h': 200.0,
        'vp': 0.127,
        'init_state': 'hard',
        'stiffness_factor': 0.40,
        'integrin_factor': 0.40,
        'integrin_engage': 75*5e-8,
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
            plt.savefig(f'./multi2/{str(n).zfill(4)}_factor_{change_factor}.png')
            f.close()
            plt.cla()
            plt.close("all")

            del cell,fig,axes
            gc.collect()