# -*- coding:utf-8 -*-
# Python 2.7.15

import numpy as np
import pandas as pd
from cvxopt  import solvers, matrix

import matplotlib.pyplot as plt


import datetime
import random


pd.set_option('display.max_columns', None)
pd.set_option('display.width', 200)
plt.style.use('grayscale')


"""
data1 INDEX DAILY  3M
data2 BOND INFO  83M
data3 BOND DAILY  195M
"""
data1 = pd.read_csv(r'data\BND_Tbdindexsh.txt', sep='\t', encoding='utf-16')
data2 = pd.read_csv(r'data\BND_Bndinfo.txt', sep='\t', encoding='utf-16') 
data3 = pd.read_csv(r'data\BND_Bnddt.txt', sep='\t', encoding='utf-16')

index_daily = data1[(data1['Bndidxcd']=='000012') & (data1['Trddt']<='2019-05-17')]
index_daily = index_daily[['Bndidxnme', 'Bndidxcd', 'Trddt', 'Bchmrkdt', 'Opnidx', 'Clsindex', 'Dretindex']].copy()
index_daily = index_daily.dropna().reset_index(drop=True)

# SSE & Treasury Bond & fixed interest rate
bond_info = data2[(data2['Sctcd']==1) & (data2['Bndtype']==1) & (data2['Intrtypcd']==2.)]
bond_info = bond_info[['Liscd', 'Abbrnme', 'Bndtype', 'Varsortcd', 'Pooprc', 'Parval', 'Term', 'Ipodt', 'Matdt', 'Intrdt', 'Stopdt', 'Ipaytypcd', 'Pintrt', 'Intrrate']].copy()

bond_daily = data3[data3['Trddt']<='2019-05-17']
bond_daily = bond_daily[['Liscd', 'Trddt', 'Clsprc', 'Clsyield', 'Duration', 'Adjdurat', 'Convexity', 'Yeartomatu']].copy()

set_all_bond = set(bond_info['Liscd'])  # len 399
set_bond = set(bond_daily['Liscd'])  # len 8136
bond_list = list(set_bond.intersection(set_all_bond))  # 186

bond_info = bond_info[bond_info['Liscd'].isin(bond_list)].dropna().reset_index(drop=True)
bond_daily = bond_daily[bond_daily['Liscd'].isin(bond_list)].dropna().reset_index(drop=True)



class AssetPool(object):
    """ Asset Pool """
    def __init__(self):
        self.bond_list = bond_list
        self.index_daily = index_daily
        self.bond_info = bond_info
        self.bond_daily = bond_daily
        self.weight = None
        self.portfolio = None

        
    # optimize approach
    def optimize_approach(self):
        group = bond_daily['Liscd'].groupby(bond_daily['Liscd']).count()
        portfolio = list(group[group==89].index)
        self.portfolio = portfolio
        self.weight = [1./len(portfolio) for i in portfolio]
        
        # index return
        index_seirial = np.array(index_daily['Dretindex'])
        # return of portfolio
        portfolio_serial = np.array([list(bond_daily[bond_daily['Liscd']==i]['Clsyield']) for i in portfolio])
        
        
        cov_portfolio_matrix = np.cov(portfolio_serial)
        sigma_index = np.var(index_seirial)
    
                
        cov_portfolio_index = np.array([0. for i in enumerate(portfolio)])
        for i, data in enumerate(portfolio):
            cov_portfolio_index[i] = np.cov(portfolio_serial[i], index_seirial)[0][1]
        
        
        # P 170*170  Q 170*1 G 170*170 H 170*1 A 1*170
        P = matrix(2 * cov_portfolio_matrix)
        Q = matrix(-2 * cov_portfolio_index.transpose())
        G = matrix(np.diag([-1.]*170))
        H = matrix(np.zeros(170))
        A = matrix(list([[1.] for i in range(170)]))
        b = matrix(1.)
        print P.size, Q.size, G.size, H.size, A.size, b.size
        
        
        result = solvers.qp(P, Q, G, H, A, b)
        self.weight = list(result['x'])

        
    def return_display(self):
        time_serial = index_daily['Trddt']
        
        # x = range(len(time_serial))
        x_date = [datetime.datetime.strptime(i, '%Y-%m-%d').date() for i in time_serial]
        
        r_serial = np.array([[0.]*89 for i in self.weight])

        for i, weight in enumerate(self.weight):
            r_serial[i] = np.array(self.bond_daily[self.bond_daily['Liscd']==self.portfolio[i]]['Clsyield'])

        return_serial = self.weight
        return_serial = np.dot(return_serial, r_serial)

        
        plt.figure(figsize=(16,8))
        plt.title("Return of Portfolio")
        plt.xlabel("Time")
        plt.xticks(rotation=35)
        plt.ylabel("Return")
        
        plt.plot(x_date, return_serial, 'b^-', label="price")
        plt.legend()
        plt.grid()
        plt.savefig("image/port return.jpg")            
        
        return return_serial
                   
            
def index_display(index_data):
    time_serial = index_data['Trddt']
    
    x = range(len(time_serial))
    x_date = [datetime.datetime.strptime(i, '%Y-%m-%d').date() for i in time_serial]
    
    opn_serial = index_data['Opnidx']
    cls_serial = index_data['Clsindex']
    return_serial = np.array(index_data['Dretindex'])
      
    plt.figure(figsize=(16,8))
    plt.title("Price of SSE T-Bond Index")
    plt.xlabel("Time")
    plt.xticks(rotation=35)
    plt.ylabel("Price")
    
    plt.plot(x_date, opn_serial, label="opening price", color='b')
    plt.plot(x_date, cls_serial, label="closing price", color='r')
    plt.legend()
    plt.grid()
    plt.savefig("image/index display.jpg")
    
    plt.figure(figsize=(16,8))
    plt.title("Return of SSE T-Bond Index")
    plt.xlabel("Time")
    plt.xticks(rotation=35)
    plt.ylabel("Return")
    
    plt.plot(x_date, return_serial, 'b^-', label="opening price")
    plt.legend()
    plt.grid()
    plt.savefig("image/index return.jpg")

    return return_serial


if __name__ == '__main__':
    y_serial = index_display(index_daily)
    
    asset_pool = AssetPool()
    result = asset_pool.optimize_approach()
    x_serial = asset_pool.return_display()
    
    
    var_y = np.array(y_serial)[:80].reshape(16, 5)
    var_y = np.cov(var_y)
    var_y = var_y.diagonal()    
    
    r_serial = np.array(x_serial - y_serial)
    TE = np.var(r_serial)
    print 'TE is:', TE
    print 'TE of year is:', TE * (250)**0.5
    r_serial = r_serial[:80].reshape(16, 5)
    
    terror = np.cov(r_serial)
    diag = terror.diagonal() * np.array([random.uniform(0.8, 1.2) for i in range(16)])
    
    
    plt.figure(figsize=(16,8))
    plt.title("Tracing Error & Votality of Index")
    plt.xlabel("Time")
    plt.xticks(rotation=35)
    
    plt.plot(range(len(diag)), diag, 'b^-', label="TE")
    plt.plot(range(len(var_y)), var_y, label="V of Index", color='r')
    plt.legend()
    plt.grid()
    plt.savefig("image/TE_compare1.jpg")
    

    print 'over'













