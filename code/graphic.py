import numpy as np
import matplotlib.pyplot as plt
import code as cd
import math as mt

def drawPolygon():
    dis_row = cd.getDisRow()
    plt.figure(figsize=(10, 4.8))
    plt.plot(dis_row.loc['x'], dis_row.loc['w'], 'r.-')
    plt.title('Полигон относительных частот')
    plt.xlabel('x (кол-во потраченных данных), Мб')
    plt.ylabel('ω (частость)')
    plt.xlim(-50, 900)
    plt.ylim(0, 0.3)
    plt.xticks(range(-50, 950, 50))
    plt.grid(True)
    plt.vlines(x=dis_row.loc['x'].to_numpy(), ymin=0, ymax=dis_row.loc['w'].to_numpy(dtype=float), color='r', linestyle='--')
    for i in range(dis_row.shape[1]):
        plt.text(x=dis_row.iat[0,i]-15, y=dis_row.iat[2,i]+0.005, s=str(dis_row.iat[2,i]), fontsize='small', fontweight='bold')
    plt.savefig(r'graphics\line.svg')
    plt.savefig(r'graphics\line.jpg')

def drawDistribFunc_dis():
    func_arr = cd.getDistribFunc_dis(cd.getDisRow())
    plt.figure(figsize=(10, 4.8))
    plt.title('Эмпирическая функция распределения')
    plt.xlabel('x, Мб')
    plt.ylabel('F(x)')
    plt.xlim(-50, 900)
    plt.ylim(-0.05, 1.05)
    plt.xticks(range(-50, 1000, 50))
    plt.yticks([i/100 for i in range(-5, 105, 5)])
    plt.grid(True)
    plt.hlines(y=np.array(func_arr[:, 0], dtype=float), xmin=func_arr[:, 2], xmax=func_arr[:, 1], color='r', linewidths=1)
    for i in range(func_arr.shape[0]):
        plt.text(x=func_arr[i, 2], y=float(func_arr[i, 0])+0.01, s=str(func_arr[i, 0]), fontsize='small')
    plt.savefig(r'graphics\distrib_func.svg')
    plt.savefig(r'graphics\distrib_func.jpg')

def drawDistribFunc_int():
    _, int_edges, _, ref_freques = cd.getIntRow()
    h = int_edges[1]-int_edges[0]
    func_arr = cd.getDistribFunc_int(int_edges, ref_freques)
    plt.figure(figsize=(14, 7))
    plt.title('Эмпирическая функция распределения')
    plt.xlabel('x, $')
    plt.ylabel('F(x)')
    plt.xlim(int_edges.min()-h//2, int_edges.max()+h//2)
    plt.ylim(-0.05, 1.05)
    plt.xticks(range(int_edges.min(), int_edges.max()+h//2, h // 2))
    plt.yticks([i/100 for i in range(-5, 105, 5)])
    plt.grid(True)
    plt.plot(func_arr[:, 1], func_arr[:, 0], marker='.')
    plt.savefig(r'graphics\distrib_func_int.svg')
    plt.savefig(r'graphics\distrib_func_int.jpg')

def drawHist():
    values, int_edges, _, _ = cd.getIntRow()
    h = int_edges[1]-int_edges[0]
    plt.figure(figsize=(10, 4.8))
    densities, _, _ = plt.hist(values, bins=int_edges, density=True, color='white', edgecolor='blue', linewidth=0.8)
    plt.title('Гистограмма относительных частот')
    plt.xlabel('x (курс биткойна), $')
    plt.ylabel('ω/h (плотность частости)')
    plt.xlim(int_edges.min()-h, int_edges.max()+h)
    plt.xticks(range(int_edges.min(), int_edges.max()+h, h))
    for i in range(len(densities)):
        plt.text(x=int_edges[i], y=densities[i]*1.01, s=f'{densities[i]:.4e}', fontsize='small')
    plt.savefig(r'graphics\hist.svg')
    plt.savefig(r'graphics\hist.jpg')

def comparePolygons():
    plt.figure(figsize=(10, 4.8))
    plt.title('Полигоны статистической и теоретической частостей')
    plt.xlabel('x')
    plt.ylabel('ω')
    plt.xlim(0, 900)
    plt.ylim(0, 0.3)
    plt.xticks(range(0, 900, 50))
    plt.grid(True)

    dis_row = cd.getDisRow()
    plt.plot(dis_row.loc['x'], dis_row.loc['w'], 'r.-', label='статистический полигон')

    theo_cords = cd.getPyasonDistrib()
    plt.plot(theo_cords[0], theo_cords[1], 'b.-', label='теоретический полигон')

    plt.legend()

    plt.savefig(r'graphics\comp_polygons41.svg')
    plt.savefig(r'graphics\comp_polygons41.jpg')

def compareHist():
    plt.figure(figsize=(10, 4.8))
    plt.xlabel('x')
    plt.ylabel('ω/f')
    plt.grid(True)

    values, int_edges, _, _ = cd.getIntRow()
    h = int_edges[1] - int_edges[0]
    plt.xlim(int_edges.min() - h, int_edges.max() + h)
    plt.xticks(range(int_edges.min(), int_edges.max() + h, h))
    densities, _, _ = plt.hist(values, bins=int_edges, density=True, color='white', alpha=1,
                               edgecolor='red', linewidth=3, label='гистограмма статистических частот')

    _, a, b = cd.getPlainDistrib()
    plt.plot([int_edges[0] - h, a, a, b, b, int_edges[-1]+h], [0, 0, 1/(b-a), 1/(b-a), 0, 0],
             'b.-', label='плотность равномерного распределения', alpha=0.5, linewidth=3)

    plt.legend()

    plt.savefig(r'graphics\comp_hist.svg')
    plt.savefig(r'graphics\comp_hist.jpg')

def drawPlainDistribFunc():
    _, int_edges, _, ref_freques = cd.getIntRow()
    h = int_edges[1] - int_edges[0]
    func_arr = cd.getDistribFunc_int(int_edges, ref_freques)

    plt.figure(figsize=(14, 7))
    plt.xlabel('x')
    plt.ylabel('F(x)')
    plt.xlim(int_edges.min() - h // 2, int_edges.max() + h // 2)
    plt.ylim(-0.05, 1.05)
    plt.xticks(range(int_edges.min(), int_edges.max() + h // 2, h // 2))
    plt.yticks([i / 100 for i in range(-5, 105, 5)])
    plt.grid(True)

    plt.plot(func_arr[:, 1], func_arr[:, 0], color='r', marker='.', label='эмпирическая ф-ция распределения')

    _, a, b = cd.getPlainDistrib()
    x = [int_edges[0] - h // 2, int_edges[0], int_edges[-1], int_edges[-1] + h // 2]
    y = [0, 0, 1, 1]
    plt.plot(x, y, marker='.', label='ф-ция равномерного распределения', alpha=0.5)

    plt.legend()

    plt.savefig(r'graphics\plain_func.svg')
    plt.savefig(r'graphics\plain_func.jpg')

def drawPyasonDistribFunc():
    plt.figure(figsize=(10, 4.8))
    plt.xlabel('x')
    plt.ylabel('F(x)')
    plt.xlim(-50, 900)
    plt.ylim(-0.05, 1.05)
    plt.xticks(range(0, 900, 50))
    plt.yticks([i / 100 for i in range(0, 105, 5)])
    plt.grid(True)

    func_arr = cd.getDistribFunc_dis(cd.getDisRow())
    plt.hlines(y=np.array(func_arr[:, 0], dtype=float), xmin=func_arr[:, 2], xmax=func_arr[:, 1], color='r',
               linewidths=1.5, label='эмпирическая ф-ция распределения')

    pyason_func_arr = cd.getPyasonDistribFunc()
    plt.hlines(y=np.array(pyason_func_arr[:, 0], dtype=float), xmin=pyason_func_arr[:, 2], xmax=pyason_func_arr[:, 1], color='b',
               linewidths=1.5, alpha=0.5, label='ф-ция распределения Пуассона')

    plt.legend(loc='lower right')

    plt.savefig(r'graphics\pyason_distrib_func70.svg')
    plt.savefig(r'graphics\pyason_distrib_func70.jpg')

drawPyasonDistribFunc()